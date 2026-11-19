"""gta6-trophy-rarity-forecast-01: forecast each GTA VI PS5 trophy's earn rate
on Exophase at the case's target, as five quantiles.

Model (see RUN-NOTES.md for the derivation and the simulation evidence):
each trophy's rate is earners over players tracked, r = E / N.
  - The denominator N is fit once, globally, as a concave quadratic in
    u = ln(1 + hours since release): the launch spike saturates, and a
    straight line in log-log would keep extrapolating it.
  - Each trophy's earners E are fit as log(E + 0.5) = a + b*u + c*u^2
    (concave: earners grow fast while the early cohort progresses, then
    decelerate), Poisson-weighted, with the slope shrunk toward a class
    prior (story trophies earn much faster than the player base grows;
    grinds earn slower). The intercept is refit at the shrunk slope.
  - The forecast rate is E_target / N_target, capped by class ceilings.
  - Quantiles sit on the logit scale around the median; the spread combines
    the fit residual, the extrapolation distance and a floor calibrated on
    simulated launch weeks (90% coverage ~0.88 there).
  - The trophy at 100% in the last read ("Just Like Paradise", earned on
    booting the game) stays ~100% with a tight spread. With fewer than
    three distinct reads the history carries no trend, so the forecast falls
    back to a diluted last rate (the first refresh after launch dilutes
    every rate as the tracked base grows).

Python 3.13, standard library only, no network, deterministic.
Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input) and writes
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import datetime
import json
import math
import os
import sys
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Spread calibration (tuned on simulated launch weeks; see RUN-NOTES.md).
SPREAD_FLOOR = 1.3
S_EXT_COEF = 0.5
SHRINK_W = 6.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(x):
    x = clamp_pct(x)
    return math.log(x / (100.0 - x))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (as JavaScript's Math.round does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def _solve3(a, b):
    """Solve a 3x3 linear system by Gaussian elimination."""
    m = [row[:] + [bb] for row, bb in zip(a, b)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise ValueError("singular")
        m[col], m[piv] = m[piv], m[col]
        for r in range(3):
            if r != col:
                f = m[r][col] / m[col][col]
                for c in range(col, 4):
                    m[r][c] -= f * m[col][c]
    return [m[i][3] / m[i][i] for i in range(3)]


def wls(xs, ys, ws):
    """Weighted least squares for y = a + b*x. Returns (a, b, resid_std)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for _, y, w in zip(xs, ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(w * r * r for r, w in zip(resid, ws)) / max(sw, 1e-9)
    dof = max(len(xs) - 2, 1)
    return a, b, math.sqrt(var * len(xs) / dof)


def wquad(xs, ys, ws):
    """Weighted least squares for y = a + b*x + c*x^2.
    Returns (a, b, c, resid_std) or None."""
    if len(xs) < 4:
        return None
    sw = sx = sx2 = sx3 = sx4 = sy = sxy = sx2y = 0.0
    for x, y, w in zip(xs, ys, ws):
        x2 = x * x
        sw += w
        sx += w * x
        sx2 += w * x2
        sx3 += w * x2 * x
        sx4 += w * x2 * x2
        sy += w * y
        sxy += w * x * y
        sx2y += w * x2 * y
    a = [[sw, sx, sx2], [sx, sx2, sx3], [sx2, sx3, sx4]]
    b = [sy, sxy, sx2y]
    try:
        coef = _solve3(a, b)
    except Exception:
        return None
    resid = [y - (coef[0] + coef[1] * x + coef[2] * x * x)
             for x, y in zip(xs, ys)]
    var = sum(w * r * r for r, w in zip(resid, ws)) / max(sw, 1e-9)
    dof = max(len(xs) - 3, 1)
    return coef[0], coef[1], coef[2], math.sqrt(var * len(xs) / dof)


def classify(trophy):
    """Class of a trophy from its tier + description.
    Returns (cls, beta_prior, cap_pct): the prior slope of log-earners
    against u = ln(1 + hours since release), and a plausibility ceiling
    for the rate at the target (a loose safety net, not the forecast)."""
    name = trophy.get("name", "")
    desc = trophy.get("description", "")
    tier = trophy.get("tier", "bronze")
    text = (name + " " + desc).lower()
    if tier == "platinum":
        return ("platinum", 0.5, 2.0)
    if "100% completion" in text:
        return ("grind", 0.6, 2.0)
    if "chapter" in text or "final mission" in text:
        return ("story", 2.0, 80.0)
    grind_words = ("stunts", "challenges", "collectible card", "waink",
                   "strands for strangers", "meditate", "laundering")
    if any(w in text for w in grind_words):
        return ("grind", 0.7, 15.0)
    if tier == "gold":
        return ("story", 1.6, 25.0)
    if tier == "silver":
        return ("side", 1.0, 20.0)
    # Easy side activities players hit naturally while playing (fishing,
    # an arcade game, discovering vehicles, a cocktail, bowling): these can
    # reach a large share in week one, so the cap is a loose net only.
    return ("side", 1.0, 60.0)


def distinct_snapshots(history):
    """Collapse consecutive identical reads (Exophase refresh cadence:
    the page is re-read often but its counts only move when Exophase
    refreshes them)."""
    out = []
    prev = None
    for h in history:
        key = (h["players_tracked"], tuple(h["rates"]))
        if key != prev:
            out.append(h)
            prev = key
    return out


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    history = distinct_snapshots(case["history"])
    horizon_days = case["horizon_minutes"] / 1440.0
    rel = datetime.datetime.fromisoformat(
        case["game"]["release_date"] + "T00:00:00+00:00")

    def u_of(at_iso):
        t = datetime.datetime.fromisoformat(at_iso.replace("Z", "+00:00"))
        return math.log1p(max(0.0, (t - rel).total_seconds() / 3600.0))

    us = [u_of(snap["at"]) for snap in history]
    lns = [math.log(max(1.0, snap["players_tracked"])) for snap in history]
    n_last = history[-1]["players_tracked"]
    u_last = us[-1]
    u_tgt = u_of(case["target"])

    # --- global N fit: log N concave quadratic in u (else linear).
    logn_tgt = None
    if len(us) >= 4:
        quad = wquad(us, lns, [1.0] * len(us))
        if quad is not None:
            if quad[2] <= 0:
                logn_tgt = quad[0] + quad[1] * u_tgt + quad[2] * u_tgt * u_tgt
            else:
                # Growth still accelerating in log-log: use the quadratic's
                # linear part (it tracks the recent growth better than a
                # fresh straight-line fit over the whole history).
                logn_tgt = quad[0] + quad[1] * u_tgt
    if logn_tgt is None:
        fit = wls(us, lns, [1.0] * len(us))
        logn_tgt = fit[0] + fit[1] * u_tgt if fit else math.log(n_last)
    n_tgt = min(max(math.exp(logn_tgt), n_last), n_last * 8.0)

    last_rates = history[-1]["rates"]
    flat = len(history) < 3
    rows = []
    for i, tr in enumerate(trophies):
        last = last_rates[i]
        if last is None:
            last = 0.0
        # The boot trophy: earned by everyone, stays ~100%.
        if last >= 99.5:
            med, s = 99.9, 0.02
            rows.append({"id": tr["id"],
                         "q": [round4(100 * sigmoid(logit_pct(med) + z * s))
                               for z in Z]})
            continue
        cls, beta0, cap = classify(tr)
        if flat:
            # No trend in the history (launch night, before the first
            # refresh): the first refresh dilutes every rate as the tracked
            # base grows, so forecast well below the last rate, wide.
            med = clamp_pct(last * 0.35 if last > 0 else 0.01)
            if cls == "platinum":
                med = min(max(med, 0.15), 1.5)
            s = 0.5 * math.sqrt(horizon_days) + 0.4
            rows.append({"id": tr["id"],
                         "q": [round4(100 * sigmoid(logit_pct(med) + z * s))
                               for z in Z]})
            continue
        xs, ys, ws = [], [], []
        for snap, u in zip(history, us):
            r = snap["rates"][i]
            if r is None:
                continue
            e = r * snap["players_tracked"] / 100.0
            xs.append(u)
            ys.append(math.log(e + 0.5))
            ws.append(e + 0.5)
        fit = wls(xs, ys, ws)
        if fit is None:
            med = clamp_pct(last)
            s = 0.5 * math.sqrt(horizon_days) + 0.4
        else:
            alpha, beta, resid = fit
            mean_x = sum(xs) / len(xs)
            w_data = sum(ws) * max(
                1e-9, sum((x - mean_x) ** 2 for x in xs) / len(xs))
            sw = sum(ws)
            quad = wquad(xs, ys, ws) if len(xs) >= 5 else None
            if quad is not None and quad[2] <= 0:
                _aq, bq, cq, resid_q = quad
                bq_s = (w_data * bq + SHRINK_W * beta0) / (w_data + SHRINK_W)
                aq_s = sum(w * (y - bq_s * x - cq * x * x)
                           for x, y, w in zip(xs, ys, ws)) / sw
                loge_tgt = aq_s + bq_s * u_tgt + cq * u_tgt * u_tgt
                resid = resid_q
            else:
                beta_s = (w_data * beta + SHRINK_W * beta0) / (w_data + SHRINK_W)
                alpha_s = sum(w * (y - beta_s * x)
                              for x, y, w in zip(xs, ys, ws)) / sw
                loge_tgt = alpha_s + beta_s * u_tgt
            e_tgt = math.exp(loge_tgt)
            med = clamp_pct(100.0 * e_tgt / n_tgt)
            med = min(med, cap)
            # Class adjustments: the platinum's first earners are extreme
            # players who do not show in the history yet, so anchor it at
            # ~0.2%; grinds accelerate late, once players finish the story.
            if cls == "platinum":
                med = min(max(med, 0.15), 1.5)
            elif cls == "grind":
                med = clamp_pct(100 * sigmoid(logit_pct(med) + 0.5))
                med = min(med, cap)
            s_fit = min(max(resid, 0.15), 0.8)
            s_ext = S_EXT_COEF * max(0.0, u_tgt - u_last)
            s = math.sqrt(s_fit ** 2 + s_ext ** 2 + SPREAD_FLOOR ** 2)
        row = [round4(100 * sigmoid(logit_pct(med) + z * s)) for z in Z]
        for k in range(1, 5):
            if row[k] < row[k - 1]:
                row[k] = row[k - 1]
        rows.append({"id": tr["id"], "q": row})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
