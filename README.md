[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Kabantuhi/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Kabantuhi/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

A make.fun "Forecast how many GTA VI players earn each trophy in week one" entry. It forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, the percent of the players Exophase tracks who will have earned it at 2026-11-26T00:00:00Z, as five quantiles (5%, 25%, 50%, 75%, 95%).

<sub>Entered on [make.fun](https://make.fun/?ref=c45jj78w&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

## How it works

`main.py` (Python 3.13, standard library only) reads a case file — every valid read the house made of Exophase's trophy page before the checkpoint — and writes one forecast file per case. It treats each trophy's earn rate as earners over players tracked, `r = E / N`:

- **The denominator is fit once, globally.** `log N` is a concave quadratic in `u = ln(1 + hours since release)`: the launch spike saturates, and a straight line in log-log would keep extrapolating it. If the fit wants to be convex (growth still accelerating), its linear part is used instead.
- **Each trophy's earners are fit separately.** `log(E + 0.5) = a + b·u + c·u²` (concave: earners grow fast while the early cohort progresses, then decelerate), Poisson-weighted, with the slope shrunk toward a class prior — story trophies earn much faster than the player base grows, grinds much slower. The intercept is refit at the shrunk slope. Trophies are classed from their tier and description (story / side / grind / platinum).
- **The forecast rate is `E_target / N_target`**, capped by loose class ceilings. The platinum is anchored at ~0.2% (its first earners are extreme players who do not show in the history yet); grinds get a small upward shift (they accelerate late, once players finish the story). The trophy at 100% in the last read ("Just Like Paradise", earned on booting the game) stays ~100%.
- **Quantiles sit on the logit scale around the median.** The spread combines the fit residual, the extrapolation distance, and a floor calibrated on simulated launch weeks.
- **Flat history fallback.** With fewer than three distinct reads (launch night, before Exophase's first refresh), the forecast is a diluted last rate with a wide spread — the first refresh dilutes every rate as the tracked base grows.

## Verification

- Run on the kit's two practice cases: valid output, byte-identical across runs, ~0.05 s per case (limit 8 s).
- Tuned on a simulator of the launch week (player join process fit to the observed anchors: 638 tracked at launch midnight, 3,868 at the 04:30Z refresh; live page rates at ~11 h; ~17 M pre-orders), across 11 scenarios × 3 seeds (33 simulated finals): mean pinball loss **0.76**, against **2.11** for the house baseline and **2.40** for the house curve baseline — beating the best house entry in every simulated final, with ~0.88 90% coverage.

## Run it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=path/to/cases MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Each `<id>.in` in the input directory produces one `<id>.out` in the output directory: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## License

MIT — see [LICENSE](LICENSE). SPDX-License-Identifier: MIT.
