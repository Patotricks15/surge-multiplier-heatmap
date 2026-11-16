# surge-multiplier-heatmap (PR03)

**Question:** what surge multiplier should each zone get, and how is it spread across the city?

## Method

- Rule: `m = clip(1 + β · (D̂ / S − 1), 1, m_max)`, with **β = 0.5** and **m_max = 3.0**.
- `D̂` is the PR01 predicted pickups for a zone-hour, clipped at 0.
- `S` is the supply proxy: the zone's trailing 168-hour mean pickups (see [data/README.md](data/README.md)).
- **Gate:** cells with `S < 1` pickups/hour get `m = 1`. Their ratio is unstable, and there is no history to justify a surge.
- **Map:** the demand-weighted average of `m` is smoothed with the shared gaussian kernel. The numerator `D̂·m` and the denominator `D̂` are smoothed separately, then divided, so the result is an average rather than a sum.
- The forecast comes from PR01, so this project does not fit or tune any model.

## Results (test window, Sep 2014)

| Check | Value |
|---|---|
| Rows | 115,388 across 182 zones and 634 hours |
| Rows gated (`S < 1`, `m = 1`) | 52.6% |
| Rows with `m > 1` | 16.5% |
| Rows at the cap (`m = 3.0`) | 0.01% |
| Mean `m` | 1.047 |
| Zone mean `m`: median / 90th / max | 1.002 / 1.128 / 1.193 |
| Zones with mean `m > 1.5` | 0% |
| Spearman corr. (zone mean demand, zone mean `m`) | 0.92 |
| Busiest predicted hour | 2014-09-18 18:00, 2,350 predicted pickups, 0 zones at the cap |

The cap is enforced: no value is above 3.0. With β = 0.5, `m` reaches 3.0 only when `D̂/S` is 5 or more, and that happens in 0.01% of rows.

Figures are in `reports/figures/`:
- `pr03_multiplier_distribution.png`: row histogram of `m` (log scale, with the cap line), and each zone's mean `m` against its demand.
- `pr03_surge_heatmap_mean.png`: demand-weighted surge over the whole test window.
- `pr03_surge_heatmap_peak.png`: surge at the busiest predicted hour, the same hour RT04 uses.

## What the numbers say

- Surge is almost entirely in busy zones, and the strong demand correlation (0.92) is mostly caused by the gate. Low-demand zones have `S < 1`, so they cannot surge. The correlation does not show unfair pricing by itself. It shows that the proxy only surges where there is history.
- At the peak hour the highest multiplier is 2.3. The top cells are in Midtown, around 40.75 N, 73.98 W, where predicted demand is about 3.6 times the cell's trailing weekly mean.
- The mean surge is small (1.047). A likely reason is that `S` is a week-long average, so it follows `D̂` closely and the ratio stays near 1. This is not tested here.

## Limitations and deviations

- **No supply data.** The plan asked for `D̂ / S` with a real supply signal. None of the datasets records driver supply, so `S` is the trailing demand mean. The multiplier is a demand-spike indicator, not a supply-demand balance. This is the main limitation.
- **Surge is only applied where history exists.** The gate means 53% of rows are never surged. A supply signal would be needed to price the rest.
- **Single city and single month.** Only NYC, September 2014 is used.
- **Basemap.** Esri World Light Gray, because CartoDB needs an API key and OSM tiles block anonymous scripts. Attribution is shown on the maps.
- **No elasticity link.** The multiplier is not tested against rider response. [PR02](https://github.com/Patotricks15/price-elasticity-loglog) estimates elasticity on Boston quotes and is weakly identified, so the two projects are not joined.

## Reproduce

```bash
make install
make test
make notebook   # reads ~/uber-data/cache/pr01_predictions.csv.gz
```

Run [PR01](https://github.com/Patotricks15/geohash-demand-forecast-xgboost) first, since its notebook writes that cache.
