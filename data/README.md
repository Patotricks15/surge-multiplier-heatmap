# Data

**Source:** the PR01 demand forecast (`pr01_predictions.csv.gz`), built from the NYC TLC pickups (Kaggle "Uber Pickups in New York City", 2014). See [geohash-demand-forecast-xgboost](https://github.com/Patotricks15/geohash-demand-forecast-xgboost).
**Used here:** only the test-window rows (2014-09-04 to 2014-09-30), 115,388 rows, 182 zones, 634 hours.
**Not committed:** the cache is kept outside the repo, in `~/uber-data/cache/`. Set `UBER_DATA_DIR` to point the notebook at it.

## Columns used

- `geohash`, `lat`, `lon`, `hour`: zone and hour (geohash precision 6).
- `pred_count`: predicted pickups for the cell-hour, already back-transformed from log1p to counts. Negative values are clipped to 0.
- `trailing_week_mean`: mean hourly pickups in the same cell over the previous 168 hours (`shift(1)`, so no leakage). This is the supply proxy.

## Caveat on the supply proxy

The datasets contain **no driver supply**. `trailing_week_mean` is demand history, not supply. The multiplier therefore measures how far predicted demand is above the zone's recent average. It is a demand-spike indicator and should not be read as a supply-demand balance.
