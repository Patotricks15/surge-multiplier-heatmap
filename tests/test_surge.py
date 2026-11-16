import numpy as np

from surge_multiplier.surge import demand_weighted_grid, surge_multiplier


def test_balanced_demand_and_supply_gives_no_surge():
    assert surge_multiplier([10.0], [10.0])[0] == 1.0


def test_multiplier_is_capped_and_never_below_one():
    m = surge_multiplier([1000.0, 0.0], [1.0, 1.0], beta=0.5, m_max=3.0)
    assert m[0] == 3.0
    assert m[1] == 1.0


def test_cells_without_supply_history_get_no_surge():
    assert surge_multiplier([50.0], [0.0])[0] == 1.0


def test_negative_forecast_is_treated_as_zero_demand():
    assert surge_multiplier([-0.5], [10.0])[0] == 1.0


def test_multiplier_is_monotone_in_demand():
    demand = np.linspace(0, 40, 50)
    m = surge_multiplier(demand, np.full(50, 10.0))
    assert np.all(np.diff(m) >= 0)


def test_weighted_grid_stays_within_input_range_and_marks_empty_cells():
    lat = np.array([40.70, 40.7001, 40.75])
    lon = np.array([-74.00, -74.0001, -73.95])
    m = np.array([1.0, 2.0, 3.0])
    d = np.array([5.0, 5.0, 0.0])
    bbox = (40.69, 40.76, -74.01, -73.94)
    grid, _ = demand_weighted_grid(lat, lon, m, d, bbox, bins=40, sigma=1.0)
    finite = grid[np.isfinite(grid)]
    assert finite.min() >= 1.0 - 1e-9 and finite.max() <= 3.0 + 1e-9
    assert np.isnan(grid).any()
