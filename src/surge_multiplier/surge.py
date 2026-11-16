"""Surge multiplier from predicted demand and a supply proxy, plus a demand-weighted smoothed map."""

from __future__ import annotations

import numpy as np

from routes_pricing.plots import smoothed_grid


def surge_multiplier(demand, supply, beta=0.5, m_max=3.0, min_supply=1.0):
    """Return `clip(1 + beta * (demand / supply - 1), 1, m_max)` element-wise.

    Negative forecasts are treated as zero demand. Where `supply < min_supply`
    the ratio is unstable, so the multiplier is 1 (no surge without history).
    """
    demand = np.clip(np.asarray(demand, dtype=float), 0.0, None)
    supply = np.asarray(supply, dtype=float)
    ratio = np.divide(demand, supply, out=np.ones_like(demand), where=supply >= min_supply)
    return np.clip(1.0 + beta * (ratio - 1.0), 1.0, m_max)


def demand_weighted_grid(lat, lon, multiplier, demand, bbox, bins=120, sigma=1.5):
    """Demand-weighted average multiplier on a grid, smoothed with a gaussian kernel.

    Numerator and denominator are smoothed with the same kernel and then divided,
    so the output is an average multiplier rather than a sum. Cells with no demand are NaN.
    """
    demand = np.clip(np.asarray(demand, dtype=float), 0.0, None)
    multiplier = np.asarray(multiplier, dtype=float)
    num, extent = smoothed_grid(lat, lon, demand * multiplier, bins=bins, sigma=sigma, bbox=bbox)
    den, _ = smoothed_grid(lat, lon, demand, bins=bins, sigma=sigma, bbox=bbox)
    grid = np.full_like(num, np.nan)
    np.divide(num, den, out=grid, where=den > 1e-9)
    return grid, extent
