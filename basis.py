"""EXERCISE 1 REQUIREMENT: CHOOSE AND JUSTIFY THE REGRESSION BASIS.

Here we choose a quadratic, following the worked example in Longstaff–Schwartz
Section 1. Section 2.2 motivates additional state variables and cross-products;
Section 8.3 discusses scaling. This is our basis choice, not a claim to reproduce
Table 1's weighted Laguerre basis. Degree two keeps the fit small and readable.
"""
import numpy as np


# SPOT BASIS — ONE ROW PER PATH, THREE COLUMNS
def spot_quadratic_basis(current_state, product):
    scaled_spot = np.asarray(current_state["spot"]) / product.strike
    if scaled_spot.ndim != 1 or not np.all(np.isfinite(scaled_spot)):
        raise ValueError("Basis inputs must be finite, one-dimensional spot states")
    # Here we use S/K so the scale is comparable across strikes.
    return np.column_stack((np.ones_like(scaled_spot), scaled_spot, scaled_spot**2))


# SPOT AND VARIANCE BASIS — ONE ROW PER PATH, SIX COLUMNS
def spot_variance_quadratic_basis(current_state, product, variance_scale):
    if not np.isfinite(variance_scale) or variance_scale <= 0:
        raise ValueError("Variance scale must be finite and positive")
    scaled_spot = np.asarray(current_state["spot"]) / product.strike
    scaled_variance = np.asarray(current_state["variance"]) / variance_scale
    if (scaled_spot.ndim != 1 or scaled_spot.shape != scaled_variance.shape
            or not np.all(np.isfinite(scaled_spot))
            or not np.all(np.isfinite(scaled_variance))):
        raise ValueError("Spot and variance must be finite vectors of matching length")
    # Here we include both Markov states. The cross-term allows their interaction.
    # Scaling variance by v0 is our fixed numerical choice, not a fitted parameter.
    return np.column_stack((np.ones_like(scaled_spot), scaled_spot, scaled_variance,
                            scaled_spot**2, scaled_spot * scaled_variance,
                            scaled_variance**2))


# EXERCISE 4 — SIX CUBIC B-SPLINES, TWO INTERIOR KNOTS
def gas_storage_spline_knots(log_mean, log_variance):
    from scipy.special import ndtri

    if not np.isfinite(log_mean) or not np.isfinite(log_variance) or log_variance <= 0:
        raise ValueError("Spline knots require a non-degenerate log-price distribution")
    # Here the paper specifies b6, but not its knots. We use model-implied
    # marginal price quantiles, fixed across path counts and independent runs.
    quantiles = np.array([0.0001, 1 / 3, 2 / 3, 0.9999])
    prices = np.exp(log_mean + np.sqrt(log_variance) * ndtri(quantiles))
    return np.concatenate((np.repeat(prices[0], 4), prices[1:3],
                           np.repeat(prices[-1], 4)))


def gas_storage_bspline_basis(spots, knots):
    from scipy.interpolate import BSpline

    spots = np.asarray(spots, dtype=float)
    knots = np.asarray(knots, dtype=float)
    if spots.ndim != 1 or not np.all(np.isfinite(spots)) or np.any(spots <= 0):
        raise ValueError("Gas-price basis inputs must be a finite positive vector")
    if knots.shape != (10,) or not np.all(np.diff(knots) >= 0) or knots[3] >= knots[6]:
        raise ValueError("Six cubic splines require ten ordered knots")
    # Here six basis functions already span a constant; no extra intercept.
    splines = BSpline(knots, np.eye(6), k=3, extrapolate=False)
    clipped_spots = np.clip(spots, knots[3], knots[6])
    design = splines(clipped_spots)
    # Here we extend each endpoint tangent linearly, in both fitting and replay.
    # This avoids unrestricted cubic extrapolation; tail approximation remains.
    outside = spots != clipped_spots
    if np.any(outside):
        design[outside] += ((spots[outside] - clipped_spots[outside])[:, None]
                            * splines.derivative()(clipped_spots[outside]))
    return design
