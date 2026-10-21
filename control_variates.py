"""EXERCISE 2: CONTROL VARIATES FOR PRICES AND GAMMA.

Here each array has shape (number_of_paths,). Entry i in target and control
uses the same path randomness; the valuation layer constructs these pairs.
"""
import numpy as np


# STEP 1: CALCULATE BETA FROM THE PAIRED TRAINING OBSERVATIONS
def estimate_control_coefficient(target_samples, control_samples):
    target_samples = np.asarray(target_samples, dtype=float)
    control_samples = np.asarray(control_samples, dtype=float)

    # Here each observation must be paired with exactly one control observation.
    if target_samples.ndim != 1 or control_samples.ndim != 1:
        raise ValueError("Target and control must be one-dimensional arrays")
    if target_samples.shape != control_samples.shape:
        raise ValueError("Target and control must contain the same number of observations")
    number_of_observations = target_samples.size
    # Here sample covariance and variance use n-1, so we need at least two pairs.
    if number_of_observations < 2:
        raise ValueError("At least two paired training observations are required")
    # Here one NaN or infinity would invalidate the covariance and beta.
    if not (np.all(np.isfinite(target_samples))
            and np.all(np.isfinite(control_samples))):
        raise ValueError("Training observations must be finite")

    # Here a constant control has zero variance: there is no variation to remove.
    # Beta=0 leaves the plain estimator unchanged and avoids division by zero.
    if np.all(control_samples == control_samples[0]):
        return 0.0

    target_deviations = target_samples - np.mean(target_samples)
    control_deviations = control_samples - np.mean(control_samples)
    sample_covariance = np.sum(target_deviations * control_deviations) / (
        number_of_observations - 1)
    control_variance = np.sum(control_deviations**2) / (number_of_observations - 1)

    # Here beta = Cov(X,Y)/Var(Y) minimises the training adjusted-sample variance.
    # Its actual benefit is then measured on independent evaluation observations.
    beta = sample_covariance / control_variance
    return float(beta)


# STEP 2: APPLY THE BETA ALREADY CALCULATED FROM TRAINING
def apply_control_variate(
    target_samples, control_samples, known_control_mean, coefficient
):
    target_samples = np.asarray(target_samples, dtype=float)
    control_samples = np.asarray(control_samples, dtype=float)

    # Here matching one-dimensional arrays preserve the observation-by-observation pairs.
    if target_samples.ndim != 1 or control_samples.ndim != 1:
        raise ValueError("Target and control must be one-dimensional arrays")
    if target_samples.shape != control_samples.shape:
        raise ValueError("Target and control must contain the same number of observations")
    # Here all terms in the adjustment must be finite numerical values.
    if not (np.all(np.isfinite(target_samples))
            and np.all(np.isfinite(control_samples))):
        raise ValueError("Evaluation samples must be finite")
    if not (np.isfinite(known_control_mean) and np.isfinite(coefficient)):
        raise ValueError("Known control mean and coefficient must be finite")

    # Here Z_i = X_i - beta*(Y_i - E[Y]). We use the known expectation,
    # not the sample mean, and leave the training coefficient unchanged.
    beta = coefficient
    control_deviation_from_known_mean = control_samples - known_control_mean
    controlled_samples = target_samples - beta * control_deviation_from_known_mean

    # Here we return each adjusted observation; valuation calculates mean and variance.
    return controlled_samples
