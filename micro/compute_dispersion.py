import numpy as np


def breakthrough_properties(
    table,
    length: float,
) -> dict[str, float]:
    """Calculate apparent longitudinal dispersion from a step breakthrough."""

    time = table["Time"].to_numpy(dtype=float)
    tracer = table["weightedAreaAverage(T)"].to_numpy(dtype=float)

    threshold = 1e-10
    arrival = np.flatnonzero(tracer > threshold)
    if len(arrival) == 0:
        raise ValueError(f"No breakthrough above threshold {threshold}.")

    start = arrival[0]
    t = time[start:]
    tracer = tracer[start:]

    plateau = 0.99

    # Normalized cumulative breakthrough.
    F = (tracer - threshold) / (plateau - threshold)
    F = np.maximum.accumulate(np.clip(F, 0.0, 1.0))

    F /= F[-1]

    # Temporal moments without explicitly differentiating F.
    dF = np.diff(F)
    t_mid = 0.5 * (t[:-1] + t[1:])

    positive = dF > 0
    weights = dF[positive]
    times = t_mid[positive]
    if len(weights) == 0:
        raise ValueError("Breakthrough curve has no positive increments.")

    weights /= weights.sum()

    mean_time = np.sum(weights * times)
    time_variance = np.sum(weights * (times - mean_time) ** 2)

    velocity = length / mean_time
    dispersion = length**2 * time_variance / (2.0 * mean_time**3)
    
    print("mean_time: ", mean_time, "time_variance: ", time_variance, "velocity: ", velocity)

    return dispersion
