from __future__ import annotations

import numpy as np
from foamlib.postprocessing.load_tables import functionobject, load_tables


def breakthrough_properties(
    dir_name,
    length: float,
    folder: str,
) -> float:
    """Calculate apparent longitudinal dispersion from a step breakthrough."""

    file = functionobject(file_name="surfaceFieldValue.dat", folder=folder)
    table = load_tables(source=file, dir_name=dir_name)
    threshold = 0.0001
    full_threshold = 1.0

    time = table["Time"].to_numpy(dtype=float)
    tracer = table["weightedAverage(T)"].to_numpy(dtype=float)

    # Filter out time before any breakthrough.
    arrival = np.flatnonzero(tracer > threshold)
    if len(arrival) == 0:
        raise ValueError(f"No breakthrough above threshold.")

    start = arrival[0]

    # Filter out time after the curve has fully reached.
    reached = np.flatnonzero(tracer >= full_threshold)
    reached = reached[reached >= start]

    if len(reached) > 0:
        end = reached[0] + 1   # include first fully reached point
    else:
        end = len(tracer)

    t = time[start:end]
    tracer = tracer[start:end]

    plateau = tracer[-1]

    # Normalized cumulative breakthrough.
    F = (tracer - threshold) / (plateau - threshold)
    F = np.maximum.accumulate(np.clip(F, 0.0, 1.0))

    # Temporal moments without explicitly differentiating F.
    dF = np.diff(F)
    t_mid = 0.5 * (t[:-1] + t[1:])

    positive = dF > 0
    weights = dF[positive]
    times = t_mid[positive]

    weights /= weights.sum()

    mean_time = np.sum(weights * times)
    time_variance = np.sum(weights * (times - mean_time) ** 2)

    velocity = length / mean_time
    dispersion = length**2 * time_variance / (2.0 * mean_time**3)
    
    print("mean_time: ", mean_time, "time_variance: ", time_variance, "velocity: ", velocity, "dispersion: ", dispersion)

    return dispersion


if __name__ == "__main__":
    breakthrough_properties(".")
