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
    threshold = 1e-10
    full_threshold = 1.0

    time = table["Time"].to_numpy(dtype=float)
    tracer = table["weightedAreaAverage(T)"].to_numpy(dtype=float)

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

def breakthrough_properties_multisample(
    dir_name,
) -> float:
    """Calculate apparent longitudinal dispersion from a step breakthrough."""

    lengths = [0.04, 0.037, 0.034, 0.031, 0.028]
    folders = ["outletFlux", "C_x1", "C_x2", "C_x3", "C_x4"]

    threshold = 1e-10
    full_threshold = 1.0

    for i in range(len(lengths)):
        file = functionobject(file_name="surfaceFieldValue.dat", folder=folders[i])
        table = load_tables(source=file, dir_name=dir_name)

        time = table["Time"].to_numpy(dtype=float)
        tracer = table["weightedAreaAverage(T)"].to_numpy(dtype=float)

        # Filter out time before any breakthrough.
        arrival = np.flatnonzero(tracer > threshold)
        if len(arrival) == 0:
            raise ValueError(f"No breakthrough above threshold {threshold} in {folders[i]}.")

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

        if len(t) < 2:
            raise ValueError(f"Too few points after filtering in {folders[i]}.")

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

        if len(weights) == 0 or weights.sum() <= 0:
            raise ValueError(f"No positive breakthrough increments in {folders[i]}.")

        weights /= weights.sum()

        mean_time = np.sum(weights * times)
        time_variance = np.sum(weights * (times - mean_time) ** 2)

        velocity = lengths[i] / mean_time
        dispersion = lengths[i]**2 * time_variance / (2.0 * mean_time**3)

        print(
            folders[i],
            "mean_time: ", mean_time,
            "time_variance: ", time_variance,
            "velocity: ", velocity,
            "dispersion: ", dispersion,
            "plateau: ", plateau
        )

    return dispersion


if __name__ == "__main__":
    breakthrough_properties(".")
