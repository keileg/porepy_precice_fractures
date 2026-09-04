from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import porepy as pp


class BreakthroughCurveWriter:
    """Write tracer breakthrough at the matrix outlet."""

    def __init__(self, model: Any, tracer_component: Any, path: Path) -> None:
        self._model = model
        self._tracer_component = tracer_component

        self._outlet_subdomain = model.mdg.subdomains(dim=3)[0]
        self._outlet_faces = np.flatnonzero(
            model.domain_boundary_sides(self._outlet_subdomain).east
        )
        self._outlet_signs = np.asarray(
            self._outlet_subdomain.cell_faces[self._outlet_faces, :].sum(axis=1)
        ).ravel()

        self._file = path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["time_s", "tracer_fraction_flux_averaged"])
        self._file.flush()

    def write_row(self) -> None:
        fluid_flux = self._model.equation_system.evaluate(
            self._model.fluid_flux([self._outlet_subdomain])
        )
        tracer_flux = self._model.equation_system.evaluate(
            self._model.component_flux(
                self._tracer_component, [self._outlet_subdomain]
            )
        )

        outward_fluid_flux = self._outlet_signs * fluid_flux[self._outlet_faces]
        outward_tracer_flux = self._outlet_signs * tracer_flux[self._outlet_faces]

        outflow = outward_fluid_flux > 0.0
        fluid_mass_rate = float(np.sum(outward_fluid_flux[outflow]))
        tracer_mass_rate = float(np.sum(outward_tracer_flux[outflow]))
        tracer_fraction = (
            tracer_mass_rate / fluid_mass_rate if fluid_mass_rate > 0.0 else np.nan
        )

        self._writer.writerow([self._model.time_manager.time, tracer_fraction])
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def write_midline_pressure_profile_x(model: Any, path: Path) -> None:
    """Write matrix pressure along an x-directed line through its y-z midpoint."""
    matrix = model.mdg.subdomains(dim=3)[0]
    cell_centers = np.asarray(matrix.cell_centers)
    pressure = np.asarray(
        model.equation_system.evaluate(model.pressure([matrix])), dtype=float
    ).reshape(-1)

    x_coordinates = cell_centers[0]
    y_coordinates = cell_centers[1]
    z_coordinates = cell_centers[2]
    y_midpoint = 0.5 * (np.min(y_coordinates) + np.max(y_coordinates))
    z_midpoint = 0.5 * (np.min(z_coordinates) + np.max(z_coordinates))
    y_distance = np.abs(y_coordinates - y_midpoint)
    z_distance = np.abs(z_coordinates - z_midpoint)
    midpoint_cells = np.isclose(
        y_distance,
        np.min(y_distance),
        rtol=0.0,
        atol=1e-12,
    ) & np.isclose(
        z_distance,
        np.min(z_distance),
        rtol=0.0,
        atol=1e-12,
    )

    profile_x = x_coordinates[midpoint_cells]
    profile_pressure = pressure[midpoint_cells]

    with path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["time_s", "x_m", "pressure_pa"])
        for x_value in np.unique(profile_x):
            same_x = np.isclose(profile_x, x_value, rtol=0.0, atol=1e-12)
            writer.writerow(
                [
                    model.time_manager.time,
                    x_value,
                    float(np.mean(profile_pressure[same_x])),
                ]
            )


def write_time_step_outputs(
    model: Any,
    tracer_component: Any,
    breakthrough_curve: BreakthroughCurveWriter,
    pressure_profile_path: Path,
) -> None:
    """Write all visualization and diagnostic output for an accepted time step."""
    model.save_data_time_step()
    breakthrough_curve.write_row()
    write_midline_pressure_profile_x(model, pressure_profile_path)