from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp


class BreakthroughCurve:
    """Record and plot tracer breakthrough at the east outlet."""

    def __init__(
        self,
        model: Any,
        tracer_component: Any,
        path: Path,
        *,
        subdomain: pp.Grid | None = None,
    ) -> None:
        self._model = model
        self._tracer_component = tracer_component
        self._outlet_subdomain = subdomain
        self._outlet_faces = np.flatnonzero(
            model.domain_boundary_sides(self._outlet_subdomain).east
        )

        self.times: list[float] = []
        self.fluid_mass_rates: list[float] = []
        self.tracer_mass_rates: list[float] = []
        self.tracer_fractions: list[float] = []

        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            [
                "time_s",
                "tracer_fraction_flux_averaged",
            ]
        )
        self._file.flush()

    def register_outlet_flux(self) -> None:
        """Evaluate and store east-outlet fluxes at the current time step."""
        fluid_flux = np.asarray(
            self._model.equation_system.evaluate(
                self._model.fluid_flux([self._outlet_subdomain])
            ),
            dtype=float,
        )
        tracer_flux = np.asarray(
            self._model.equation_system.evaluate(
                self._model.component_flux(
                    self._tracer_component, [self._outlet_subdomain]
                )
            ),
            dtype=float,
        )

        fluid_rate = float(np.sum(fluid_flux[self._outlet_faces]))
        tracer_rate = float(np.sum(tracer_flux[self._outlet_faces]))
        tracer_fraction = (
            tracer_rate / fluid_rate if fluid_rate > 0.0 else np.nan
        )

        time = float(self._model.time_manager.time)
        self.times.append(time)
        self.tracer_fractions.append(tracer_fraction)
        self._writer.writerow([time, tracer_fraction])
        self._file.flush()

    def close(self) -> None:
        self._file.close()

class BreakthroughCurve2D(BreakthroughCurve):
    """Record the east-outlet breakthrough curve for a 2D fracture."""

    def __init__(
        self,
        model: Any,
        tracer_component: Any,
        path: Path,
        *,
        subdomain: pp.Grid | None = None,
    ) -> None:
        if subdomain is None:
            subdomain = model.mdg.subdomains(dim=2)[0]
        if subdomain.dim != 2:
            raise ValueError("BreakthroughCurve2D requires a 2D subdomain.")
        super().__init__(
            model,
            tracer_component,
            path,
            subdomain=subdomain,
        )


class OutletFluxPressureCurve:
    """Write and plot BC pressure difference and total east-outlet fluid flux."""

    def __init__(
        self,
        model: Any,
        path: Path,
        *,
        subdomain: pp.Grid | None = None,
    ) -> None:
        self._model = model
        self._subdomain = subdomain
        self._outlet_faces = np.flatnonzero(
            model.domain_boundary_sides(self._subdomain).east
        )

        self._boundary_grid = model.mdg.subdomain_to_boundary_grid(self._subdomain)

        self.data: dict[str, list[float]] = {
            "time": [],
            "bc_pressure_difference": [],
            "outlet_fluid_flux": [],
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.data.keys())
        self._file.flush()

    def _pressure_difference_from_bc(self) -> float:
        boundary_values = np.asarray(
            self._model.bc_values_pressure(self._boundary_grid), dtype=float
        )
        sides = self._model.domain_boundary_sides(self._boundary_grid)

        west_pressure = float(np.mean(boundary_values[sides.west]))
        east_pressure = float(np.mean(boundary_values[sides.east]))
        return west_pressure - east_pressure

    def register(self) -> None:
        fluid_flux = np.asarray(
            self._model.equation_system.evaluate(
                self._model.fluid_flux([self._subdomain])
            ),
            dtype=float,
        ).reshape(-1)

        time = float(self._model.time_manager.time)
        pressure_difference = self._pressure_difference_from_bc()
        total_outlet_flux = float(np.sum(fluid_flux[self._outlet_faces]))

        self.data["time"].append(time)
        self.data["bc_pressure_difference"].append(pressure_difference)
        self.data["outlet_fluid_flux"].append(total_outlet_flux)
        self._writer.writerow([time, pressure_difference, total_outlet_flux])
        self._file.flush()

    def close(self) -> None:
        self._file.close()