from __future__ import annotations

from typing import Sequence, cast

import numpy as np
import porepy as pp
from porepy.applications.material_values.fluid_values import water
from porepy.models.compositional_flow import (
    BoundaryConditionsMulticomponent,
    InitialConditionsFractions,
)


class ModifiedGeometry:
    mesh_size = 0.025
    fracture_points = np.array(
        [[0.1, 0.9, 0.9, 0.1], [0.1, 0.1, 0.9, 0.9], [0.2, 0.2, 0.2, 0.2]],
        dtype=float,
    )

    def set_domain(self) -> None:
        self._domain = pp.Domain(
            bounding_box={
                "xmin": 0.0,
                "xmax": 1.0,
                "ymin": 0.0,
                "ymax": 1.0,
                "zmin": 0.15,
                "zmax": 0.35,
            }
        )

    def set_fractures(self) -> None:
        frac_1_points = self.units.convert_units(self.fracture_points, "m")
        frac_1 = pp.PlaneFracture(frac_1_points)
        self._fractures = [frac_1]

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(self.mesh_size, "m")
        return {"cell_size": cell_size}


class ModifiedGeometry2D:
    mean_aperture = 0.001
    aperture_amplitude = 0.2  # Relative amplitude

    def set_geometry(self) -> None:
        bounds = {
            "xmin": 0.0,
            "xmax": 1.0,
            "ymin": 0.0,
            "ymax": 0.05,
        }

        self._domain = pp.Domain(bounding_box=bounds)

        grid = pp.CartGrid(
            nx=np.array([80, 1]),
            physdims=bounds,
        )

        self.mdg = pp.MixedDimensionalGrid()
        self.mdg.add_subdomains(grid)

        self.mdg.compute_geometry()
        self.mdg.set_boundary_grid_projections()

        self.nd = 2

        self._aperture_by_grid = {id(grid): self.grid_aperture(grid)}

    def grid_aperture(
        self,
        grid: pp.Grid,
    ) -> np.ndarray:

        y = grid.cell_centers[1, :]

        Ly = 1.0

        aperture = self.mean_aperture * (
            1.0 + self.aperture_amplitude * np.sin(np.pi * y / (Ly))
        )

        return aperture

    def specific_volume(
        self,
        grids: pp.GridLikeSequence,
    ) -> pp.ad.Operator:

        if len(grids) == 0:
            return pp.wrap_as_dense_ad_array(
                0.0,
                size=0,
            )

        if all(isinstance(g, pp.Grid) for g in grids):
            aperture = np.concatenate([self._aperture_by_grid[id(g)] for g in grids])

            return pp.wrap_as_dense_ad_array(
                aperture,
                size=aperture.size,
                name="fracture_specific_volume",
            )

        return super().specific_volume(grids)


class TracerFluid:
    def get_components(self) -> Sequence[pp.FluidComponent]:
        component_1 = pp.FluidComponent(**water)
        component_2 = pp.FluidComponent(name="tracer")
        return [component_1, component_2]


class TracerIC(InitialConditionsFractions):
    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        return self.reference_variable_values.pressure * np.ones(sd.num_cells)

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        assert component.name == "tracer", "Only the tracer is independent."
        return np.zeros(sd.num_cells)


class TracerBC(BoundaryConditionsMulticomponent):
    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign Dirichlet to the west and east boundaries. The rest are Neumann by
        default."""
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, domain_sides.west + domain_sides.east, "dir")
        return bc

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Zero bc value on top and bottom, p_l on west side, p_r on east side."""
        domain_sides = self.domain_boundary_sides(bg)
        values = np.zeros(bg.num_cells)
        # See section on scaling for explanation of the conversion.
        values[domain_sides.west] = self.units.convert_units(100, "Pa")
        values[domain_sides.east] = self.units.convert_units(0, "Pa")
        return values

    def bc_values_overall_fraction(
        self, component: pp.Component, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        """Inject tracer at the west inlet."""

        z = np.zeros(bg.num_cells)

        assert component.name == "tracer", "Only the tracer is independent."

        domain_sides = self.domain_boundary_sides(bg)
        if self.time_manager.time < 5:
            z[domain_sides.west] = (
                1.0  # * (bg.cell_centers[1, domain_sides.west] <= 0.55) * (bg.cell_centers[1, domain_sides.west] >= 0.45)
            )

        return z
