from __future__ import annotations

from typing import Sequence

import numpy as np
import porepy as pp
from porepy.applications.material_values.fluid_values import water
from porepy.models.compositional_flow import (
    BoundaryConditionsMulticomponent,
    InitialConditionsFractions,
)
from porepy.applications.md_grids.domains import nd_cube_domain

class ModifiedGeometry:
    mesh_size = 0.1
    fracture_points = np.array(
        [[0.2, 0.8, 0.8, 0.2], [0.2, 0.2, 0.8, 0.8], [0.5, 0.5, 0.5, 0.5],]
    )

    def set_domain(self) -> None:
        size = self.units.convert_units(1, "m")
        self._domain = nd_cube_domain(3, size)

    def set_fractures(self) -> None:
        frac_1_points = self.units.convert_units(self.fracture_points, "m")
        frac_1 = pp.PlaneFracture(frac_1_points)
        self._fractures = [frac_1]

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(self.mesh_size, "m")
        return {"cell_size": cell_size}

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
        values[domain_sides.west] = self.units.convert_units(10, "Pa")
        values[domain_sides.east] = self.units.convert_units(0, "Pa")
        return values

    def bc_values_overall_fraction(
        self, component: pp.Component, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        """Defines some non-trivial inflow of the tracer component on the inlet
        (north)."""

        z = np.zeros(bg.num_cells)

        assert component.name == "tracer", "Only the tracer is independent."

        # Set the tracer concentration to 0.2 on the left boundary
        domain_sides = self.domain_boundary_sides(bg)
        z[domain_sides.west] = 0.2 

        return z
