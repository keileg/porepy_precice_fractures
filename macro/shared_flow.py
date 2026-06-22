from __future__ import annotations

from typing import Sequence, cast

import numpy as np
import porepy as pp
import scipy.sparse as sps
from porepy.applications.material_values.fluid_values import water
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.models.compositional_flow import (BoundaryConditionsMulticomponent,
                                              InitialConditionsFractions)

from shared_coupling import pressure_gradient_matrix

class ModifiedGeometry:
    mesh_size = 0.1
    fracture_points = np.array(
        [[0.2, 0.4, 0.4, 0.2], [0.2, 0.2, 0.3, 0.3], [0.5, 0.5, 0.5, 0.5],]
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
        values[domain_sides.west] = self.units.convert_units(20, "Pa")
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

class FaceDispersionMixin:

    face_dispersion_cpl = "face_dispersion"
    face_dispersion_no_cpl = "face_dispersion_mask"

    def _tracer_fraction_gradient_matrix(
        self, domains: list[pp.Grid]
    ) -> sps.csr_matrix:
        matrices = [pressure_gradient_matrix(sd) for sd in domains]
        if len(matrices) == 0:
            return sps.csr_matrix((0, 0))
        return sps.block_diag(matrices, format="csr")

    def component_flux(
        self, component: pp.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        flux = super().component_flux(component, domains)

        if component.name != "tracer":
            return flux

        if len(domains) == 0 or all(isinstance(g, pp.BoundaryGrid) for g in domains):
            return flux

        if not all(isinstance(g, pp.Grid) for g in domains):
            raise ValueError("Domains must consist entirely of subdomains.")

        domains = cast(list[pp.Grid], domains)
        gradient = pp.ad.SparseArray(
            self._tracer_fraction_gradient_matrix(domains),
            name="tracer_fraction_gradient_matrix",
        ) @ component.fraction(domains)
        dispersion = pp.ad.TimeDependentDenseArray(
            name=self.face_dispersion_cpl,
            domains=domains,
        )
        mask = pp.ad.TimeDependentDenseArray(
            name=self.face_dispersion_no_cpl,
            domains=domains,
        )

        dispersive_flux = pp.ad.Scalar(-1.0) * mask * dispersion * gradient
        flux += dispersive_flux
        flux.set_name(f"component_flux_{component.name}_with_dispersion")
        return flux
