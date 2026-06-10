from __future__ import annotations

from typing import Sequence, Union, cast

import numpy as np
import porepy as pp
import scipy.sparse as sps

from shared_coupling import pressure_gradient_matrix

class LinearProblemMixin:
    def _is_nonlinear_problem(self) -> bool:
        return False

class FaceTransmissibilityFluxMixin:
    """Replace selected face fluxes by a pressure-gradient transmissibility law."""

    face_transmissibility_cpl = "micro_face_transmissibility"
    face_transmissibility_no_cpl = "micro_face_transmissibility_mask"

    def _pressure_gradient_matrix(self, domains: list[pp.Grid]) -> sps.csr_matrix:
        matrices = [pressure_gradient_matrix(sd) for sd in domains]
        if len(matrices) == 0:
            return sps.csr_matrix((0, 0))
        return sps.block_diag(matrices, format="csr")

    @pp.ad.cached_method
    def darcy_flux(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        if len(domains) == 0 or all(
            isinstance(g, pp.BoundaryGrid) for g in domains
        ):
            return self.create_boundary_operator(
                name=self.bc_data_darcy_flux_key,
                domains=cast(Sequence[pp.BoundaryGrid], domains),
            )

        if not all(isinstance(g, pp.Grid) for g in domains):
            raise ValueError(
                "domains should either be grids or boundary grids."
            )

        domains = cast(list[pp.Grid], domains)

        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(domains, [1])
        intf_projection = pp.ad.MortarProjections(self.mdg, domains, interfaces, dim=1)

        boundary_operator = self.combine_boundary_operators_darcy_flux(
            subdomains=domains
        )

        discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.darcy_flux_discretization(
            domains
        )
        porepy_flux: pp.ad.Operator = (
            discr.flux() @ self.pressure(domains)
            + discr.bound_flux()
            @ (
                boundary_operator
                + intf_projection.mortar_to_primary_int()
                @ self.interface_darcy_flux(interfaces)
            )
            + discr.vector_source() @ self.vector_source_darcy_flux(domains)
        )

        grad = pp.ad.SparseArray(
            self._pressure_gradient_matrix(domains),
            name="pressure_gradient_matrix",
        ) @ self.pressure(domains)
        transmissibility = pp.ad.TimeDependentDenseArray(
            name=self.face_transmissibility_cpl,
            domains=domains,
        )
        mask = pp.ad.TimeDependentDenseArray(
            name=self.face_transmissibility_no_cpl,
            domains=domains,
        )

        flux: pp.ad.Operator = porepy_flux + mask * (
            transmissibility * grad - porepy_flux
        )
        flux.set_name("Darcy_flux")
        return flux
