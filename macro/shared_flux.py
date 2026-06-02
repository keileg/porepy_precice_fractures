from __future__ import annotations

import numpy as np
import porepy as pp

class CubicLawPermeabilityModified(pp.constitutive_laws.CubicLawPermeability):
    """Modified cubic-law permeability for fractures.
    """
    def fracture_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        n_cells = sum(sd.num_cells for sd in subdomains)
        fracture_permeability = np.asarray(
            self.params.get("fracture_permeability", 1.0), dtype=float
        ).reshape(-1)

        if fracture_permeability.size == 1:
            fracture_permeability = np.full(n_cells, fracture_permeability.item())

        return self.isotropic_second_order_tensor(subdomains, fracture_permeability)
        # TODO align the permeability computed on each cell face to eahc cell
        # basis = self.basis(subdomains, 9)
        # diagonal_indices = [0, 4] # instead of [1,4,8] since normal direction doesn't have simulation
        # for c in n_cells:
        #     permeability = pp.ad.sum_operator_list(
        #         [basis[i] @ permeability[i + i % 3 * vertical_offset] for i in diagonal_indices]
        #     )

        # return permeability

class LinearProblemMixin:
    def _is_nonlinear_problem(self) -> bool:
        return False
