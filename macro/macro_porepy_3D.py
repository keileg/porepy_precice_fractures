from typing import Callable, Optional, Sequence, cast

import porepy as pp
import numpy as np
import precice
from porepy.models.fluid_mass_balance import SinglePhaseFlow, BoundaryConditionsSinglePhaseFlow, FluidMassBalanceEquations
from porepy.applications.md_grids.domains import nd_cube_domain


h = 0.1

class ModifiedGeometry:
    def set_domain(self) -> None:
        """Defining a three-dimensional cubic domain with sidelength 1."""
        size = self.units.convert_units(1, "m")
        self._domain = nd_cube_domain(3, size)

    def set_fractures(self) -> None:
        """Setting a diagonal fracture"""
        frac_1_points = self.units.convert_units(
            np.array([[0.4, 0.7, 0.7, 0.4], [0.4, 0.4, 0.7, 0.7], [0.5, 0.5, 0.5, 0.5]]), "m"
        )
        frac_1 = pp.PlaneFracture(frac_1_points)
        self._fractures = [frac_1]

    def grid_type(self) -> str:
        """Choosing the grid type for our domain.

        As we have a diagonal fracture we cannot use a Cartesian grid.
        Cartesian grid is the default grid type, and we therefore override this method
        to assign simplex instead.

        """
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        """Meshing arguments for md-grid creation.

        Here we determine the cell size.

        """
        cell_size = self.units.convert_units(h, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

class ModifiedSources(FluidMassBalanceEquations):
    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        internal_sources: pp.ad.Operator = super().fluid_source(subdomains)

        values = []
        for sd in subdomains:
            q = np.zeros(sd.num_cells)

            if sd.dim == 3:
                centers = sd.cell_centers.T

                source_pos = np.array([0.2, 0.5, 0.8])
                sink_pos   = np.array([0.8, 0.5, 0.2])

                source_cell = np.argmin(np.linalg.norm(centers - source_pos, axis=1))
                sink_cell   = np.argmin(np.linalg.norm(centers - sink_pos, axis=1))

                Q_mass = fluid_constants.density * 1.0e-5
                q[source_cell] = +Q_mass
                q[sink_cell]   = -Q_mass

            values.append(q)
        external_sources = pp.wrap_as_dense_ad_array(np.hstack(values))

        # Add up both contributions
        source = internal_sources + external_sources
        source.set_name("fluid sources")

        return source

class ModifiedDarcyFlux:
    def porepy_darcy_flux(
        self,
        domains: pp.SubdomainsOrBoundaries,
    ) -> pp.ad.Operator:
        domains = cast(list[pp.Grid], domains)

        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(domains, [1])
        intf_projection = pp.ad.MortarProjections(self.mdg, domains, interfaces, dim=1)

        boundary_operator = self.combine_boundary_operators_darcy_flux(
            subdomains=domains
        )

        discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.darcy_flux_discretization(domains)

        flux: pp.ad.Operator = (
            discr.flux() @ self.pressure(domains)
            + discr.bound_flux()
            @ (
                boundary_operator
                + intf_projection.mortar_to_primary_int()
                @ self.interface_darcy_flux(interfaces)
            )
            + discr.vector_source()
            @ self.vector_source_darcy_flux(domains)
        )
        flux.set_name("PorePy_Darcy_flux")
        return flux

    def darcy_flux(
        self,
        domains: pp.SubdomainsOrBoundaries,
    ) -> pp.ad.Operator:
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

        flux = self.porepy_darcy_flux(domains) + self.internal_flux(domains)
        flux.set_name("Darcy_flux")
        return flux

    def internal_flux(self, domains: Sequence[pp.Grid]) -> pp.ad.TimeDependentDenseArray:
        internal_flux = pp.ad.TimeDependentDenseArray(name="read_flux", domains = domains)
        return internal_flux

class ModifiedSolver():
    def _is_nonlinear_problem(self) -> bool:
        return False

class SinglePhaseFlowGeometry(
    ModifiedGeometry,
    ModifiedSources,
    ModifiedDarcyFlux,
    ModifiedSolver,
    SinglePhaseFlow):
    """Combining the modified geometry and the default model."""
    pass

def full_face_flux_from_coupling_faces(
    sd: pp.Grid,
    coupling_faces: np.ndarray,
    read_flux: np.ndarray,
) -> np.ndarray:
    read_flux = np.asarray(read_flux, dtype=float).reshape(-1)

    q_full = np.zeros(sd.num_faces)
    q_full[coupling_faces] = read_flux

    return q_full

def get_pressure_diff(
    sd: pp.Grid,
    coupling_faces: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    fc = sd.cell_faces.tocsr()

    dp = np.zeros(coupling_faces.size)
    dist = np.zeros(coupling_faces.size)
    grad = np.zeros(coupling_faces.size)

    for i, f in enumerate(coupling_faces):
        start = fc.indptr[f]
        end = fc.indptr[f + 1]

        cells = fc.indices[start:end]
        signs = fc.data[start:end]

        if cells.size != 2:
            raise ValueError(f"Face {f} is not an internal face.")

        # Orientation-dependent jump.
        dp[i] = signs[0] * p[cells[0]] + signs[1] * p[cells[1]]

        x0 = sd.cell_centers[:, cells[0]]
        x1 = sd.cell_centers[:, cells[1]]
        dist[i] = np.linalg.norm(x1 - x0)

        grad[i] = dp[i] / dist[i]

    return grad

fluid_constants = pp.FluidComponent(viscosity=1.0e-3, density=1000.0) #mu(Pa * second)(kg/m^3) for H2O
material_constants = {"fluid": fluid_constants}
model_params = {"material_constants": material_constants}

model = SinglePhaseFlowGeometry(model_params)
model.prepare_simulation()

solver = pp.LinearSolver({})

participant = precice.Participant("Macro", "../precice-config.xml", 0, 1)

# get coupling vertices coordinates: face centers
sd = model.mdg.subdomains(dim = 2)[0]
all_faces = model.mdg.subdomains(dim = 2)
face_cells = sd.cell_faces.tocsr()
num_adjacent_cells = np.diff(face_cells.indptr)
coupling_faces = np.where(num_adjacent_cells == 2)[0]
coords = sd.face_centers[: sd.dim, coupling_faces].T
print("coords", coords)
vertex_ids = participant.set_mesh_vertices("Macro-Mesh", coords)
participant.initialize()

for g in model.mdg.subdomains():
    pp.set_solution_values(
        name="read_flux",
        values=np.zeros(g.num_faces),
        data=model.mdg.subdomain_data(g),
        iterate_index=0,
    )

# prepare aperture for each vertex according to the x-coord
aperture = [ 0.001/(x/h-1) for x, _ in coords ]

exporter = pp.Exporter(
    model.mdg,
    file_name="output",
    folder_name="results",
)
t = 0.0

while participant.is_coupling_ongoing():
    if (participant.requires_writing_checkpoint()):
        pass
    dt = participant.get_max_time_step_size()

    read_avg_v = participant.read_data("Macro-Mesh", "flux", vertex_ids, dt)
    read_flux = read_avg_v * h * aperture
    # The units of the Darcy flux are [m^2 Pa / s].
    # q_darcy_overall = model.equation_system.evaluate(model.porepy_darcy_flux([sd]))
    # print("all flux darcy", q_darcy_overall)
    q_darcy = model.equation_system.evaluate(model.porepy_darcy_flux([sd]))[coupling_faces]
    read_flux = read_flux - q_darcy
    print("compute flux darcy", q_darcy)
    print("compute flux correction", read_flux)
    q_full = np.zeros(sd.num_faces)
    q_full[coupling_faces] = read_flux

    pp.set_solution_values(
        name="read_flux",
        values=q_full,
        data=model.mdg.subdomain_data(sd),
        iterate_index=0,
    )

    converged = solver.solve(model)

    pressure = model.equation_system.evaluate(model.pressure([sd]))
    # print("pressure", pressure)
    pressure_diff = get_pressure_diff(sd, coupling_faces, pressure)
    print("pressure gradient", pressure_diff)
    participant.write_data("Macro-Mesh", "pressure-difference", vertex_ids, pressure_diff)
    participant.write_data("Macro-Mesh", "aperture", vertex_ids, aperture)

    participant.advance(dt)

    if (participant.requires_reading_checkpoint()):
        pass
    else:
        t += dt
        model.update_time_step_solution()
        exporter.write_vtu(["pressure", model.interface_darcy_flux_variable], time_step=t)

participant.finalize()