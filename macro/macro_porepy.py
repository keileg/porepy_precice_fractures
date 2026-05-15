from typing import Callable, Optional, Sequence, cast

import porepy as pp
import numpy as np
import precice
from porepy.models.fluid_mass_balance import SinglePhaseFlow, BoundaryConditionsSinglePhaseFlow, FluidMassBalanceEquations
from porepy.applications.md_grids.domains import nd_cube_domain



class ModifiedGeometry:
    def set_domain(self) -> None:
        """Defining a two-dimensional square domain with sidelength 2."""
        size = self.units.convert_units(1, "m")
        self._domain = nd_cube_domain(2, size)

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
        cell_size = self.units.convert_units(0.25, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

class ModifiedBC(BoundaryConditionsSinglePhaseFlow):
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
        values[domain_sides.west] = self.units.convert_units(3, "Pa")
        values[domain_sides.east] = self.units.convert_units(2, "Pa")
        return values


# class ModifiedFlux(FluidMassBalanceEquations):
#     def fluid_flux(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
#         if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
#             return self.create_boundary_operator(
#                 name=self.bc_data_fluid_flux_key,
#                 domains=cast(Sequence[pp.BoundaryGrid], domains),
#             )
#
#         # Verify that the domains are subdomains.
#         if not all(isinstance(d, pp.Grid) for d in domains):
#             raise ValueError("domains must consist entirely of subdomains.")
#         # Now we can cast the domains
#         domains = cast(list[pp.Grid], domains)
#
#         flux = self.advective_flux(
#             domains,
#             self.advection_weight_mass_balance(domains),
#             self.mobility_discretization(domains),
#             self.boundary_fluid_flux(domains)
#         )
#         flux.set_name("fluid_flux")
#         return flux
#
class ModifiedDarcyFlux:
    def darcy_flux(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        if len(domains) == 0 or all([isinstance(g, pp.BoundaryGrid) for g in domains]):
            # Note: in case of the empty subdomain list, the time dependent array is
            # still returned. Otherwise, this method produces an infinite recursion
            # loop. It does not affect real computations anyhow.
            return self.create_boundary_operator(
                name=self.bc_data_darcy_flux_key,
                domains=cast(Sequence[pp.BoundaryGrid], domains),
            )# Check that the domains are grids.
        if not all([isinstance(g, pp.Grid) for g in domains]):
            raise ValueError(
                """Argument `domains` should either be a list of grids or a list of
                boundary grids."""
            )
        # By now we know that subdomains is a list of grids, so we can cast it as such
        # (in the typing sense).
        domains = cast(list[pp.Grid], domains)

        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(domains, [1])
        intf_projection = pp.ad.MortarProjections(self.mdg, domains, interfaces, dim=1)

        boundary_operator = self.combine_boundary_operators_darcy_flux(
            subdomains=domains
        )

        discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.darcy_flux_discretization(
            domains
        )
        
        flux: pp.ad.Operator = (
            discr.flux() @ self.pressure(domains) # use pp.ad.Scalar(0) instead
            + self.internal_flux()
            + discr.bound_flux()
            @ (
                boundary_operator
                + intf_projection.mortar_to_primary_int()
                @ self.interface_darcy_flux(interfaces)
            )
            + discr.vector_source() @ self.vector_source_darcy_flux(domains)
        )
        flux.set_name("Darcy_flux")
        return flux

    def internal_flux(self) -> pp.ad.TimeDependentDenseArray:
        internal_flux = pp.ad.TimeDependentDenseArray(name="read_flux", domains = self.mdg.subdomains())
        return internal_flux

class ModifiedSolver():
    def _is_nonlinear_problem(self) -> bool:
        return False
# class ModifiedTimeManager(TimeManager):
#     def compute_time_step(self):
#         dt = participant.get_max_time_step_size()
#         return dt

class SinglePhaseFlowGeometry(
    ModifiedGeometry,
    ModifiedBC,
    ModifiedDarcyFlux,
    ModifiedSolver,
    # ModifiedTimeManager,
    # ModifiedFlux,
    SinglePhaseFlow):
    """Combining the modified geometry and the default model."""
    pass

def full_face_flux_from_internal_faces(
    sd: pp.Grid,
    internal_faces: np.ndarray,
    read_flux: np.ndarray,
) -> np.ndarray:
    read_flux = np.asarray(read_flux, dtype=float).reshape(-1)

    q_full = np.zeros(sd.num_faces)
    q_full[internal_faces] = read_flux

    return q_full

def get_pressure_diff(
    sd: pp.Grid,
    internal_faces: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    fc = sd.cell_faces.tocsr()

    dp = np.zeros(internal_faces.size)
    dist = np.zeros(internal_faces.size)
    grad = np.zeros(internal_faces.size)

    for i, f in enumerate(internal_faces):
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
# time_manager = pp.TimeManager(
#     schedule=[0, 3e-1],
#     dt_init=1e-1,
#     constant_dt=True,
#     iter_max=10,
#     print_info=True,
# )

model = SinglePhaseFlowGeometry(model_params)
model.prepare_simulation()

solver = pp.LinearSolver({})

participant = precice.Participant("Macro", "../precice-config.xml", 0, 1)

# get coupling vertices coordinates: face centers
sd = model.mdg.subdomains()[0]
face_cells = sd.cell_faces.tocsr()
num_adjacent_cells = np.diff(face_cells.indptr)
internal_faces = np.where(num_adjacent_cells == 2)[0]
coords = sd.face_centers[: sd.dim, internal_faces].T
print("coords", coords)
vertex_ids = participant.set_mesh_vertices("Macro-Mesh", coords)
participant.initialize()

pp.set_solution_values(
    name="read_flux",
    values=np.zeros(sd.num_faces),
    data=model.mdg.subdomain_data(sd),
    iterate_index=0,
)

while participant.is_coupling_ongoing():
    if (participant.requires_writing_checkpoint()):
        pass
    dt = participant.get_max_time_step_size()

    read_flux = participant.read_data("Macro-Mesh", "flux", vertex_ids, dt)
    print("read flux", read_flux)
    q_full = full_face_flux_from_internal_faces(sd=sd, internal_faces=internal_faces,    read_flux=read_flux)
    pp.set_solution_values(name = "read_flux", values = q_full, data = model.mdg.subdomain_data(sd), iterate_index = 0)
    
    op = pp.ad.TimeDependentDenseArray(name="read_flux", domains=[sd])
    stored_flux = model.equation_system.evaluate(op)

    converged = solver.solve(model)

    pressure = model.equation_system.evaluate(model.pressure([sd]))
    print("pressure", pressure)
    pressure_diff = get_pressure_diff(sd, internal_faces, pressure)
    print("pressure diff", pressure_diff)
    participant.write_data("Macro-Mesh", "pressure-difference", vertex_ids, pressure_diff)

    participant.advance(dt)
    
    if (participant.requires_reading_checkpoint()):
        pass
    else:
        model.update_time_step_solution()  
        pp.plot_grid(model.mdg, "pressure", figsize=(10, 8), plot_2d=True)

participant.finalize() 



