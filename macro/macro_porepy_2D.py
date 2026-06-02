from typing import Callable, Optional, Sequence, cast

import porepy as pp
import numpy as np
import precice
from porepy.models.fluid_mass_balance import SinglePhaseFlow, BoundaryConditionsSinglePhaseFlow, FluidMassBalanceEquations
from porepy.applications.md_grids.domains import nd_cube_domain
from shared_coupling import get_pressure_grad


h = 0.25

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
        cell_size = self.units.convert_units(h, "m")
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
        values[domain_sides.west] = self.units.convert_units(0.001, "Pa")
        values[domain_sides.east] = self.units.convert_units(0, "Pa")
        return values

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

        flux = self.porepy_darcy_flux(domains) + self.internal_flux()
        flux.set_name("Darcy_flux")
        return flux

    def internal_flux(self) -> pp.ad.TimeDependentDenseArray:
        internal_flux = pp.ad.TimeDependentDenseArray(name="read_flux", domains = self.mdg.subdomains())
        return internal_flux

class ModifiedSolver():
    def _is_nonlinear_problem(self) -> bool:
        return False

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

solid_constants = pp.SolidConstants(permeability=0.833687/h)
fluid_constants = pp.FluidComponent(viscosity=1.0e-3, density=1000.0) #mu(Pa * second)(kg/m^3) for H2O
material_constants = {"fluid": fluid_constants, "solid":solid_constants}
model_params = {"material_constants": material_constants}

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
    q_darcy = model.equation_system.evaluate(model.porepy_darcy_flux([sd]))[internal_faces]
    read_flux = read_flux - q_darcy
    print("compute flux correction", read_flux)
    q_full = full_face_flux_from_internal_faces(sd=sd, internal_faces=internal_faces, read_flux=read_flux)
    pp.set_solution_values(name = "read_flux", values = q_full, data = model.mdg.subdomain_data(sd), iterate_index = 0)
    
    op = pp.ad.TimeDependentDenseArray(name="read_flux", domains=[sd])
    stored_flux = model.equation_system.evaluate(op)

    converged = solver.solve(model)

    pressure = model.equation_system.evaluate(model.pressure([sd]))
    print("pressure", pressure)
    pressure_grad = get_pressure_grad(sd, internal_faces, pressure)
    print("pressure gradient", pressure_grad)
    participant.write_data("Macro-Mesh", "pressure-grad", vertex_ids, pressure_grad)

    # prepare aperture for each vertex according to the x-coord
    aperture = [ 0.1/(x/h) for x, _ in coords ]
    participant.write_data("Macro-Mesh", "aperture", vertex_ids, aperture)

    participant.advance(dt)
    
    if (participant.requires_reading_checkpoint()):
        pass
    else:
        model.update_time_step_solution()  
        pp.plot_grid(model.mdg, "pressure", figsize=(10, 8), plot_2d=True)

participant.finalize()