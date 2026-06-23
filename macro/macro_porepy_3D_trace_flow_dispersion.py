from __future__ import annotations

import numpy as np
import porepy as pp
import precice
from porepy.compositional.compositional_mixins import CompositionalVariables
from porepy.models.compositional_flow import (
    ComponentMassBalanceEquations,
)
from porepy.models.fluid_mass_balance import SinglePhaseFlow

from shared_coupling import (
    coupling_faces_and_coords,
    get_face_scalar_grad,
    get_pressure_grad,
    face_average_from_cells
)
from shared_flux import (
    FaceTransmissibilityFluxMixin,
    LinearProblemMixin,
)
from shared_flow import TracerBC, TracerFluid, TracerIC, ModifiedGeometry, FaceDispersionMixin

class SinglePhaseFlowGeometry(
    ModifiedGeometry,
    TracerFluid,
    CompositionalVariables,
    FaceDispersionMixin,
    ComponentMassBalanceEquations,
    TracerIC,
    TracerBC,
    pp.constitutive_laws.CubicLawPermeability,
    FaceTransmissibilityFluxMixin,
    LinearProblemMixin,
    SinglePhaseFlow,
):
    pass


fluid_constants = pp.FluidComponent(viscosity=1e-3, density=1000.0)
solid_constants = pp.SolidConstants(
    permeability=1e-10, normal_permeability=1e-8, residual_aperture=0.001)
material_constants = {"fluid": fluid_constants, "solid": solid_constants}
model_params = {"material_constants": material_constants, 
                "time_manager": pp.TimeManager(
                    schedule=[0.0, 18000],
                    dt_init=60,
                    constant_dt=True,),
                }
model = SinglePhaseFlowGeometry(model_params)
model.prepare_simulation()

participant = precice.Participant("Macro", "../precice-config.xml", 0, 1)

sd = model.mdg.subdomains(dim=2)[0]
coupling_faces, coords = coupling_faces_and_coords(sd)
coupling_face_widths = sd.face_areas[coupling_faces]
vertex_ids = participant.set_mesh_vertices("Macro-Mesh", coords)
participant.initialize()
tracer_component = next(
    component for component in model.fluid.components if component.name == "tracer"
)

for subdomain in model.mdg.subdomains():
    data = model.mdg.subdomain_data(subdomain)
    pp.set_solution_values(
        name=model.face_transmissibility_cpl,
        values=np.ones(subdomain.num_faces),
        data=data,
        iterate_index=0,
    )
    pp.set_solution_values(
        name=model.face_transmissibility_no_cpl,
        values=np.zeros(subdomain.num_faces),
        data=data,
        iterate_index=0,
    )
    pp.set_solution_values(
        name=model.face_dispersion_cpl,
        values=np.zeros(subdomain.num_faces),
        data=data,
        iterate_index=0,
    )
    pp.set_solution_values(
        name=model.face_dispersion_no_cpl,
        values=np.zeros(subdomain.num_faces),
        data=data,
        iterate_index=0,
    )

fracture_cell_aperture = model.equation_system.evaluate(model.aperture([sd]))
aperture_cpl = face_average_from_cells(sd, coupling_faces, fracture_cell_aperture)
pressure = model.equation_system.evaluate(model.pressure([sd]))
pressure_grad = get_pressure_grad(sd, coupling_faces, pressure)
tracer_fraction = model.equation_system.evaluate(tracer_component.fraction([sd]))
tracer_fraction_grad = get_face_scalar_grad(sd, coupling_faces, tracer_fraction)

solver = pp.LinearSolver({})

while participant.is_coupling_ongoing():
    if participant.requires_writing_checkpoint():
        pass

    dt = participant.get_max_time_step_size()
    model.time_manager.dt = dt

    # The received flux values are volumetric fluxes per width
    # (total_phi [m^3/s] / micro width). Multiply by the PorePy fracture face
    # length to get an integrated macro face flux.
    read_flux_per_width = participant.read_data("Macro-Mesh", "flux", vertex_ids, dt)
    read_flux = model.fluid.reference_component.viscosity * read_flux_per_width * coupling_face_widths

    valid = np.isfinite(read_flux) & np.isfinite(pressure_grad)
    valid &= np.abs(pressure_grad) > 1e-20

    face_transmissibility = np.zeros(sd.num_faces)
    face_transmissibility[coupling_faces[valid]] = (
        read_flux[valid] / pressure_grad[valid]
    )

    face_mask = np.zeros(sd.num_faces)
    face_mask[coupling_faces[valid]] = 1.0

    pp.set_solution_values(
        name=model.face_transmissibility_cpl,
        values=face_transmissibility,
        data=model.mdg.subdomain_data(sd),
        iterate_index=0,
    )
    pp.set_solution_values(
        name=model.face_transmissibility_no_cpl,
        values=face_mask,
        data=model.mdg.subdomain_data(sd),
        iterate_index=0,
    )

    # Read in dispersion
    read_dispersion = participant.read_data("Macro-Mesh", "dispersion", vertex_ids, dt)
    face_dispersion = np.zeros(sd.num_faces)
    # The micro model returns an apparent dispersion coefficient [m^2/s].
    # The macro AD operator expects the coefficient in the integrated face mass flux
    # -C_f grad(z), so C_f = rho * effective_face_area * D.
    fracture_cell_aperture = model.equation_system.evaluate(model.aperture([sd]))
    face_dispersion[coupling_faces] = (
        fluid_constants.density
        * aperture_cpl
        * coupling_face_widths
        * read_dispersion
    )

    dispersion_mask = np.zeros(sd.num_faces)
    dispersion_mask[coupling_faces] = 1.0

    pp.set_solution_values(
        name=model.face_dispersion_cpl,
        values=face_dispersion,
        data=model.mdg.subdomain_data(sd),
        iterate_index=0,
    )
    pp.set_solution_values(
        name=model.face_dispersion_no_cpl,
        values=dispersion_mask,
        data=model.mdg.subdomain_data(sd),
        iterate_index=0,
    )

    solver.solve(model)

    pressure = model.equation_system.evaluate(model.pressure([sd]))
    pressure_grad = get_pressure_grad(sd, coupling_faces, pressure)
    participant.write_data("Macro-Mesh", "pressure-grad", vertex_ids, pressure_grad)

    tracer_fraction = model.equation_system.evaluate(tracer_component.fraction([sd]))
    tracer_fraction_grad = get_face_scalar_grad(sd, coupling_faces, tracer_fraction)
    participant.write_data(
        "Macro-Mesh", "tracer-fraction-grad", vertex_ids, tracer_fraction_grad
    )

    aperture_cpl = face_average_from_cells(sd, coupling_faces, fracture_cell_aperture)
    participant.write_data("Macro-Mesh", "aperture", vertex_ids, aperture_cpl)

    participant.advance(dt)

    if participant.requires_reading_checkpoint():
        pass
    else:
        model.time_manager.increase_time()
        model.time_manager.increase_time_index()
        model.update_time_step_solution()
        model.save_data_time_step()

participant.finalize()
