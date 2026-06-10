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
    get_pressure_grad,
)
from shared_flux import (
    FaceTransmissibilityFluxMixin,
    LinearProblemMixin,
)
from shared_flow import TracerBC, TracerFluid, TracerIC, ModifiedGeometry

H = 0.1
mu = 1e-3
Aperture = 0.001

class SinglePhaseFlowGeometry(
    ModifiedGeometry,
    TracerFluid,
    CompositionalVariables,
    ComponentMassBalanceEquations,
    TracerIC,
    TracerBC,
    pp.constitutive_laws.CubicLawPermeability,
    FaceTransmissibilityFluxMixin,
    LinearProblemMixin,
    SinglePhaseFlow,
):
    pass


fluid_constants = pp.FluidComponent(viscosity=mu, density=1000.0)
solid_constants = pp.SolidConstants(
    permeability=1e-10, normal_permeability=1e-8, residual_aperture=Aperture)
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
vertex_ids = participant.set_mesh_vertices("Macro-Mesh", coords)
participant.initialize()

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

aperture_cpl = np.full(coords.shape[0], Aperture, dtype=float)
pressure = model.equation_system.evaluate(model.pressure([sd]))
pressure_grad = get_pressure_grad(sd, coupling_faces, pressure)

solver = pp.LinearSolver({})

while participant.is_coupling_ongoing():
    if participant.requires_writing_checkpoint():
        pass

    dt = participant.get_max_time_step_size()
    model.time_manager.dt = dt

    # The received values is volumetric fluxes per width (total_phi(m^3/s)/width)
    read_flux_per_width = participant.read_data("Macro-Mesh", "flux", vertex_ids, dt)
    read_flux = mu * read_flux_per_width * H

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

    solver.solve(model)

    pressure = model.equation_system.evaluate(model.pressure([sd]))
    pressure_grad = get_pressure_grad(sd, coupling_faces, pressure)
    # print("pressure gradient", pressure_grad)
    participant.write_data("Macro-Mesh", "pressure-grad", vertex_ids, pressure_grad)
    participant.write_data("Macro-Mesh", "aperture", vertex_ids, aperture_cpl)

    participant.advance(dt)

    if participant.requires_reading_checkpoint():
        pass
    else:
        model.time_manager.increase_time()
        model.time_manager.increase_time_index()
        model.update_time_step_solution()
        model.save_data_time_step()
        for intf, data in model.mdg.interfaces(return_data=True):
            print(data[pp.TIME_STEP_SOLUTIONS][model.interface_darcy_flux_variable])

participant.finalize()
