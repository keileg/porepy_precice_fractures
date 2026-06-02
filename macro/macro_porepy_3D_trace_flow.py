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
from shared_flux import CubicLawPermeabilityModified, LinearProblemMixin
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
    CubicLawPermeabilityModified,
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
model.params["fracture_permeability"] = Aperture * Aperture / 12.0
model.prepare_simulation()

participant = precice.Participant("Macro", "../precice-config.xml", 0, 1)

sd = model.mdg.subdomains(dim=2)[0]
coupling_faces, coords = coupling_faces_and_coords(sd)
vertex_ids = participant.set_mesh_vertices("Macro-Mesh", coords)
participant.initialize()

aperture_cpl = np.full(coords.shape[0], Aperture, dtype=float)
pressure = model.equation_system.evaluate(model.pressure([sd]))
pressure_grad = get_pressure_grad(sd, coupling_faces, pressure)

solver = pp.LinearSolver({})

while participant.is_coupling_ongoing():
    if participant.requires_writing_checkpoint():
        pass

    dt = participant.get_max_time_step_size()
    model.time_manager.dt = dt

    read_avg_v = participant.read_data("Macro-Mesh", "flux", vertex_ids, dt)

    # Infer an effective fracture permeability from received average velocity:
    k_default = aperture_cpl * aperture_cpl / 12.0
    k_face = mu * np.abs(read_avg_v) / np.abs(pressure_grad)
    k_face = np.nan_to_num(k_face, nan=k_default, posinf=k_default, neginf=k_default)

    model.params["fracture_permeability"] = k_face

    solver.solve(model)

    pressure = model.equation_system.evaluate(model.pressure([sd]))
    pressure_grad = get_pressure_grad(sd, coupling_faces, pressure)
    # print("pressure gradient", pressure_grad)
    participant.write_data("Macro-Mesh", "pressure-difference", vertex_ids, pressure_grad)
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
