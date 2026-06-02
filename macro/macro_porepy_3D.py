from __future__ import annotations

import numpy as np
import porepy as pp
import precice
from porepy.models.fluid_mass_balance import SinglePhaseFlow

from shared_coupling import (
    coupling_faces_and_coords,
    full_face_flux_from_coupling_faces,
    get_pressure_grad,
)
from shared_flux import LinearProblemMixin
from shared_flow import ModifiedGeometry

H = 0.1


class Macro3DGeometry(ModifiedGeometry):
    fracture_points = np.array(
        [[0.4, 0.7, 0.7, 0.4], [0.4, 0.4, 0.7, 0.7], [0.5, 0.5, 0.5, 0.5]]
    )


class SinglePhaseFlowGeometry(
    Macro3DGeometry,
    LinearProblemMixin,
    SinglePhaseFlow,
):
    pass


fluid_constants = pp.FluidComponent(viscosity=1.0e-3, density=1000.0)
material_constants = {"fluid": fluid_constants}
model_params = {"material_constants": material_constants}

model = SinglePhaseFlowGeometry(model_params)
model.prepare_simulation()

solver = pp.LinearSolver({})

participant = precice.Participant("Macro", "../precice-config.xml", 0, 1)

sd = model.mdg.subdomains(dim=2)[0]
coupling_faces, coords = coupling_faces_and_coords(sd)
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

aperture = np.full(coords.shape[0], 0.001, dtype=float)

exporter = pp.Exporter(
    model.mdg,
    file_name="output",
    folder_name="results",
)
t = 0.0

while participant.is_coupling_ongoing():
    if participant.requires_writing_checkpoint():
        pass

    dt = participant.get_max_time_step_size()

    read_avg_v = participant.read_data("Macro-Mesh", "flux", vertex_ids, dt)
    read_flux = read_avg_v * H * aperture

    q_darcy = model.equation_system.evaluate(model.porepy_darcy_flux([sd]))[coupling_faces]
    read_flux = read_flux - q_darcy
    print("compute flux darcy", q_darcy)
    print("compute flux correction", read_flux)

    q_full = full_face_flux_from_coupling_faces(
        sd=sd,
        coupling_faces=coupling_faces,
        read_flux=read_flux,
    )

    pp.set_solution_values(
        name="read_flux",
        values=q_full,
        data=model.mdg.subdomain_data(sd),
        iterate_index=0,
    )

    solver.solve(model)

    pressure = model.equation_system.evaluate(model.pressure([sd]))
    pressure_grad = get_pressure_grad(sd, coupling_faces, pressure)
    print("pressure gradient", pressure_grad)
    participant.write_data("Macro-Mesh", "pressure-grad", vertex_ids, pressure_grad)
    participant.write_data("Macro-Mesh", "aperture", vertex_ids, aperture)

    participant.advance(dt)

    if participant.requires_reading_checkpoint():
        pass
    else:
        t += dt
        model.update_time_step_solution()
        exporter.write_vtu(["pressure", model.interface_darcy_flux_variable], time_step=t)

participant.finalize()
