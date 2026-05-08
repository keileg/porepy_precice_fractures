import numpy as np
import precice

participant = precice.Participant("Macro", "precice-config.xml", 0, 1)

mesh_name = "Macro-Mesh"
vertex = np.array([[0.5, 0.5, 0.5], [0.2, 0.5, 0.5]], dtype=float)
vertex_ids = participant.set_mesh_vertices(mesh_name, vertex)

participant.initialize()

t = 0.0
while participant.is_coupling_ongoing():
    dt = participant.get_max_time_step_size()

    dp = np.array([t + 1.0, t+1.1], dtype=float)

    flux = participant.read_data(mesh_name, "flux", vertex_ids, 0)
    participant.write_data(mesh_name, "pressure-difference", vertex_ids, dp)

    print("read flux", flux)

    t += dt

    participant.advance(dt)

participant.finalize()
