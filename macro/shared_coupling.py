from __future__ import annotations

import numpy as np
import porepy as pp


def coupling_faces_and_coords(sd: pp.Grid) -> tuple[np.ndarray, np.ndarray]:
    face_cells = sd.cell_faces.tocsr()
    num_adjacent_cells = np.diff(face_cells.indptr)
    coupling_faces = np.where(num_adjacent_cells == 2)[0]
    coords = sd.face_centers[: sd.dim, coupling_faces].T
    return coupling_faces, coords


def full_face_flux_from_coupling_faces(
    sd: pp.Grid,
    coupling_faces: np.ndarray,
    read_flux: np.ndarray,
) -> np.ndarray:
    read_flux = np.asarray(read_flux, dtype=float).reshape(-1)
    q_full = np.zeros(sd.num_faces)
    q_full[coupling_faces] = read_flux
    return q_full


def get_pressure_grad(
    sd: pp.Grid,
    coupling_faces: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    fc = sd.cell_faces.tocsr()

    grad = np.zeros(coupling_faces.size)

    for i, f in enumerate(coupling_faces):
        start = fc.indptr[f]
        end = fc.indptr[f + 1]

        cells = fc.indices[start:end]
        signs = fc.data[start:end]

        if cells.size != 2:
            raise ValueError(f"Face {f} is not an internal face.")

        dp = signs[0] * p[cells[0]] + signs[1] * p[cells[1]]

        x0 = sd.cell_centers[:, cells[0]]
        x1 = sd.cell_centers[:, cells[1]]
        dist = np.linalg.norm(x1 - x0)

        grad[i] = dp / dist

    return grad
