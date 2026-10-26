#!/usr/bin/env python3

import numpy as np
from pysimfrac import SimFrac
from pathlib import Path
import classy_blocks as cb
import argparse
import io
import matplotlib.pyplot as plt

SEED = 42
_ly = 10


def print_fracture_aperture_metrics(fracture, top_m, bottom_m):
    def format_value(value):
        if value is None:
            return "None"
        return f"{value:0.6g}"

    fracture.top = top_m * 1000
    fracture.bottom = bottom_m * 1000
    fracture.aperture = fracture.top - fracture.bottom
    fracture.mean_aperture = float(np.mean(fracture.aperture))

    fracture.get_effective_aperture("gmean")
    fracture.get_effective_aperture("hmean")

    fracture.compute_acf(surface="aperture", num_lags=20)
    acf = fracture.acf["aperture"]
    print("Aperture correlation length")
    print(f"  x: {format_value(acf['x']['correlation'])} {fracture.units}")
    print(f"  y: {format_value(acf['y']['correlation'])} {fracture.units}")


def makeFracture(
    aperture: float,
    roughness: float,
    shear: float,
    disc: float,
    lx: float = 20,
    lambda_abs: float = 10,
):
    from pysimfrac import SimFrac

    # We need to use mm. Using m leads to div by zero
    fracture = SimFrac(
        h=disc, lx=lx, ly=_ly, shear=shear, method="spectral", units="mm"
    )
    fracture.params["seed"]["value"] = SEED
    fracture.params["mean-aperture"]["value"] = aperture
    fracture.params["roughness"]["value"] = roughness
    fracture.params["mismatch"]["value"] = 1  # 1 for fully matched
    fracture.params["lambda_0"]["value"] = (
        lambda_abs / _ly
    )  # determine the min. frequency; the larger l_0, the lower f_min
    fracture.params["H"]["value"] = 0.8  # common value for natural rock fracture
    fracture.create_fracture()

    fracture.compute_moments()
    fig, ax = fracture.plot_surface_pdf(surface="aperture")
    fig.savefig(fname="surface")

    # transform mm to m to be consistent with openfoam
    top = fracture.top / 1000
    bottom = fracture.bottom / 1000

    midpoint = 0.5 * (top + bottom)
    min_aperture = 1e-6
    aperture_field = np.maximum(top - bottom, min_aperture)

    top = midpoint + 0.5 * aperture_field
    bottom = midpoint - 0.5 * aperture_field

    assert (top > bottom).all()
    print_fracture_aperture_metrics(fracture, top, bottom)
    # fracture.plot_acf(surface="aperture", figname="acf")

    # To build periodic geometry in y-direction for OpenFOAM, need to add the back in front of current front.
    # shape changes from (ny, nx) -> (ny+1, nx+1)
    top = np.pad(top, ((0, 1), (0, 1)), mode="wrap")
    bottom = np.pad(bottom, ((0, 1), (0, 1)), mode="wrap")

    # Coordinates in metres for OpenFOAM
    x = np.arange(fracture.nx + 1, dtype=float) * fracture.h / 1000.0
    y = np.arange(fracture.ny + 1, dtype=float) * fracture.h / 1000.0

    X, Y = np.meshgrid(x, y)

    print("Mesh AABB")

    return X, Y, top, bottom


def make_fitted_from_top_bot(X, Y, Top, Bot, filename: Path, factor=2, nz=6):
    """Fit conforming hex blocks between rectangular, matching surface samples."""
    X, Y, Top, Bot = [np.asarray(a, dtype=float) for a in (X, Y, Top, Bot)]
    factor = int(factor)
    ny, nx = X.shape
    count = nx * ny
    points = np.concatenate(
        [np.column_stack((X.ravel(), Y.ravel(), wall.ravel())) for wall in (Bot, Top)]
    )
    patches = {
        p: [] for p in ("inlet", "outlet", "upperWall", "lowerWall", "front", "back")
    }
    blocks = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b, c, d = (
                j * nx + i,
                j * nx + i + 1,
                (j + 1) * nx + i + 1,
                (j + 1) * nx + i,
            )
            A, B, C, D = (v + count for v in (a, b, c, d))
            vertices = " ".join(map(str, (a, b, c, d, A, B, C, D)))
            blocks.append(
                f"hex ({vertices}) ({factor} {factor} {nz}) simpleGrading (1 1 1)"
            )
            patches["lowerWall"].append((a, d, c, b))
            patches["upperWall"].append((A, B, C, D))
            if i == 0:
                patches["inlet"].append((a, A, D, d))
            if i == nx - 2:
                patches["outlet"].append((b, c, C, B))
            if j == 0:
                patches["back"].append((a, b, B, A))
            if j == ny - 2:
                patches["front"].append((d, D, C, c))
    lines = [
        "FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }",
        "scale 1;",
        "vertices",
        "(",
    ]
    lines += ["(" + " ".join(f"{v:.17g}" for v in point) + ")" for point in points]
    lines += [");", "blocks", "(", *blocks, ");", "edges ();", "boundary", "("]
    for patch, faces in patches.items():
        if patch == "back":
            lines += [
                "back",
                "{",
                "    type cyclic;",
                "    neighbourPatch front;",
                "    faces",
                "    (",
            ]

        elif patch == "front":
            lines += [
                "front",
                "{",
                "    type cyclic;",
                "    neighbourPatch back;",
                "    faces",
                "    (",
            ]
        else:
            kind = "patch" if patch in ("inlet", "outlet") else "wall"

            lines += [
                f"{patch}",
                "{",
                f"    type {kind};",
                "    faces",
                "    (",
            ]
        lines += ["(" + " ".join(map(str, face)) + ")" for face in faces]
        lines += ["); }"]
    lines += [");", "mergePatchPairs ();", ""]
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=Path(__file__).parent)
    parser.add_argument("--aperture", type=float, default=1.0)
    parser.add_argument("--roughness", type=float, default=0.2)
    parser.add_argument("--shear", type=float, default=1)
    parser.add_argument("--disc", type=float, default=1)
    parser.add_argument(
        "--lx", type=float, default=40, help="fracture length in x direction [mm]"
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=3,
        help="background mesh xy-cell factor [cells/mm]",
    )
    parser.add_argument("--nz", type=int, default=6)

    args = parser.parse_args()

    x, y, t, b = makeFracture(
        aperture=args.aperture,
        shear=args.shear,
        disc=args.disc,
        roughness=args.roughness,
        lx=args.lx,
    )

    case = Path(args.case)
    make_fitted_from_top_bot(
        x, y, t, b, args.case / "system/blockMeshDict", args.factor, args.nz
    )
