#!/usr/bin/env python3

import numpy as np
from pysimfrac import SimFrac
from pathlib import Path
import classy_blocks as cb
import argparse

def normal(a, b, c):
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n)
    return n / length if length else n


def write_ascii_stl(filename, region_triangles):
    print(f"Writing to {filename.absolute()}")
    with filename.open("w") as f:
        for region, triangles in region_triangles.items():
            f.write(f"solid {region}\n")

            for a, b, c in triangles:
                n = normal(a, b, c)

                f.write(f"  facet normal {n[0]} {n[1]} {n[2]}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {a[0]} {a[1]} {a[2]}\n")
                f.write(f"      vertex {b[0]} {b[1]} {b[2]}\n")
                f.write(f"      vertex {c[0]} {c[1]} {c[2]}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")

            f.write(f"endsolid {region}\n")


def make_stl_from_top_bot(X, Y, Top, Bot, filename: Path):
    X = np.asarray(X)
    Y = np.asarray(Y)
    Top = np.asarray(Top)
    Bot = np.asarray(Bot)

    if X.shape != Y.shape or X.shape != Top.shape or X.shape != Bot.shape:
        raise ValueError("X, Y, Top, and Bot must have the same shape")

    ny, nx = X.shape

    T = np.dstack((X, Y, Top))
    B = np.dstack((X, Y, Bot))

    midpoint = (T[ny//2, nx//2] + B[ny//2, nx//2])/2
    print(f"Midpoint {midpoint}")

    regions = {
        "top": [],
        "bottom": [],
        "left": [],
        "right": [],
        "front": [],
        "back": [],
    }

    def add_quad(region, p00, p10, p11, p01):
        regions[region].append((p00, p10, p11))
        regions[region].append((p00, p11, p01))

    # Top and bottom surfaces
    for j in range(ny - 1):
        for i in range(nx - 1):
            add_quad("top", T[j, i], T[j, i + 1], T[j + 1, i + 1], T[j + 1, i])
            add_quad("bottom", B[j, i], B[j + 1, i], B[j + 1, i + 1], B[j, i + 1])

    # x-min side: inlet
    ileft, iright = 0, nx - 1
    for j in range(ny - 1):
        add_quad("left", B[j, ileft], T[j, ileft], T[j + 1, ileft], B[j + 1, ileft])
        add_quad("right", B[j, iright], B[j + 1, iright], T[j + 1, iright], T[j, iright])

    # y-min side: empty
    jfront, jback = 0, ny - 1
    for i in range(nx - 1):
        add_quad("front", B[jfront, i], B[jfront, i + 1], T[jfront, i + 1], T[jfront, i])
        add_quad("back", B[jback, i], T[jback, i], T[jback, i + 1], B[jback, i + 1])

    write_ascii_stl(filename, regions)

SEED = 42

def makeFracture(aperture = 1, roughness = 0.5, shear = 0, disc = 1.0):
    # We need to use mm. Using m leads to div by zero
    x, y = 10, 5
    fracture = SimFrac(h=disc, lx=x, ly=y, shear=shear, method="spectral", units="mm")
    fracture.params["seed"]["value"] = SEED
    fracture.params["mean-aperture"]["value"] = aperture
    fracture.params["roughness"]["value"] = roughness
    fracture.create_fracture()

    # move X and Y to 0 origin
    # transform mm to m to be consistent with openfoam
    X = (fracture.X - fracture.X.min()) / 1000
    Y = (fracture.Y - fracture.Y.min()) / 1000
    top = fracture.top / 1000
    bottom = fracture.bottom / 1000

    print(f"X {X.min()} {X.max()}")
    print(f"Y {Y.min()} {Y.max()}")
    print(f"Z {bottom.min()} {top.max()}")

    return X, Y, top, bottom

def make_background_from_top_bot(X, Y, Top, Bot, filename: Path):
    X = np.asarray(X)
    Y = np.asarray(Y)
    Top = np.asarray(Top)
    Bot = np.asarray(Bot)

    if X.shape != Y.shape or X.shape != Top.shape or X.shape != Bot.shape:
        raise ValueError("X, Y, Top, and Bot must have the same shape")

    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Bot.min(), Top.max()

    minp = [xmin, ymin, zmin]
    maxp = [xmax, ymax, zmax]

    factor = 2

    mesh = cb.Mesh()
    box = cb.Box(minp, maxp)
    box.chop(0, count=10*factor)
    box.chop(1, count=5*factor)
    box.chop(2, count=(zmax-zmin)*1000*factor)
    mesh.add(box)

    print(f"Writing background mesh to {filename.absolute()}")
    mesh.write(filename, debug_path="debugbg.vtk")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=Path(__file__).parent)
    parser.add_argument("--aperture", type=float, default=1.0)
    parser.add_argument("--roughness", type=float, default=0.5)
    parser.add_argument("--shear", type=float, default=0.0)
    parser.add_argument("--disc", type=float, default=1.0)
    args = parser.parse_args()

    x, y, t, b = makeFracture(aperture=args.aperture, shear=args.shear, disc=args.disc, roughness=args.roughness)
    make_stl_from_top_bot(x, y, t, b, args.case / "constant/triSurface/fracture.stl")
    make_background_from_top_bot(x, y, t, b, args.case / "system/blockMeshDict")
