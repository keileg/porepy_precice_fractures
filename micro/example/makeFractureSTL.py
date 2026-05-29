#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["pysimfrac @ git+https://github.com/lanl/pySimFrac.git","matplotlib"]
# ///


import numpy as np
from pysimfrac import SimFrac
from pathlib import Path
import argparse

def normal(a, b, c):
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n)
    return n / length if length else n


def write_ascii_stl(filename, region_triangles):
    """
    Write an ASCII STL with named regions.

    region_triangles is a dict:
        {
            "top":    [(a, b, c), ...],
            "bot":    [(a, b, c), ...],
            "inlet":  [(a, b, c), ...],
            "outlet": [(a, b, c), ...],
            "ymin":   [(a, b, c), ...],
            "ymax":   [(a, b, c), ...],
        }

    Each triangle is three 3D numpy arrays or array-like points.
    """
 
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

    regions = {
        "upperWall": [],
        "lowerWall": [],
        "inlet": [],
        "outlet": [],
        "frontAndBack": [],
    }

    def add_quad(region, p00, p10, p11, p01):
        regions[region].append((p00, p10, p11))
        regions[region].append((p00, p11, p01))

    # Top and bottom surfaces
    for j in range(ny - 1):
        for i in range(nx - 1):
            add_quad("upperWall", T[j, i], T[j, i + 1], T[j + 1, i + 1], T[j + 1, i])
            add_quad("lowerWall", B[j, i], B[j + 1, i], B[j + 1, i + 1], B[j, i + 1])

    # x-min side: inlet
    i = 0
    for j in range(ny - 1):
        add_quad("inlet", B[j, i], T[j, i], T[j + 1, i], B[j + 1, i])

    # x-max side: outlet
    i = nx - 1
    for j in range(ny - 1):
        add_quad("outlet", B[j, i], B[j + 1, i], T[j + 1, i], T[j, i])

    # y-min side: empty
    j = 0
    for i in range(nx - 1):
        add_quad("frontAndBack", B[j, i], B[j, i + 1], T[j, i + 1], T[j, i])

    # y-max side: empty
    j = ny - 1
    for i in range(nx - 1):
        add_quad("frontAndBack", B[j, i], T[j, i], T[j, i + 1], B[j, i + 1])

    write_ascii_stl(filename, regions)

SEED = 42

def makeFracture(aperture = 1, roughness = 0.5, shear = 0):
    # We need to use mm. Using m leads to div by zero
    x, y = 10, 5
    fracture = SimFrac(h=1, lx=x, ly=y, shear=shear, method="spectral", units="mm")
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

    return X, Y, top, bottom

def fractureSTL(output: Path, **kwargs):
    x, y, t, b = makeFracture(**kwargs)
    make_stl_from_top_bot(x, y, t, b, output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("constant/triSurface/fracture.stl"))
    parser.add_argument("--aperture", type=float, default=1.0)
    parser.add_argument("--roughness", type=float, default=0.5)
    parser.add_argument("--shear", type=float, default=0.0)
    args = parser.parse_args()

    fractureSTL(**vars(args))
