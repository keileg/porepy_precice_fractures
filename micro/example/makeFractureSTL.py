#!/usr/bin/env python3

import numpy as np
from pysimfrac import SimFrac
from pathlib import Path
import classy_blocks as cb
import argparse
import matplotlib.pyplot as plt

def plot_spatial_frequency_psd(X, Y, Z, filename="psd.png"):
    """
    Plot radially averaged 2D power spectral density of surface Z.

    X, Y, Z must have the same shape.
    X and Y should be in physical units, e.g. mm or m.
    Spatial frequency will be 1 / that unit.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)

    # Remove mean height before FFT
    z = Z - np.mean(Z)

    ny, nx = z.shape
    dx = np.mean(np.diff(X[0, :]))
    dy = np.mean(np.diff(Y[:, 0]))

    fft_z = np.fft.fft2(z)
    psd_2d = (np.abs(fft_z) ** 2) * dx * dy / (nx * ny)

    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)
    f = np.sqrt(FX**2 + FY**2)

    # Flatten and remove zero frequency
    f = f.ravel()
    psd = psd_2d.ravel()

    mask = f > 0
    f = f[mask]
    psd = psd[mask]

    # Radial binning
    bins = np.logspace(np.log10(f.min()), np.log10(f.max()), 50)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    psd_radial = np.zeros(len(bin_centers))

    for i in range(len(bin_centers)):
        in_bin = (f >= bins[i]) & (f < bins[i + 1])
        psd_radial[i] = np.mean(psd[in_bin]) if np.any(in_bin) else np.nan

    valid = ~np.isnan(psd_radial)

    fig, ax = plt.subplots()
    ax.loglog(bin_centers[valid], psd_radial[valid], "o-")
    ax.set_xlabel("Spatial frequency")
    ax.set_ylabel("Power spectral density")
    ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)

def normal(a, b, c):
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n)
    return n / length if length else n


def write_ascii_stl(filename, region_triangles):
    print(f"Writing to {filename.absolute()}")
    with filename.open("w", buffering=2**14) as f:
        for region, triangles in region_triangles.items():
            print(f"Region {region}")

            triangles = np.array(triangles, dtype=np.float32)  # shape: (N, 3, 3)
            v1 = triangles[:, 1] - triangles[:, 0]
            v2 = triangles[:, 2] - triangles[:, 0]
            n = np.cross(v1, v2)
            n /= np.linalg.norm(n, axis=1, keepdims=True)  # shape: (N, 3)
            result = np.concatenate([n[:, np.newaxis, :], triangles], axis=1)

            f.write(f"solid {region}\n")

            for a, b, c, n in result:
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

def makeFracture(aperture: float, roughness: float, shear: float, disc: float):
    # We need to use mm. Using m leads to div by zero
    x, y = 200,200
    fracture = SimFrac(h=disc, lx=x, ly=y, shear=shear, method="spectral", units="mm")
    fracture.params["seed"]["value"] = SEED
    fracture.params["mean-aperture"]["value"] = aperture
    fracture.params["roughness"]["value"] = roughness
    fracture.params["mismatch"]["value"] = 1
    fracture.params["lambda_0"]["value"] = 1
    fracture.params["H"]["value"] = 0.8 # common value for natural rock fracture
    fracture.create_fracture()

    fracture.compute_moments()
    fig,ax = fracture.plot_surface_pdf()
    fig.savefig(fname="surface")

    # move X and Y to 0 origin
    # transform mm to m to be consistent with openfoam
    X = (fracture.X - fracture.X.min()) / 1000
    Y = (fracture.Y - fracture.Y.min()) / 1000
    top = fracture.top / 1000
    bottom = fracture.bottom / 1000

    midpoint = 0.5 * (top + bottom)
    min_aperture = 1e-6
    aperture = np.maximum(top - bottom, min_aperture)

    top = midpoint + 0.5 * aperture
    bottom = midpoint - 0.5 * aperture

    assert((top > bottom).all())

    print("Mesh AABB")
    print(f"X {X.min()} {X.max()}")
    print(f"Y {Y.min()} {Y.max()}")
    print(f"Z {bottom.min()} {top.max()}")

    plot_spatial_frequency_psd(
        X,
        Y,
        top - bottom,
        filename="aperture_psd.png",
    )

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
    mesh.write(filename)

def patch_midpoint(X, Y, T, B, filename: Path):
    assert X.shape == Y.shape and Y.shape == T.shape and T.shape == B.shape
    ny, nx = X.shape
    T = np.dstack((X, Y, T))
    B = np.dstack((X, Y, B))

    midpoint = (T[ny//2, nx//2] + B[ny//2, nx//2])/2
    print(f"Patching midpoint {midpoint} in {filename.absolute()}")

    lines = [ line if "locationInMesh" not in line else f"  locationInMesh ({midpoint[0]} {midpoint[1]} {midpoint[2]});"
            for line in filename.read_text().splitlines(keepends=False)]
    filename.write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=Path(__file__).parent)
    parser.add_argument("--aperture", type=float, default=1.0)
    parser.add_argument("--roughness", type=float, default=3.0)
    parser.add_argument("--shear", type=float, default=1)
    parser.add_argument("--disc", type=float, default=1)
    args = parser.parse_args()

    x, y, t, b = makeFracture(aperture=args.aperture, shear=args.shear, disc=args.disc, roughness=args.roughness)
    make_stl_from_top_bot(x, y, t, b, args.case / "constant/triSurface/fracture.stl")
    make_background_from_top_bot(x, y, t, b, args.case / "system/blockMeshDict")
    patch_midpoint(x, y, t, b, args.case / "system/snappyHexMeshDict")
