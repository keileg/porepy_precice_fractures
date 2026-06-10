#!/usr/bin/env python3

import numpy as np
from pysimfrac import SimFrac
from pathlib import Path
import classy_blocks as cb
import argparse
import io
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
    print(f"Serializing to STL")

    f = io.StringIO()
    for region, triangles in region_triangles.items():
        print(f" Region {region}")

        f.write(f"solid {region}\n")
        f.writelines((
            f"facet normal {n[0]} {n[1]} {n[2]}\nouter loop\n"
            f"vertex {a[0]} {a[1]} {a[2]}\n"
            f"vertex {b[0]} {b[1]} {b[2]}\n"
            f"vertex {c[0]} {c[1]} {c[2]}\n"
            "endloop\nendfacet\n"
            for a, b, c, n in triangles
            ))
        f.write(f"endsolid {region}\n")

    print(f"Writing to {filename.absolute()}")
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(f.getvalue())


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

    n_top_bot = (ny - 1) * (nx - 1) * 2
    n_sides   = (ny - 1) * 2          # left, right
    n_fb      = (nx - 1) * 2          # front, back

    # n triangles of 3 points + normal in 3D
    regions = {
        "top":    np.empty((n_top_bot, 4, 3), dtype=np.float32),
        "bottom": np.empty((n_top_bot, 4, 3), dtype=np.float32),
        "left":   np.empty((n_sides,   4, 3), dtype=np.float32),
        "right":  np.empty((n_sides,   4, 3), dtype=np.float32),
        "front":  np.empty((n_fb,      4, 3), dtype=np.float32),
        "back":   np.empty((n_fb,      4, 3), dtype=np.float32),
    }

    def add_quad(region, idx, p00, p10, p11, p01):
        regions[region][2*idx, :-1] = (p00, p10, p11)
        regions[region][2*idx+1, :-1] = (p00, p11, p01)

    # Top and bottom surfaces
    for j in range(ny - 1):
        for i in range(nx - 1):
            idx = j * (nx - 1) + i
            add_quad("top", idx, T[j, i], T[j, i + 1], T[j + 1, i + 1], T[j + 1, i])
            add_quad("bottom", idx, B[j, i], B[j + 1, i], B[j + 1, i + 1], B[j, i + 1])

    # x-min side: inlet
    ileft, iright = 0, nx - 1
    for j in range(ny - 1):
        add_quad("left", j, B[j, ileft], T[j, ileft], T[j + 1, ileft], B[j + 1, ileft])
        add_quad("right", j, B[j, iright], B[j + 1, iright], T[j + 1, iright], T[j, iright])

    # y-min side: empty
    jfront, jback = 0, ny - 1
    for i in range(nx - 1):
        add_quad("front", i, B[jfront, i], B[jfront, i + 1], T[jfront, i + 1], T[jfront, i])
        add_quad("back", i, B[jback, i], T[jback, i], T[jback, i + 1], B[jback, i + 1])

    for reg in regions.values():
        v1 = reg[:, 1] - reg[:, 0]
        v2 = reg[:, 2] - reg[:, 0]
        n = np.cross(v1, v2)
        n /= np.linalg.norm(n, axis=1, keepdims=True)  # shape: (N, 3)
        reg[:, 3] = n

    write_ascii_stl(filename, regions)

SEED = 42

def print_fracture_aperture_metrics(fracture, top_m, bottom_m):
    def format_value(value):
        if value is None:
            return "None"
        return f"{value:0.6g}"

    fracture.top = top_m * 1000
    fracture.bottom = bottom_m * 1000
    fracture.aperture = fracture.top - fracture.bottom
    fracture.mean_aperture = float(np.mean(fracture.aperture))

    fracture.compute_acf(surface="aperture")
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
    ly: float = 20,
):
    from pysimfrac import SimFrac

    # We need to use mm. Using m leads to div by zero
    fracture = SimFrac(h=disc, lx=lx, ly=ly, shear=shear, method="spectral", units="mm")
    fracture.params["seed"]["value"] = SEED
    fracture.params["mean-aperture"]["value"] = aperture
    fracture.params["roughness"]["value"] = roughness
    fracture.params["mismatch"]["value"] = 1
    fracture.params["lambda_0"]["value"] = 10/lx # determine the min. frequency; the larger l_0, the lower f_min
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
    print_fracture_aperture_metrics(fracture, top, bottom)

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


def make_background_from_top_bot(X, Y, Top, Bot, filename: Path, factor: float):
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

    mesh = cb.Mesh()
    box = cb.Box(minp, maxp)
    box.chop(0, count=X.shape[1] * factor)
    box.chop(1, count=X.shape[0] * factor)
    box.chop(2, count=(zmax - zmin) * 1000 * 8 * factor)
    mesh.add(box)

    print(f"Writing background mesh to {filename.absolute()}")
    filename.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--roughness", type=float, default=0.2)
    parser.add_argument("--shear", type=float, default=1)
    parser.add_argument("--disc", type=float, default=1)
    parser.add_argument("--lx", type=float, default=40, help="fracture length in x direction [mm]")
    parser.add_argument("--ly", type=float, default=40, help="fracture length in y direction [mm]")
    parser.add_argument("--factor", type=float, default=2, help="background mesh z-cell factor [cells/mm]")
 
    args = parser.parse_args()

    x, y, t, b = makeFracture(
        aperture=args.aperture,
        shear=args.shear,
        disc=args.disc,
        roughness=args.roughness,
        lx=args.lx,
        ly=args.ly,
    )

    make_stl_from_top_bot(x, y, t, b, args.case / "constant/triSurface/fracture.stl")
    make_background_from_top_bot(x, y, t, b, args.case / "system/blockMeshDict", args.factor)
    patch_midpoint(x, y, t, b, args.case / "system/snappyHexMeshDict")
