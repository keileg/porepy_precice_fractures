from __future__ import annotations

import csv
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from foamlib import FoamCase


# ---------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------

BASE_CASE = Path("example")
RUN_ROOT = Path("validation_laminar_channel")

INLET_PATCH = "inlet"

# Geometry
L = 1.0       # channel length [m]
H = 0.1       # wall-to-wall gap [m]
W = 0.1       # nominal width [m]; only used for informational flow-rate checks

# Fluid properties
NU = 1.0e-6   # kinematic viscosity [m^2/s]
RHO = 1.0e3   # density [kg/m^3]

# Pressure-drop sweep. Unit: Pa.
DP_VALUES = [2.5e-3, 5.0e-3, 1.0e-2, 1.6e-2]

# Name of the OpenFOAM application to run.
SOLVER = "simpleFoam"

OVERWRITE_RUN_ROOT = True

@dataclass
class ValidationResult:
    dp_input: float
    u_max_num: float
    u_mean_num: float
    u_max_ana: float
    u_mean_ana: float
    err_umax_rel: float
    err_umean_rel: float

# Analytical solution
def to_kinematic_pressure_drop(dp_input: float) -> float:
    """Convert input pressure drop to kinematic pressure drop."""
    return dp_input / RHO


def poiseuille_analytical(dp_kin: float) -> tuple[float, float]:
    """
    Plane Poiseuille flow between plates, using kinematic pressure drop.

    dp_kin = p_in - p_out over channel length L.
    """
    grad_mag = dp_kin / L
    u_max = grad_mag * H**2 / (8.0 * NU)
    u_mean = grad_mag * H**2 / (12.0 * NU)
    return u_max, u_mean


# OpenFOAM case editing
def set_pressure_drop(case: FoamCase, dp_kin: float) -> None:
    """
    Set fixed pressure values in 0/p.
    """
    p0 = case[0]["p"]
    p0.boundary_field[INLET_PATCH].value = dp_kin

# Running and reading results
def run_case(case: FoamCase) -> None:
    """Build mesh and run solver."""
    case.clean(check=False)
    case.block_mesh(check=True, log=True)
    case.run(cmd=SOLVER, check=True, log=True)


def extract_u_max(case: FoamCase) -> tuple[float, float]:
    final = case[-1]
    U = np.asarray(final["U"].internal_field, dtype=float)

    u_mag = np.linalg.norm(U, axis=1)
    return float(np.max(u_mag)), float(final.time)

def parse_surface_mean(case_dir: Path) -> float:
    dat = case_dir / "postProcessing" / "outletAverageVelocity/0/surfaceFieldValue.dat"

    lines = [
        line.strip()
        for line in dat.read_text().splitlines()
    ]
    last = lines[-1]

    return float(last.split()[1])


def run_single(dp: float) -> ValidationResult:
    case_name = f"dp_{dp:.3e}"
    case_dir = RUN_ROOT / case_name
    
    dp_kin = to_kinematic_pressure_drop(dp)

    if case_dir.exists():
        shutil.rmtree(case_dir)

    base = FoamCase(BASE_CASE)
    case = base.clone(case_dir)

    case = FoamCase(case_dir)

    set_pressure_drop(case, dp_kin)
    run_case(case)

    u_max_num, final_time = extract_u_max(case)
    u_mean_num = parse_surface_mean(case_dir)

    u_max_ana, u_mean_ana = poiseuille_analytical(dp_kin)

    err_umax_rel = abs(u_max_num - u_max_ana) / abs(u_max_ana)
    err_umean_rel = abs(u_mean_num - u_mean_ana) / abs(u_mean_ana)

    return ValidationResult(
        dp_input=dp,
        u_max_num=u_max_num,
        u_mean_num=u_mean_num,
        u_max_ana=u_max_ana,
        u_mean_ana=u_mean_ana,
        err_umax_rel=err_umax_rel,
        err_umean_rel=err_umean_rel,
    )

def write_csv(results: Iterable[ValidationResult], out_file: Path) -> None:
    rows = list(results)
    if not rows:
        return

    with out_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].__dict__.keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r.__dict__)

def main() -> None:
    if not BASE_CASE.exists():
        raise FileNotFoundError(f"Base case not found: {BASE_CASE}")

    if OVERWRITE_RUN_ROOT and RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    results: list[ValidationResult] = []
    for dp in DP_VALUES:
        print(f"\n=== Running dp={dp:.4e} ===")
        result = run_single(dp)
        results.append(result)

    csv_path = RUN_ROOT / "poiseuille_validation_summary.csv"
    write_csv(results, csv_path)
    print(f"\nWrote summary done.")

if __name__ == "__main__":
    main()