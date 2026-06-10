#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from foamlib import FoamCase

BASE_CASE = "./example"
DEFAULT_RUN_ROOT = "./field_mesh_study"
GENERATOR = BASE_CASE / "makeFractureSTL.py"


@dataclass(frozen=True)
class StudyPoint:
    lx_mm: float
    ly_mm: float
    disc_mm: float
    factor: float
    pressure_grad: float


@dataclass
class StudyResult:
    # case: str
    lx_mm: float
    ly_mm: float
    disc_mm: float
    factor: float
    pressure_grad: float | None
    final_time: float | None
    outlet_flux_phi_per_width: float | None
    status: str


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def case_name(point: StudyPoint) -> str:
    return (
        f"lx{point.lx_mm:g}_ly{point.ly_mm:g}_disc{point.disc_mm:g}"
        f"_factor{point.factor:g}_dp{point.pressure_grad:g}"
    )


def is_time_dir(name: str) -> bool:
    return re.fullmatch(r"[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?", name) is not None


def copy_template(dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Case already exists: {dst}")
        shutil.rmtree(dst)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        ignored = {"postProcessing", "__pycache__"}
        ignored.update(name for name in names if name.startswith("processor"))
        ignored.update(name for name in names if name.startswith("log."))
        ignored.update(name for name in names if is_time_dir(name) and name != "0")
        return ignored

    shutil.copytree(BASE_CASE, dst, ignore=ignore)
    shutil.rmtree(dst / "constant" / "polyMesh", ignore_errors=True)
    (dst / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)


def run_logged(cmd: list[str], cwd: Path, log_name: str) -> None:
    log_path = cwd / log_name
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)


def set_pressure(case_dir: Path, pressure_gradient: float | None, lx_mm: float) -> float | None:
    dp = pressure_gradient * lx_mm / 1000.0

    case = FoamCase(case_dir)
    p0 = case[0]["p"]
    p0.boundary_field["inlet"].value = float(dp)
    p0.boundary_field["outlet"].value = 0.0

    return float(pressure_gradient)

def last_data_line(path: Path) -> tuple[float | None, str | None]:
    if not path.exists():
        return None, None

    data_lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not data_lines:
        return None, None

    parts = data_lines[-1].split(maxsplit=1)
    time = float(parts[0])
    value = parts[1] if len(parts) > 1 else None
    return time, value

def latest_surface_value(case_dir: Path, function_name: str) -> tuple[float | None, str | None]:
    files = sorted((case_dir / "postProcessing" / function_name).glob("*/surfaceFieldValue.dat"))
    if not files:
        return None, None
    return last_data_line(files[-1])


def flux_per_width(outlet_flux_phi: str | None, ly_mm: float) -> float | None:
    if outlet_flux_phi is None:
        return None
    return float(outlet_flux_phi) / (ly_mm / 1000.0)


def generate_case(case_dir: Path, point: StudyPoint, aperture: float, roughness: float, shear: float)-> None:
    run_logged(
        [
            sys.executable,
            str(GENERATOR),
            "--case",
            str(case_dir),
            "--aperture",
            str(aperture),
            "--roughness",
            str(roughness),
            "--shear",
            str(shear),
            "--disc",
            str(point.disc_mm),
            "--lx",
            str(point.lx_mm),
            "--ly",
            str(point.ly_mm),
            "--factor",
            str(point.factor),
        ],
        cwd=case_dir,
        log_name="log.generateGeometry",
    )


def run_study_case(
    run_root: Path,
    point: StudyPoint,
    args: argparse.Namespace,
) -> StudyResult:
    name = case_name(point)
    case_dir = run_root / name
    copy_template(case_dir, overwrite=args.overwrite)
    generate_case(case_dir=case_dir, point=point, aperture=args.aperture, roughness=args.roughness, shear=args.shear)
    dp = set_pressure(case_dir, point.pressure_grad, point.lx_mm)

    if args.dry_run:
        status = "generated"
        final_time = None
        outlet_phi_per_width = None
    else:
        try:
            run_logged(["blockMesh"], cwd=case_dir, log_name="log.blockMesh")
            run_logged(["snappyHexMesh", "-overwrite"], cwd=case_dir, log_name="log.snappyHexMesh")
            if args.check_mesh:
                run_logged(["checkMesh"], cwd=case_dir, log_name="log.checkMesh")
            run_logged([args.solver], cwd=case_dir, log_name=f"log.{args.solver}")

            final_time, outlet_phi = latest_surface_value(case_dir, "outletFlux")
            outlet_phi_per_width = flux_per_width(outlet_phi, point.ly_mm)
            status = "ok"
        except subprocess.CalledProcessError as exc:
            final_time = None
            outlet_phi_per_width = None
            status = f"failed: {' '.join(exc.cmd)}"

    return StudyResult(
        # case=name,
        lx_mm=point.lx_mm,
        ly_mm=point.ly_mm,
        disc_mm=point.disc_mm,
        factor=point.factor,
        pressure_grad=dp,
        final_time=final_time,
        outlet_flux_phi_per_width=outlet_phi_per_width,
        status=status,
    )


def write_summary(path: Path, results: list[StudyResult]) -> None:
    if not results:
        return

    fieldnames = list(asdict(results[0]).keys())
    write_header = not path.exists() or path.stat().st_size == 0

    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="") as f:
            existing_header = next(csv.reader(f), None)
        if existing_header != fieldnames:
            raise ValueError(
                f"Existing summary header in {path} does not match current output fields: "
                f"{existing_header} != {fieldnames}"
            )

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run field-size and background-mesh sweeps for micro/example."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--lx", type=parse_float_list, default=parse_float_list("10,20,40"), help="comma-separated x sizes [mm]")
    parser.add_argument("--ly", type=parse_float_list, default=None, help="comma-separated y sizes [mm]; defaults to --lx")
    parser.add_argument("--disc", type=parse_float_list, default=parse_float_list("0.2"), help="comma-separated SimFrac/STL spacings [mm]")
    parser.add_argument("--factor", type=parse_float_list, default=parse_float_list("4"), help="comma-separated background mesh z-cell factors [cells/mm]")
    parser.add_argument("--aperture", type=float, default=1.0, help="mean aperture [mm]")
    parser.add_argument("--roughness", type=float, default=0.3)
    parser.add_argument("--shear", type=float, default=2.0)
    parser.add_argument("--pressure-gradient", type=parse_float_list, default=parse_float_list("0.05"))
    parser.add_argument("--solver", default="simpleFoam")
    parser.add_argument("--check-mesh", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="generate cases without running OpenFOAM")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    ly_values = args.ly if args.ly is not None else args.lx

    args.run_root.mkdir(parents=True, exist_ok=True)
    summary = args.run_root / "summary.csv"

    for lx_mm, ly_mm in zip(args.lx, ly_values):
        for disc_mm in args.disc:
            for factor in args.factor:
                for pressure_grad in args.pressure_gradient:
                    point = StudyPoint(
                        lx_mm=lx_mm,
                        ly_mm=ly_mm,
                        disc_mm=disc_mm,
                        factor=factor,
                        pressure_grad=pressure_grad,
                    )
                    print(f"Running {case_name(point)}")
                    result = run_study_case(args.run_root, point, args)
                    print(f"  {result.status}")
                    write_summary(summary, [result])
                    print(f"  wrote {summary}")

    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
