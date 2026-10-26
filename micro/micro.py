import os
import sys
from copy import copy
from pathlib import Path
from shutil import copy2, copytree, rmtree
from argparse import ArgumentParser

import numpy as np
from foamlib import DimensionSet, FoamCase, FoamFile
from foamlib.postprocessing.load_tables import functionobject, load_tables
from rich import get_console

from example_flow.makeFractureSTL import make_fitted_from_top_bot, makeFracture
from compute_dispersion import breakthrough_properties
from cacheDB import MicroCache


cGradTol = 1e-5
pGradTol = 1e-1

switch_on_scalar_solver = os.getenv("solve_scalar", "False").lower()
run_scalar_solver = switch_on_scalar_solver in ("true", "1", "yes", "y")
switch_on_flow_solver = os.getenv("solve_flow", "False").lower()
run_flow_solver = (
    switch_on_flow_solver in ("true", "1", "yes", "y") or run_scalar_solver
)


get_console().quiet = True


class MicroSimulation:
    _shared_cache = MicroCache("./micro-cache/results.sqlite")

    def __init__(self, sim_id):
        """
        Constructor of MicroSimulation class.
        """
        self._sim_id = sim_id
        self._root = Path(f"./micro-runs/{sim_id}")
        self._root_path_flow = self._root / "flow"
        self._root_path_scalar = self._root / "scalar"

        self._flow_case_created = self._root_path_flow.is_dir()
        self._scalar_case_created = self._root_path_scalar.is_dir()

    def initialize(self):
        return {"pressure-grad": 0.0, "tracer-fraction-grad": 0.0}

    def solve(self, macro_data, dt):
        use_cache = True

        dx = 0.04
        dp = float(macro_data["pressure-grad"])
        thickness = float(macro_data["aperture"])
        tracer_fraction_grad = float(macro_data["tracer-fraction-grad"])
        print("micro input aperture:", abs(thickness), " and dp", abs(dp), flush=True)

        flow_solver_active = run_flow_solver and abs(dp) > pGradTol
        scalar_solver_active = (
            flow_solver_active
            and run_scalar_solver
            and abs(tracer_fraction_grad) > cGradTol
        )
        cache_inputs = {
            "pressure_gradient": abs(dp),
            "mean_aperture": thickness,
            "scalar_solver_active": scalar_solver_active,
        }

        micro_flux = dp * thickness**3 / (12.0 * 1e-3)
        micro_dispersion = 0

        if flow_solver_active:
            if use_cache:
                micro_flux_cached, micro_dispersion_cached = self._shared_cache.lookup(
                    cache_inputs
                )
                if micro_flux_cached is not None:
                    return {
                        "flux": micro_flux_cached,
                        "dispersion": micro_dispersion_cached,
                    }

            self.copy_flow_case()
            fc = FoamCase(self._root_path_flow)

            # generate fracture geometry and mesh
            x, y, t, b = makeFracture(
                aperture=thickness * 1000,  # it uses mm
                shear=1.0,
                disc=0.5,
                roughness=0.2,
                lambda_abs=2.0,
                lx=dx * 1000,
            )
            make_fitted_from_top_bot(
                x, y, t, b, self._root_path_flow / "system/blockMeshDict", 2, 16
            )

            # set pressure difference between inlet and outlet
            fc[0]["p"].boundary_field["inlet"].value = abs(dp * dx / 1000.0)

            fc.run()

            # read flux am outlet from postprocessing results
            file = functionobject(
                file_name="surfaceFieldValue.dat", folder="outletFlowRate"
            )
            fluxes = load_tables(source=file, dir_name=self._root_path_flow)
            flux = fluxes.iloc[-1]["sum(phi)"]  # m^3/s
            micro_flux = flux / 0.01

            # read max.(Co) to be used for time step approximation in scalarTransportFoam
            file = functionobject(file_name="volFieldValue.dat", folder="fieldMaxCo")
            maxCo = load_tables(source=file, dir_name=self._root_path_flow).iloc[-1][
                "max(Co)"
            ]

            if scalar_solver_active:
                print("=====run scalar solve=======")
                self.copy_scalar_case()
                fs = FoamCase(self._root_path_scalar)
                latest_time = fc[-1]
                scalar_zero = fs[0].path
                copy2(latest_time["p"].path, scalar_zero / "p")
                copy2(latest_time["U"].path, scalar_zero / "U")
                copy2(latest_time["phi"].path, scalar_zero / "phi")

                scalar_mesh = fs.path / "constant" / "polyMesh"
                copytree(fc.path / "constant" / "polyMesh", scalar_mesh)
                scalar_time_step = 1.0 / maxCo
                with FoamFile(fs.path / "system" / "controlDict") as f:
                    f["deltaT"] = scalar_time_step
                    f["endTime"] = 5000 * scalar_time_step

                fs.run()

                # read T flux at outlet
                micro_dispersion = breakthrough_properties(
                    self._root_path_scalar, dx, "outletFlux"
                )

                fc.clean(check=True)
                fs.clean(check=True)
            else:
                print("=====skip scalar solve==")
                fc.clean(check=True)
            if use_cache:
                self._shared_cache.store(
                    cache_inputs, {"flux": micro_flux, "dispersion": micro_dispersion}
                )
                if scalar_solver_active:  # save the flow only solve too
                    cache_inputs_flow = {
                        "pressure_gradient": abs(dp),
                        "mean_aperture": thickness,
                        "scalar_solver_active": 0,
                    }
                    self._shared_cache.store(
                        cache_inputs_flow, {"flux": micro_flux, "dispersion": 1e-9}
                    )

        # print("micro dispersion coefficient:", micro_dispersion, " bei flux", micro_flux," tracer grad:", abs(tracer_fraction_grad), " and dp", abs(dp))
        return {"flux": micro_flux, "dispersion": micro_dispersion}

    def copy_flow_case(self):
        if self._flow_case_created:
            return

        flow_exists = self._root_path_flow.exists()
        if flow_exists:
            raise RuntimeError(
                f"Incomplete OpenFOAM case for simulation {self._sim_id}: flow exists"
            )

        FoamCase("./example_flow").clone(self._root_path_flow)
        self._flow_case_created = True
        print(f"Sim {self._sim_id} flow case created")

    def copy_scalar_case(self):
        if self._scalar_case_created:
            return

        scalar_exists = self._root_path_scalar.exists()
        if scalar_exists:
            raise RuntimeError(
                f"Incomplete OpenFOAM case for simulation {self._sim_id}: scalar solver exists"
            )

        FoamCase("./example_scalar").clone(self._root_path_scalar)
        self._scalar_case_created = True
        print(f"Sim {self._sim_id} scalar case created.")

    def set_state(self, state):
        pass

    def get_state(self):
        pass

    def get_global_id(self):
        return self._sim_id

    def set_global_id(self, global_id):
        self._sim_id = global_id

    def output(self):
        pass
