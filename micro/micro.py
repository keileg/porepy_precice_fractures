import sys
from copy import copy
from pathlib import Path
from shutil import copy2, copytree, rmtree

import numpy as np
from foamlib import DimensionSet, FoamCase, FoamFile
from foamlib.postprocessing.load_tables import functionobject, load_tables

import example_flow.makeBlockMesh as mesh
from example_flow.makeFractureSTL import (make_background_from_top_bot,
                                     make_stl_from_top_bot, makeFracture,
                                     patch_midpoint)
from compute_dispersion import breakthrough_properties

class MicroSimulation():
    def __init__(self, sim_id):
        """
        Constructor of MicroSimulation class.
        """
        self._sim_id = sim_id
        self._root =  Path(f"./micro-runs/{sim_id}")
        self._root_path_flow = self._root / "flow"
        self._root_path_scalar = self._root / "scalar"
        self._width = 0.005 # width of the channel

        FoamCase("./example_flow").clone(self._root_path_flow)
        FoamCase("./example_scalar").clone(self._root_path_scalar)
        print(f"Sim {sim_id} created {self._root}")

    def initialize(self, initial_data=None):
        return {"pressure-grad": 0.0}

    def solve(self, macro_data, dt):
        dp = macro_data["pressure-grad"]
        dp_input = abs(dp * 0.01 / 1000.0) # compute dp from gradient, then convert to kinetic pressure, force flow in positive direction
        
        if abs(dp) < 1e-15:
            return {"flux": 0.0}
            
        fc = FoamCase(self._root_path_flow)
        thickness=macro_data["aperture"]
        # use channel geometry
        # mesh.meshFor(nx=10,amplitude=0,waves=1,aperture=thickness,shift=0).write(self._root_path_flow + "/system/blockMeshDict")

        # use fracture geometry
        x, y, t, b = makeFracture(
            aperture=thickness * 1000, # it uses mm
            shear=1.0,
            disc=1,
            roughness=0.2,
            lx=40,
            ly=40,
        )
        make_stl_from_top_bot(x, y, t, b, self._root_path_flow / "constant/triSurface/fracture.stl")
        make_background_from_top_bot(x, y, t, b, self._root_path_flow / "system/blockMeshDict", 2)
        patch_midpoint(x, y, t, b, self._root_path_flow / "system/snappyHexMeshDict")

        # overwrite pressure at inlet
        with fc[0]["p"] as f:
            f.dimensions = DimensionSet(length=2, time=-2)
            f.internal_field = 0.0
            f.boundary_field = {
                "inlet": {"type": "fixedValue", "value": dp_input},
                "outlet": {"type": "fixedValue", "value": 0},
                "upperWall": {"type": "zeroGradient"},
                "lowerWall": {"type": "zeroGradient"},
                "frontAndBack": {"type": "zeroGradient"}, # use "empty" for channel flow
            }

        fc.run()

        file = functionobject(file_name="surfaceFieldValue.dat", folder="outletFlowRate")
        fluxes = load_tables(source=file, dir_name=self._root_path_flow)
        flux = fluxes.iloc[-1]["sum(phi)"]  # m^3/s
        flux_per_width = flux / self._width
        flux_ana = dp * thickness * thickness / (12.0 * 1e-3) * thickness
        diff = abs(flux_ana-flux_per_width)
        print("=====flux on sim ", self._sim_id, "with p_in ", dp_input, " flux", flux_per_width, "with diff ", diff, "===")

        fs = FoamCase(self._root_path_scalar)
        latest_time = fc[-1]
        scalar_zero = fs[0].path
        copy2(latest_time["p"].path, scalar_zero / "p")
        copy2(latest_time["U"].path, scalar_zero / "U")

        scalar_mesh = fs.path / "constant" / "polyMesh"
        copytree(fc.path / "constant" / "polyMesh", scalar_mesh)
        fs.run()

        # read T flux at outlet
        file = functionobject(file_name="surfaceFieldValue.dat", folder="outletFlux")
        fluxes = load_tables(source=file, dir_name=self._root_path_scalar)

        dispersion = breakthrough_properties(fluxes, 0.04)

        fc.clean(check=True)
        fs.clean(check=True)

        return {"flux": flux_per_width}

    def set_state(self, state):
        self._root_path = copy(state[0])

    def get_state(self):
        return copy([self._root_path])

    def get_global_id(self):
        return self._sim_id

    def set_global_id(self, global_id):
        self._sim_id = global_id

    def output(self):
        pass 
