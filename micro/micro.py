from pathlib import Path

import numpy as np
from foamlib import DimensionSet, FoamCase
from foamlib.postprocessing.load_tables import load_tables, functionobject
import sys
from copy import copy

class MicroSimulation():
    def __init__(self, sim_id):
        """
        Constructor of MicroSimulation class.
        """
        self._sim_id = sim_id
        self._root_path = f"./micro-runs/micro-{sim_id}"
        FoamCase("./example").clone(self._root_path)
        print(f"Sim {sim_id} created {self._root_path}")

    def initialize(self, initial_data=None):
        return {"pressure-difference": 0.0}

    def solve(self, macro_data, dt):
        dp = macro_data["pressure-difference"]
        
        if dp == 0:
            return {"flux": 0.0}
            
        fc = FoamCase(self._root_path)

        with fc[0]["p"] as f:
            f.dimensions = DimensionSet(length=2, time=-2)
            f.internal_field = 0.0
            f.boundary_field = {
                "inlet": {"type": "fixedValue", "value": dp},
                "outlet": {"type": "fixedValue", "value": 0},
                "upperWall": {"type": "zeroGradient"},
                "lowerWall": {"type": "zeroGradient"},
                "frontAndBack": {"type": "empty"},
            }

        fc.run()

        file = functionobject(file_name="surfaceFieldValue.dat", folder="outletFlux")
        fluxes = load_tables(source=file, dir_name=self._root_path)
        flux = fluxes["sum(phi)"][1]
        print("flux on sim ", self._sim_id, " flux", flux)
        perturbed_flux = flux * (self._sim_id / 16 + 0.5)
        fc.clean(check=True)

        return {"flux": perturbed_flux}

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
