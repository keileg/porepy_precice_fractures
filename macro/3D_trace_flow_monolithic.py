from __future__ import annotations

import porepy as pp
from porepy.compositional.compositional_mixins import CompositionalVariables
from porepy.models.compositional_flow import (
    ComponentMassBalanceEquations,
)

from shared_flow import TracerBC, TracerFluid, TracerIC, ModifiedGeometry


class SinglePhaseFlowGeometry(
    ModifiedGeometry,
    TracerFluid,
    CompositionalVariables,
    ComponentMassBalanceEquations,
    TracerIC,
    TracerBC,
    pp.constitutive_laws.CubicLawPermeability,
    pp.SinglePhaseFlow,
):
    pass


fluid_constants = pp.FluidComponent(viscosity=1.0e-3, density=1000.0)
solid_constants = pp.SolidConstants(
    permeability=1e-10, normal_permeability=1e-8, residual_aperture= 0.001
)
material_constants = {"fluid": fluid_constants, "solid": solid_constants}
model_params = {
    "material_constants": material_constants,
    "time_manager": pp.TimeManager(
        schedule=[0.0, 300*pp.MINUTE],
        dt_init=pp.MINUTE,
        dt_min_max=(0.01*pp.MINUTE, 10*pp.MINUTE),
        constant_dt=False,
    ),
}

model = SinglePhaseFlowGeometry(model_params)
pp.ModelRunner(model).run()
for intf, data in model.mdg.interfaces(return_data=True):
            print(data[pp.TIME_STEP_SOLUTIONS][model.interface_darcy_flux_variable])