from .waymo import Waymonic
from .sumo import Sumonic

class Waymo2SUMO(Waymonic, Sumonic):
    def __init__(self,
                 scenario,
                 sumonize_config = {},
                 ) -> None:
        print(f"--------------------SCENARIO ID: {scenario.scenario_id}-------------------")
        self.scenario = scenario
        Waymonic.__init__(self,scenario)
        Sumonic.__init__(self, scenario, self.lanecenters, sumonize_config)
