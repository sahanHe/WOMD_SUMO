from pathlib import Path
import sys
sys.path.append(".")

from src.utils.read_data import read_data
from src import Waymo2SUMO

# read all scenarios from a tfrecord shard, the result is a list of scenarios
# FIXME: replace the path to a WOMD tfrecord shard file
scenario_list = read_data(
    "//home/sahan/Desktop/PhD/waymo_motion_training/uncompressed_scenario_validation_validation.tfrecord-00000-of-00150",
)

for i, scenario in enumerate(scenario_list):
    
    # initiantiate the Waymo2SUMO class. upon instantiation, the class will convert the scenario to data that is needed for the corresponding SUMO .net.xml file
    waymo2sumo = Waymo2SUMO(scenario)
    
    # what you can do now:
    # 1. get a very rough visualization using matplotlib
    waymo2sumo.plot_sumonic_map(base_dir=f"outputs/{scenario.scenario_id}", filename=f"{scenario.scenario_id}.png")
    # 2. save the .net.xml file
    waymo2sumo.save_SUMO_netfile(base_dir=f"outputs/{scenario.scenario_id}", filename=f"{scenario.scenario_id}.net.xml")
    # 3. or get a more detailed visualization using either sumo-gui,
    # or using SUMONetVis package available here (https://github.com/patmalcolm91/SumoNetVis)