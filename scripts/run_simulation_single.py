import copy
import os

from pathlib import Path
from deepmerge import always_merger
import numpy as np
import sumolib

from src.dynamic_behavior.agents import AgentAttr, AgentRole, AgentState, AgentType, TrafficLightHead
from src.dynamic_behavior.geometry import center_to_front_bumper_position, rad2degree, wrap_radians
from src.dynamic_behavior.sumo_baseline import SUMOBaseline
from src.dynamic_behavior.sumo_baseline_log import SUMOBaselineLog
from src.static_context.utils.generic import Pt
from src.utils.read_data import read_data
from src.utils.pack_h5_womd import pack_scenario_raw

import traci

from src.utils.sumobaseline_visualizer import SUMOBaselineVisualizer





history_length = 91

scenario_list = read_data(
    "/home/sahan/Desktop/PhD/waymo_motion_training/uncompressed_scenario_validation_validation.tfrecord-00000-of-00150",
)
scenario_id = "368ae7945c8202ec"
scenario_assigned = None
for i, scenario in enumerate(scenario_list):
    if scenario.scenario_id == scenario_id:
        print(f"Found scenario with ID: {scenario_id}")
        scenario_assigned = scenario

womd_scenario_path = f"/home/sahan/Desktop/PhD/waymo_motion_training/uncompressed_scenario_validation_validation.tfrecord-00000-of-00150"
base_dir = Path(f"/home/sahan/Desktop/PhD/SUMO-Benchmark/outputs/")

sumo_cfg_path = Path(base_dir / f"{scenario_id}/{scenario_id}.sumocfg")
sumo_net_path = Path(base_dir / f"{scenario_id}/{scenario_id}.net.xml")


simulator = SUMOBaselineLog(scenario_assigned, sumo_cfg_path, sumo_net_path, sim_wallstep=91)
agent_attrs, agent_states, one_sim_time_buff_agents, one_sim_time_buff_tls = simulator.run_simulation(
    offroad_mapping=True,
    gui=False,
    sumo_oracle_parameters={}
)

visualizer = SUMOBaselineVisualizer(scenario_assigned, shaded=True, types_to_draw="withedge", simplify_agent_color=False)
visualizer.save_map(base_dir="sim_out_new/",) # save the map picture
visualizer.save_video(
    one_sim_time_buff_agents,
    one_sim_time_buff_tls,
    base_dir="sim_out_new/",
    video_name=f"{scenario_id}.mp4",
    with_ground_truth=True
) # save the simulation as a video
                                                   

