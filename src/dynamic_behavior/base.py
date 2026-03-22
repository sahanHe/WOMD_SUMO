from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path
import pickle
import numpy as np

import sumolib
from src.utils.pack_h5_womd import pack_scenario_raw
from . import roadedge_mapper
from .agents import *
from .roadedge_mapper import RoadEdgeMapper


class SUMOBaselineBase(ABC):

    def __init__(
        self,
        scenario,
        sumo_cfg_path: Union[str, Path],
        sumo_net_path: Union[str, Path],
        sim_wallstep: int = 91,
        history_length: int = 11,
    ) -> None:
        """
        Load a specific scenario data into the class instance. This call must be done before calling .run_simulations(initial_batch_cpu).
        """

        self.sumo_cfg_path = sumo_cfg_path
        self.scenario = scenario
        self.scenario_id: str = scenario.scenario_id

        self.sumonet: sumolib.net.Net = sumolib.net.readNet(
            sumo_net_path, withInternal=True, withConnections=True, withPrograms=False
        )
        self.episode: dict[str, np.ndarray] = pack_scenario_raw(scenario)

        self.sim_wallstep = sim_wallstep
        self.history_length = history_length

        # warmup period
        self.warmup_time_buff_agents, self.warmup_time_buff_tls = self.convert_batch_to_time_buff(
            start_step=0, end_step=history_length
        )
        # register agents to be simulated
        self.agent_attrs, self.agent_states = self._register_agents(self.history_length)

    def run_simulation(
        self,
        offroad_mapping: bool = True,
        gui: bool = False,
        sumo_oracle_parameters: dict = {},
        offroad_mapping_parameters: dict = {},
    ) -> tuple[dict[str, AgentAttr], dict[str, list[AgentState]]]:
        """
        Runs a single simulation based on the provided parameters.

        Args:
            - offroad_mapping (bool): If True, performs offroad mapping for agents. Default is True.
            - gui (bool): If True, runs the simulation with a graphical user interface. Default is False.
            - verbose (bool): If True, prints detailed simulation information. Default is True.
            - save_vis (bool): If True, saves a visualization video of the simulation. Default is True.
            - save_traj_and_tls (bool): If True, saves the trajectory and traffic light states to a pickle file. Default is False.
            - video_base_dir (Union[str, Path, None]): Directory to save the visualization video. Default is None.
            - video_name (Union[str, Path, None]): Name of the visualization video file. Default is None.
            - sumo_oracle_parameters (dict): Parameters for the SUMO oracle. Default is an empty dictionary.

        Returns:
            - tuple[dict[str, AgentAttr], dict[str, list[AgentState]]]: A tuple containing agent attributes and agent states.
        """

        print(f"simulation {self.scenario_id}".center(60, "-"))

        agent_attrs, agent_states, time_buff_sumo_tls, stationary_agent_ids = self.sumo_oracle(
            sumo_oracle_parameters, gui
        )
        if offroad_mapping:
            mapper = RoadEdgeMapper(self.scenario, **offroad_mapping_parameters)
            agent_states = mapper.perform_mapping(agent_attrs, agent_states, stationary_agent_ids)

        # do some conversion
        one_sim_time_buff_agents: list[list[Agent]] = [
            self._get_veh_list(agent_attrs, agent_states, t) for t in range(self.sim_wallstep)
        ]
        one_sim_time_buff_tls: list[list[Agent]] = self.warmup_time_buff_tls + time_buff_sumo_tls
            
        return agent_attrs, agent_states, one_sim_time_buff_agents, one_sim_time_buff_tls

    @abstractmethod
    def sumo_oracle(
        self,
        parameters: dict,
        gui: bool = False,
    ) -> tuple[list[list[Agent]], list[list[TrafficLightHead]]]:
        pass

    @staticmethod
    def _get_veh_list(agent_attrs: dict[str, AgentAttr], agent_states: dict[str, list[AgentState]], t: int):
        agent_list = []

        for v_id in agent_attrs.keys():
            agent_attr = agent_attrs[v_id]
            agent_state = agent_states[v_id][t]
            if agent_state:
                agent_list.append(
                    Agent(
                        id=v_id,
                        size=agent_attr.size,
                        agent_type=agent_attr.type,
                        agent_role=agent_attr.role,
                        route=agent_attr.route,
                        control_state=agent_state.control_state,
                        location=agent_state.location,
                        heading=agent_state.heading,
                        speed=agent_state.speed,
                        acceleration=agent_state.acc,
                        yaw_rate=agent_state.yaw_rate,
                    )
                )
        return agent_list

    def convert_batch_to_time_buff(
        self, start_step: int = 0, end_step: int = 11
    ) -> tuple[list[list[Agent]], list[list[TrafficLightHead]]]:

        time_buff = [[] for _ in range(end_step - start_step)]
        time_buff_tls = [[] for _ in range(end_step - start_step)]

        # for every time step
        for t in range(start_step, end_step):
            # agents
            for i in range(self.episode["agent/valid"].shape[0]):
                if not self.episode["agent/valid"][i, t].item():
                    continue

                agent_waymo_id = str(self.episode["agent/object_id"][i].item())
                pos = list(self.episode["agent/pos"][i, t])
                theta = self.episode["agent/yaw_bbox"][i, t].item()
                speed = self.episode["agent/spd"][i, t].item()
                acc = self.episode["agent/acc"][i, t].item()
                yaw_rate = self.episode["agent/yaw_rate"][i, t].item()
                agent_size = list(self.episode["agent/size"][i].astype(float))
                agent_type = AgentType(np.argmax(self.episode["agent/type"][i]).item())
                agent_role_one_hot = self.episode["agent/role"][i]
                # check if agent_role_one_hot is all False
                if np.all(~agent_role_one_hot):  # all False
                    agent_role = AgentRole.NORMAL
                else:
                    agent_role = AgentRole(np.argmax(agent_role_one_hot).item())

                time_buff[t].append(
                    Agent(
                        id=agent_waymo_id,
                        size=Size3d(*agent_size),
                        location=pos,
                        heading=theta,
                        speed=speed,
                        acceleration=acc,
                        agent_type=agent_type,
                        yaw_rate=yaw_rate,
                        agent_role=agent_role,
                        control_state=None,
                    )
                )

            # traffic lights
            for i in range(self.episode["tl_stop/valid"].shape[0]):  # n_tl_stop, n_step
                if not self.episode["tl_stop/valid"][i, t].item():
                    continue
                pos = list(self.episode["tl_stop/pos"][i])
                state = np.argmax(self.episode["tl_stop/state"][i, t])
                tl_dir = list(self.episode["tl_stop/dir"][i])
                tl_id = str(i)
                time_buff_tls[t].append(TrafficLightHead(id=tl_id, location=pos, tl_dir=tl_dir, state=state))

        return time_buff, time_buff_tls

    def _register_agents(
        self, history_length: int
    ) -> tuple[dict[str, AgentAttr], dict[str, list[AgentState]]]:
        """
        Registers agents and their historical states from the episode data.
        Args:
            history_length (int): The length of the history to consider for each agent.
        Returns:
            tuple: A tuple containing:
            - dict[str, AgentAttr]: A dictionary mapping agent IDs to their attributes.
            - dict[str, list[AgentState]]: A dictionary mapping agent IDs to a list of their historical states.
        """

        agent_attrs: dict[str, AgentAttr] = {}
        agent_states: dict[str, list[AgentState]] = {}

        for i in range(self.episode["agent/object_id"].shape[0]):
            if not self.episode["agent/valid"][i, history_length - 1]:
                continue

            # register this agent to [self.agent_info]
            agent_id = str(self.episode["agent/object_id"][i].item())
            agent_size = list(self.episode["agent/size"][i].astype(float))
            agent_type = AgentType(np.argmax(self.episode["agent/type"][i]).item())
            agent_role_one_hot = self.episode["agent/role"][i]
            agent_role = (
                AgentRole.NORMAL
                if np.all(~agent_role_one_hot)
                else AgentRole(np.argmax(agent_role_one_hot).item())
            )

            agent_attrs[agent_id] = AgentAttr(
                id=agent_id, size=agent_size, agent_type=agent_type, agent_role=agent_role
            )

            # find the history agent trajectory
            agent_states[agent_id] = []
            for t in range(history_length):
                if self.episode["agent/valid"][i, t]:
                    pos = list(self.episode["agent/pos"][i, t])
                    theta = self.episode["agent/yaw_bbox"][i, t].item()
                    speed = self.episode["agent/spd"][i, t].item()
                    acc = self.episode["agent/acc"][i, t].item()
                    yaw_rate = self.episode["agent/yaw_rate"][i, t].item()
                    agent_states[agent_id].append(
                        AgentState(
                            location=pos, heading=theta, speed=speed, acceleration=acc, yaw_rate=yaw_rate
                        )
                    )
                else:
                    agent_states[agent_id].append(None)

        return agent_attrs, agent_states
