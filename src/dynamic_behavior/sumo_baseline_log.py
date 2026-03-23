import numpy as np
from typing import Union
import random
import copy
from pathlib import Path
import sys
import os

import libsumo as traci


import traci
from sumolib import net

from .geometry import (
    wrap_radians,
    center_to_front_bumper_position,
    front_bumper_to_center_position,
    rad2degree,
    degree2rad,
)
from .agents import *
from .baselog import SUMOBaselineBaseLog
from .veh_params import *
from deepmerge import always_merger
from src.static_context.utils import Pt


class SUMOBaselineLog(SUMOBaselineBaseLog):
    """
    This class defines sumo_oracle() method

    The simulation works like this:
    at very moment, each agent is in one of the four possible simiulation states:
        |--------|---------------|-----------------|
        |        |   immobile    |  NOT immobile   |
        |--------|---------------|-----------------|
        |        |      Q2       |      Q1         |
        | onroad |      🔄       |      🔄⬇️        |
        |        |               |                 |
        |--------|---------------|-----------------|
        |        |      Q3       |      Q4         |
        |offroad |      🔄       |      🔄⬆️        |
        |        |               |                 |
        |--------|---------------|-----------------|
        the arrows show how each agent will change its simulation state from one step to the next:
        - the mobility state will not change, it is determined at the beginning;
        - for an immobile agent, the onroad state will not change (Q2->Q2, Q3->Q3), since they are hold still;
        - an mobile agent previously onroad in the prior step might be offroad (Q1->Q4)
        - for a mobile but offroad agent (Q4), we always try to make it onroad (Q4->Q1)

        the code below has a few lists of agent ids, which represent the following, respectively:
        - self.immobile_agent_ids: Q2+Q3
        - onroad_agent_ids: Q1+Q2
        - remaining_agent_ids: Q3+Q4
        -
    """

    def __init__(
        self,
        scenario,
        sumo_net_path: Union[str, Path],
        sim_wallstep: int = 91,
        history_length: int = 91,
    ) -> None:
        super().__init__(scenario, sumo_net_path, sim_wallstep, history_length)

        self.default_parameters = {
            "agent_speedfactor_std": 0.15,
            "vehicle_min_speedfactor": 0.6,
            "pedestrian_min_speedfactor": 0.3,
            "treat_ped_as_vehicle": True,
            "randomize_simplemove": False,
            "acceleration_std": 0.2,
            "deceleration_mean": 3,
            "deceleration_std": 0.2,
            "lateral_resolution": 0.2,
            "outofbound_lanecenter_threshold": 5,
            "outofbound_roadedge_threshold": 5,
            "sideroad_probability": 0.1,
        }

        self.time_buff_sumo_tls: list[list[TrafficLightHead]] = []
        self.stationary_agent_ids: list[str] = []
        self.outofbound_agent_ids: list[str] = []
        self.offroad_agent_ids: list[str] = []
        self.immobile_agent_ids: list[str] = []
    
    def set_enviorenment_data(self, parameters: dict = {}) -> dict:

        self.route_counter = 0
        self.parameters: dict = always_merger.merge(self.default_parameters, parameters)
        # ------------------ 2. Setup vehicles & tls
        self.stationary_agent_ids: list[str] = [
            v_id for v_id in self.agent_attrs.keys() if self._is_agent_stationary(self.agent_states[v_id])
        ]
        self.outofbound_agent_ids: list[str] = [
            v_id for v_id in self.agent_attrs.keys() if self._is_agent_outofbound(self.agent_states[v_id], consider_roadedge=False)
        ]
        self.immobile_agent_ids: list[str] = list(set(self.stationary_agent_ids).union(self.outofbound_agent_ids))

        self.offroad_agent_ids = set(self.agent_attrs.keys())
        agent_attrs = copy.deepcopy(self.agent_attrs)
        agent_states = copy.deepcopy(self.agent_states)


        return agent_attrs, agent_states
    

    def place_agents(
        self, agent_attrs, agent_states, t: int
    ) -> tuple[dict[str, AgentAttr], dict[str, list[AgentState]], list[list[TrafficLightHead]]]:
        

        



        print(f"-----------------{t}-------------------")

        # store the state of vehicles for a single time step; updated every time
        self.control_state_dict: dict[str, int] = {v_id: None for v_id in agent_attrs.keys()}

            # [explicit command] for vehicles dropped in the last step, we try to add it back
        for agent_id in self.offroad_agent_ids.difference(self.outofbound_agent_ids):
            if agent_states[agent_id][0] is None:
                continue
            if agent_attrs[agent_id].type == AgentType.VEHICLE or self.parameters["treat_ped_as_vehicle"]:
                self.place_a_vehicle(agent_attrs[agent_id], agent_states[agent_id], t)
            else:
                self.place_a_pedestrian(agent_attrs[agent_id], agent_states[agent_id])

            # [explicit command] for stationary vehicles
            onroad_agent_ids = set(agent_attrs.keys()) - self.offroad_agent_ids
            immobile_and_onroad_agent_ids = onroad_agent_ids.intersection(self.immobile_agent_ids)
            for agent_id in immobile_and_onroad_agent_ids:
                self.control_immobile_agent(agent_attrs[agent_id], agent_states[agent_id])

            # [explicit command] for vehicles at intersection
            for node in self.sumonet.getNodes():
                node_type = node.getType()
                if node_type in ["traffic_light", "traffic_light_right_on_red"]:
                    self.control_veh_behavior_at_intersection(node)

            # get simulation result
            # traci.simulationStep()
            onroad_agent_states = self.get_onroad_agents(agent_states, self.offroad_agent_ids)
            self.offroad_agent_ids = set(agent_attrs.keys()) - set(onroad_agent_states.keys())

            # [state obtain] for successfully simulated (onroad) agents, we store whatever is obtained
            for agent_id, agent_state in onroad_agent_states.items():  # (Q1 + Q2)
                agent_states[agent_id].append(agent_state)

            # [state obtain] for unsuccessfully simulated (offroad) agents, we apply a simple control
            for agent_id in self.offroad_agent_ids:
                agent_state = self.predict_agent_simple_movement(
                    agent_attrs[agent_id],
                    agent_states[agent_id],
                    random=self.parameters["randomize_simplemove"],
                )
                if agent_id in self.immobile_agent_ids:  # (Q3)
                    agent_state.control_state = AgentControlState.SIMPLE_STATIONARY
                else:  # (Q4)
                    agent_state.control_state = AgentControlState.SIMPLE_MOVE
                agent_states[agent_id].append(agent_state)

            # [state obtain] Get the current traffic light states
            tls_list = self.get_tls_list_from_SUMO()
            self.time_buff_sumo_tls.append(tls_list)

            # for veh_id in traci.vehicle.getIDList():
            #     print(f"Vehicle ID: {veh_id}")
            #     print(f"Vehicle ID: {veh_id}, Position: {traci.vehicle.getPosition(veh_id)}")
            #     print(f"Vehicle ID: {veh_id}, Lane ID: {traci.vehicle.getLaneID(veh_id)}")
            #     print(f"Vehicle ID: {veh_id}, Speed: {traci.vehicle.getSpeed(veh_id)}")
            #     print(f"Vehicle ID: {veh_id}, Heading: {traci.vehicle.getAngle(veh_id)}")
            #     print(f"Vehicle ID: {veh_id}, Acceleration: {traci.vehicle.getAcceleration(veh_id)}")


        
        return agent_attrs, agent_states, self.time_buff_sumo_tls, self.immobile_agent_ids



    def sumo_oracle(
        self, parameters: dict = {}, gui: bool = False
    ) -> tuple[dict[str, AgentAttr], dict[str, list[AgentState]], list[list[TrafficLightHead]]]:

        self.route_counter = 0
        self.parameters: dict = always_merger.merge(self.default_parameters, parameters)

        # ------------------ 1. Start SUMO simulation


        # ------------------ 2. Setup vehicles & tls
        agent_attrs = copy.deepcopy(self.agent_attrs)
        agent_states = copy.deepcopy(self.agent_states)

        stationary_agent_ids: list[str] = [
            v_id for v_id in agent_attrs.keys() if self._is_agent_stationary(agent_states[v_id])
        ]
        outofbound_agent_ids: list[str] = [
            v_id for v_id in agent_attrs.keys() if self._is_agent_outofbound(agent_states[v_id], consider_roadedge=False)
        ]
        self.immobile_agent_ids: list[str] = list(set(stationary_agent_ids).union(outofbound_agent_ids))

        offroad_agent_ids = set(agent_attrs.keys())
        time_buff_sumo_tls: list[list[TrafficLightHead]] = []

        # the main loop
        for t in range(0, self.sim_wallstep):
            print(f"-----------------{t}-------------------")

            # store the state of vehicles for a single time step; updated every time
            self.control_state_dict: dict[str, int] = {v_id: None for v_id in agent_attrs.keys()}

            # [explicit command] for vehicles dropped in the last step, we try to add it back
            for agent_id in offroad_agent_ids.difference(outofbound_agent_ids):
                if agent_states[agent_id][0] is None:
                    continue
                if agent_attrs[agent_id].type == AgentType.VEHICLE or self.parameters["treat_ped_as_vehicle"]:
                    self.place_a_vehicle(agent_attrs[agent_id], agent_states[agent_id], t)
                else:
                    self.place_a_pedestrian(agent_attrs[agent_id], agent_states[agent_id])

            # [explicit command] for stationary vehicles
            onroad_agent_ids = set(agent_attrs.keys()) - offroad_agent_ids
            immobile_and_onroad_agent_ids = onroad_agent_ids.intersection(self.immobile_agent_ids)
            for agent_id in immobile_and_onroad_agent_ids:
                self.control_immobile_agent(agent_attrs[agent_id], agent_states[agent_id])

            # [explicit command] for vehicles at intersection
            for node in self.sumonet.getNodes():
                node_type = node.getType()
                if node_type in ["traffic_light", "traffic_light_right_on_red"]:
                    self.control_veh_behavior_at_intersection(node)

            # get simulation result
            traci.simulationStep()
            onroad_agent_states = self.get_onroad_agents(agent_states, offroad_agent_ids)
            offroad_agent_ids = set(agent_attrs.keys()) - set(onroad_agent_states.keys())

            # [state obtain] for successfully simulated (onroad) agents, we store whatever is obtained
            for agent_id, agent_state in onroad_agent_states.items():  # (Q1 + Q2)
                agent_states[agent_id].append(agent_state)

            # [state obtain] for unsuccessfully simulated (offroad) agents, we apply a simple control
            for agent_id in offroad_agent_ids:
                agent_state = self.predict_agent_simple_movement(
                    agent_attrs[agent_id],
                    agent_states[agent_id],
                    random=self.parameters["randomize_simplemove"],
                )
                if agent_id in self.immobile_agent_ids:  # (Q3)
                    agent_state.control_state = AgentControlState.SIMPLE_STATIONARY
                else:  # (Q4)
                    agent_state.control_state = AgentControlState.SIMPLE_MOVE
                agent_states[agent_id].append(agent_state)

            # [state obtain] Get the current traffic light states
            tls_list = self.get_tls_list_from_SUMO()
            time_buff_sumo_tls.append(tls_list)

            for veh_id in traci.vehicle.getIDList():
                print(f"Vehicle ID: {veh_id}")
                print(f"Vehicle ID: {veh_id}, Position: {traci.vehicle.getPosition(veh_id)}")
                print(f"Vehicle ID: {veh_id}, Lane ID: {traci.vehicle.getLaneID(veh_id)}")
                print(f"Vehicle ID: {veh_id}, Speed: {traci.vehicle.getSpeed(veh_id)}")
                print(f"Vehicle ID: {veh_id}, Heading: {traci.vehicle.getAngle(veh_id)}")
                print(f"Vehicle ID: {veh_id}, Acceleration: {traci.vehicle.getAcceleration(veh_id)}")


        
        return agent_attrs, agent_states, time_buff_sumo_tls, self.immobile_agent_ids

    """
    ############################### EDGE CASE CHECK ###############################
    """

    def _is_agent_stationary(self, agent_states: list[AgentState]) -> bool:

        # criteria 1: speed has been always 0
        is_speed_0 = self._is_speed_0(agent_states, threshold=0.2) # True if average speed of the agent in history is below Threshold
        agent_center_pos = np.array(agent_states[-1].get_position())
        # criteria 2: it is close to roadedge
        is_close_to_road_edge = self._min_map_feature_distance(self.episode, agent_center_pos, [4]) < 2.3
        # criteria 3: it is far from lanecenter
        is_far_from_lanecenter = (
            self._min_map_feature_distance(self.episode, agent_center_pos, [0, 1, 2]) > 2.0
        )
        return is_speed_0 and (is_close_to_road_edge or is_far_from_lanecenter)

    def _is_agent_outofbound(self, agent_states: list[AgentState], consider_roadedge: bool = True) -> bool:

        agent_center_pos = np.array(agent_states[-1].get_position())
        is_very_far_from_lanecenter = (
            self._min_map_feature_distance(self.episode, agent_center_pos, [0, 1, 2]) > self.parameters["outofbound_lanecenter_threshold"]
        )
        if consider_roadedge:
            is_very_far_from_road_edge = (
                self._min_map_feature_distance(self.episode, agent_center_pos, [4]) > self.parameters["outofbound_roadedge_threshold"]
            )
            return is_very_far_from_road_edge and is_very_far_from_lanecenter
        else:
            return is_very_far_from_lanecenter


    # Compute the mean of agent's speed history and check if it is below a certain threshold
    @staticmethod
    def _is_speed_0(agent_states: list[AgentState], threshold: float = 0.5) -> bool:
        agent_history_speed: list[float] = []
        for state in agent_states:
            if state != None:
                agent_history_speed.append(state.speed)
        # is_speed_0 = not any(speed > threshold for speed in agent_history_speed)
        is_speed_0 = np.mean(agent_history_speed) <= threshold
        return is_speed_0

    @staticmethod
    def _min_map_feature_distance(
        episode: dict[str, np.ndarray], position: np.ndarray, feature_type: list[int]
    ) -> tuple[float]:
        valid = episode["map/valid"]
        distances = np.linalg.norm(episode["map/pos"] - position, axis=2)

        # criteria 2: closed to any road edge
        type_mask = episode["map/type"][:, feature_type]
        type_mask = np.any(type_mask, axis=1)
        type_mask = np.tile(type_mask[:, np.newaxis], (1, episode["map/pos"].shape[1]))
        map_feature_distance = np.copy(distances)
        map_feature_distance[~valid] = np.inf
        map_feature_distance[~type_mask] = np.inf
        min_road_edge_distance = np.min(map_feature_distance)
        return min_road_edge_distance

    def predict_agent_simple_movement(
        self, agent_attr: AgentAttr, agent_states: list[AgentState], random=False
    ) -> AgentState:

        if agent_attr.id in self.immobile_agent_ids:
            return self.agent_states[agent_attr.id][self.history_length - 1]

        else:
            new_veh_state = copy.deepcopy(agent_states[-1])
            x = agent_states[-1].get_position()[0]
            y = agent_states[-1].get_position()[1]
            speed = agent_states[-1].speed
            if random:
                speed += np.random.uniform(-0.1, 0.1)
                speed = max(0, speed)
            theta = agent_states[-1].heading
            if agent_states[-2] == None:
                theta_new = theta
            else:
                theta_old = agent_states[-2].heading
                yaw_rate = (theta - theta_old) / 0.1
                if random:
                    yaw_rate += np.random.uniform(-0.05, 0.05)
                theta_new = theta + yaw_rate * 0.1

            theta_new = wrap_radians(theta)

            x_new = x + speed * np.cos(theta_new) * 0.1
            y_new = y + speed * np.sin(theta_new) * 0.1
            new_veh_state.location = Pt(x_new, y_new, agent_states[-1].get_position()[2])
            new_veh_state.heading = theta_new
            return new_veh_state

    """
    ############################### VEHICLE PLACEMENT ###############################
    """

    def place_a_vehicle(self, agent_attr: AgentAttr, agent_states: list[AgentState], t: int) -> None:
        """
        Places a vehicle in the simulation based on the agent's attributes and states.
        This method predicts the next state of the agent, determines the appropriate lane,
        and places the vehicle in the simulation. It handles three cases when finding a lane:
        
        Case 1: No edge found
            If no edge is found for the vehicle, the method returns without placing the vehicle.
        
        Case 2: Vehicle lane found
            If a vehicle lane is found, a route is created for the vehicle, and the vehicle is added
            to the simulation with the specified route.
        
        Case 3: Non-vehicle lane found
            If a non-vehicle lane is found, the vehicle is added to the simulation without a route.
        Args:
            agent_attr (AgentAttr): The attributes of the agent.
            agent_states (list[AgentState]): The list of states of the agent.
        Returns:
            None
        """


        # next state
        next_agent_state = self.predict_agent_simple_movement(agent_attr, agent_states, random=False)
        front_bumper_pos = center_to_front_bumper_position(
            next_agent_state.get_position(), agent_attr.size.length, next_agent_state.heading
        )
        heading_degrees_sumo = rad2degree(next_agent_state.heading)

        # try to find a lane to be situated at
        edge_id, lane_idx = self._get_vehicle_nearest_lane(front_bumper_pos, heading_degrees_sumo)

        # case 1: does not find an edge at all
        if not (not edge_id):
            print(edge_id, lane_idx)
        if not edge_id:
            return
        selected_lane: net.lane.Lane = self.sumonet.getLane(f"{edge_id}_{lane_idx}")
        # case 2: find a vehicle lane
        if selected_lane.allows("passenger"):
            route: list[str] = self._search_route(edge_id, lane_idx)
            route_id = f"route-{agent_attr.id}-{self.route_counter}"
            self.route_counter += 1
            keep_route = 0b011
            traci.route.add(route_id, route)

        # case 3: find a non-vehicle lane
        else:
            route_id = ""
            keep_route = 0b010

        type_id = self._set_vehicle_type(agent_attr, agent_states, edge_id, lane_idx)
        speed = max(0, next_agent_state.speed)
        try:
            traci.vehicle.remove(vehID=str(agent_attr.id))
        except:
            pass
        traci.vehicle.add(
            vehID=agent_attr.id,
            routeID=route_id,
            typeID=type_id,
            depart="now",
            departPos="0",
            departSpeed=str(speed),
        )
        traci.vehicle.moveToXY(
            agent_attr.id,
            edgeID=edge_id,
            laneIndex=lane_idx,
            x=front_bumper_pos[0],
            y=front_bumper_pos[1],
            angle=heading_degrees_sumo,
            keepRoute=keep_route,
        )
        self._set_veh_attributes(agent_attr.id)
        return

    def _set_vehicle_type(
        self, agent_attr: AgentAttr, agent_states: list[AgentState], edge_id: str, lane_idx: int
    ) -> str:

        type_id = f"agent_{agent_attr.id}_type"

        # 0. if the vehicle type has already exists
        type_list: list[int] = traci.vehicletype.getIDList()
        if type_id in type_list:
            return type_id

        traci.vehicletype.copy("DEFAULT_VEHTYPE", type_id)

        # 1. bbox attributes
        traci.vehicletype.setLength(type_id, agent_attr.size.length)
        traci.vehicletype.setWidth(type_id, agent_attr.size.width)
        traci.vehicletype.setHeight(type_id, agent_attr.size.height)

        # 2. general attributes
        speed_factor_mean = max(
            # the threshold
            (
                self.parameters["vehicle_min_speedfactor"]
                if agent_attr.type == AgentType.VEHICLE
                else self.parameters["pedestrian_min_speedfactor"]
            ),
            # speed factor by calculating current_speed / speed_limit
            agent_states[-1].speed / float(self.sumonet.getLane(f"{edge_id}_{lane_idx}").getSpeed()),
        )
        traci.vehicletype.setSpeedFactor(type_id, speed_factor_mean)
        traci.vehicletype.setSpeedDeviation(type_id, self.parameters["agent_speedfactor_std"])

        # 3. car following attributes: general
        traci.vehicletype.setAccel(type_id, accel(std=self.parameters["acceleration_std"]))
        traci.vehicletype.setDecel(type_id, decel(mean=self.parameters["deceleration_mean"], std=self.parameters["deceleration_std"]))
        traci.vehicletype.setEmergencyDecel(type_id, 9)
        traci.vehicletype.setMinGap(type_id, mingap())
        traci.vehicletype.setImperfection(type_id, sigma())
        traci.vehicletype.setTau(type_id, tau())
        traci.vehicletype.setParameter(type_id, "startupDelay", str(startupDelay()))

        # 4. lane changing attributes: general
        traci.vehicletype.setMaxSpeedLat(type_id, 1)
        traci.vehicletype.setMinGapLat(type_id, minGapLat())
        traci.vehicletype.setLateralAlignment(type_id, "center")

        # 5. junction model attributes
        traci.vehicletype.setParameter(type_id, "junctionModel.jmStoplineGap", str(jm_stoplinegap()))
        traci.vehicletype.setParameter(type_id, "junctionModel.jmSigmaMinor", str(sigma()))
        traci.vehicletype.setParameter(
            type_id, "junctionModel.jmIgnoreKeepClearTime", str(jm_ignorekeepcleartime())
        )

        return type_id

    def _set_veh_attributes(self, v_id: str) -> None:

        # lane changing attributes: more
        traci.vehicle.setSpeedMode(v_id, 0b011111)  # all checks on
        traci.vehicle.setLaneChangeMode(v_id, 0b011001000001)  # no cooperative lane change

        traci.vehicle.setParameter(v_id, "laneChangeModel.lcStrategic", str(lc_strategic()))
        traci.vehicle.setParameter(v_id, "laneChangeModel.lcCooperative", str(-1))  # disable
        traci.vehicle.setParameter(v_id, "laneChangeModel.lcSpeedGain", str(-1))  # disable
        traci.vehicle.setParameter(v_id, "laneChangeModel.lcKeepRight", str(-1))  # disable
        traci.vehicle.setParameter(v_id, "laneChangeModel.lcSublane", str(lc_sublane()))
        traci.vehicle.setParameter(v_id, "laneChangeModel.lcAccelLat", str(1.0))
        traci.vehicle.setParameter(v_id, "laneChangeModel.lcMaxSpeedLatStanding", str(0))
        traci.vehicle.setParameter(v_id, "laneChangeModel.lcMaxDistLatStanding", str(0))
        traci.vehicle.setParameter(v_id, "laneChangeModel.lcSigma", str(sigma()))

    def _get_vehicle_nearest_lane(self, position: tuple[float], angle: float) -> tuple[str, int]:
        """
        Given a point (global_x, global_y) in SUMO global coordinates that represents a vehicle's position,
        and veh_angle (in degrees) in SUMO global coordinates that represents the vehicle's heading,
        find the most suitable lane that it should be at.

        """

        # find all lanes around this position
        neighboring_lanes = self._get_neighboring_lanes(position, r=3)

        # pick the best lane in terms of distance and angle
        best_lane_idx = -1
        degree_difference = np.Inf
        for i, lane in enumerate(neighboring_lanes):
            lane_angle = traci.lane.getAngle(lane.getID())
            diff = lane_angle - angle
            while diff > 180:
                diff -= 2 * 180
            while diff < -180:
                diff += 2 * 180
            if abs(diff) < degree_difference - 10:
                degree_difference = abs(diff)
                best_lane_idx = i

        if best_lane_idx != -1:
            nearest_lane = neighboring_lanes[best_lane_idx]
            edge_id = nearest_lane.getEdge().getID()
            lane_idx = int(nearest_lane.getID().split("_")[-1])
            return edge_id, lane_idx
        else:
            edge_id = ""
            lane_idx = -1

        return edge_id, lane_idx

    def _search_route(self, curr_edge_id: str, curr_lane_idx: int) -> list[str]:

        if curr_edge_id == "":
            return []

        def _is_edge_allow_vehicles(edge_id: str):
            return "w" not in edge_id and "c" not in edge_id

        def _is_edge_end_edge(edge_id: str):
            return not traci.junction.getOutgoingEdges(traci.edge.getToJunction(edge_id))

        edges = [edge_id for edge_id in traci.edge.getIDList() if _is_edge_allow_vehicles(edge_id)]
        end_edges = [edge_id for edge_id in edges if _is_edge_end_edge(edge_id)]

        initial_lane_id = str(curr_edge_id) + "_" + str(curr_lane_idx)
        curr_lane_id = initial_lane_id
        complete_route: list[str] = [curr_edge_id]
        accumulated_length = 0
        ACCUMULATED_LENGTH_THRESHOLD = 100
        while complete_route[-1] not in end_edges:
            if accumulated_length < ACCUMULATED_LENGTH_THRESHOLD:
                current_lane: net.lane.Lane = self.sumonet.getLane(curr_lane_id)
                next_possible_lanes: list[net.lane.Lane] = current_lane.getOutgoingLanes()
                next_possible_edges: list[net.edge.Edge] = [lane.getEdge() for lane in next_possible_lanes]
            else:
                next_possible_edges = list(self.sumonet.getEdge(complete_route[-1]).getOutgoing().keys())

            next_possible_edges = [
                edge for edge in next_possible_edges if _is_edge_allow_vehicles(edge.getID())
            ]
            if not next_possible_edges:
                break
            next_edge_idx = self._select_next_edge(next_possible_edges)

            if accumulated_length < ACCUMULATED_LENGTH_THRESHOLD:
                next_lane: net.lane.Lane = next_possible_lanes[next_edge_idx]
                curr_lane_id = next_lane.getID()
                accumulated_length += next_lane.getLength()

            next_edge: net.edge.Edge = next_possible_edges[next_edge_idx]
            complete_route.append(next_edge.getID())

        # if the beginning of the route is an internal route,
        # this causes seg fault when calling traci.setRoute(),
        # so we try to replace it with its incomming edge
        if complete_route[0][0] == ":":
            prior_lanes = self.sumonet.getLane(initial_lane_id).getIncoming()
            prior_lane_ids = sorted(
                [lane.getEdge().getID() for lane in prior_lanes if ":" not in lane.getEdge().getID()]
            )
            if not len(prior_lane_ids):
                complete_route.pop(0)
            else:
                complete_route[0] = prior_lane_ids[0]

        return complete_route

    def _select_next_edge(self, next_possible_edges: list[net.edge.Edge]) -> int:
        scores = [edge.getSpeed() * edge.getLaneNumber() for edge in next_possible_edges]

        main_edges_idx = [i for i in range(len(scores)) if scores[i] >= 30]
        side_edges_idx = [i for i in range(len(scores)) if scores[i] < 30]
        if len(main_edges_idx) and len(side_edges_idx):
            if random.random() < self.parameters["sideroad_probability"]:
                select_edge_idx = random.choice(side_edges_idx)
            else:
                select_edge_idx = random.choice(main_edges_idx)
        elif len(main_edges_idx):
            select_edge_idx = random.choice(main_edges_idx)
        else:
            select_edge_idx = random.choice(side_edges_idx)
        return select_edge_idx

    """
    ############################### PEDESTRIAN PLACEMENT ###############################
    """

    def place_a_pedestrian(self, agent_attr: AgentAttr, agent_states: list[AgentState]) -> None:
        """
        Places a pedestrian in the simulation based on the given agent attributes and states.
        Args:
            agent_attr (AgentAttr): Attributes of the agent, including ID and size.
            agent_states (list[AgentState]): List of agent states, including position, heading, and speed.
        Returns:
            None
        Possible Results:
            - The pedestrian is placed on a lane that allows pedestrians.
            - If no suitable edge is found, the function returns without placing the pedestrian.
            - If a lane is found but does not allow pedestrians, the function returns without placing the pedestrian.
            - If the pedestrian is successfully placed, their walking stage is appended, and they are moved to the specified position.
            - If the pedestrian already exists in the simulation, they are removed before being re-added.
        """
        

        # next state
        next_agent_state = self.predict_agent_simple_movement(agent_attr, agent_states)
        front_bumper_pos = center_to_front_bumper_position(
            next_agent_state.get_position(), agent_attr.size.length, next_agent_state.heading
        )
        heading_degrees_sumo = rad2degree(next_agent_state.heading)
        edge_id, lane_idx = self._get_pedestrian_nearest_lane(front_bumper_pos)

        # case 1: does not find an edge at all
        if not edge_id:
            return
        selected_lane: net.lane.Lane = self.sumonet.getLane(f"{edge_id}_{lane_idx}")
        # case 2: find a lane, but does not allow pedestrian
        if not selected_lane.allows("pedestrian"):
            return
        # case 3: find a pedestrian lane

        type_id = self._set_pedestrian_type(agent_attr, agent_states, edge_id, lane_idx)
        try:
            traci.person.remove(agent_attr.id)
        except:
            pass
        traci.person.add(personID=agent_attr.id, edgeID=edge_id, pos=0, typeID=type_id)

        walking_edges = [edge_id]
        if agent_attr.id == "3937":
            # walking_edges = []
            pass
            # return
        traci.person.appendWalkingStage(
            personID=agent_attr.id,
            edges=walking_edges,
            speed=max(0, next_agent_state.speed),
            arrivalPos=-1,
        )
        traci.person.moveToXY(
            personID=agent_attr.id,
            edgeID=edge_id,
            x=front_bumper_pos[0],
            y=front_bumper_pos[1],
            angle=heading_degrees_sumo,
            keepRoute=0b110,
        )

        return

    def _get_pedestrian_nearest_lane(self, position: tuple[float]) -> tuple[str, int]:

        neighboring_lanes = self._get_neighboring_lanes(position, r=1)
        neighboring_lanes_for_pedestrian = [lane for lane in neighboring_lanes if lane.allows("pedestrian")]
        neighboring_lanes_for_vehicles = [lane for lane in neighboring_lanes if not lane.allows("pedestrian")]
        neighboring_lanes = (
            neighboring_lanes_for_pedestrian
            if len(neighboring_lanes_for_pedestrian)
            else neighboring_lanes_for_vehicles
        )

        if len(neighboring_lanes):
            edge_id = neighboring_lanes[0].getEdge().getID()
            lane_idx = int(neighboring_lanes[0].getID().split("_")[-1])
        else:
            edge_id = ""
            lane_idx = -1

        return edge_id, lane_idx

    def _set_pedestrian_type(
        self, agent_attr: AgentAttr, agent_states: list[AgentState], edge_id: str, lane_idx: int
    ):
        type_id = f"agent_{agent_attr.id}_type"
        type_list: list[int] = traci.vehicletype.getIDList()
        if type_id in type_list:
            return type_id

        traci.vehicletype.copy("DEFAULT_PEDTYPE", type_id)

        # bbox attributes
        traci.vehicletype.setLength(type_id, agent_attr.size.length)
        traci.vehicletype.setWidth(type_id, agent_attr.size.width)
        traci.vehicletype.setHeight(type_id, agent_attr.size.height)

        # general attributes
        # speed_factor_mean = agent_states[-1].speed / float(self.sumonet.getLane(f"{edge_id}_{lane_idx}").getSpeed())
        # traci.vehicletype.setSpeedFactor(type_id, speed_factor_mean)
        # traci.vehicletype.setSpeedDeviation(type_id, 0.1)

        return type_id

    """
    ############################### EXTRA CONTROL ###############################
    """

    def control_immobile_agent(self, agent_attr: AgentAttr, agent_states: list[AgentState]) -> None:

        reference_state = agent_states[self.history_length - 1]
        center_xy_waymo = reference_state.get_position()
        front_bumper_pos = center_to_front_bumper_position(
            center_xy_waymo, agent_attr.size.length, reference_state.heading
        )
        # this agent should be found in vehicle list
        if agent_attr.type == AgentType.VEHICLE or self.parameters["treat_ped_as_vehicle"]:
            assert agent_attr.id in traci.vehicle.getIDList()
            traci.vehicle.moveToXY(
                agent_attr.id,
                edgeID="",
                laneIndex=-1,
                x=float(front_bumper_pos[0]),
                y=float(front_bumper_pos[1]),
                angle=float(rad2degree(reference_state.heading)),
                keepRoute=0b010,
            )
        # this agent should be found in person list
        else:
            assert agent_attr.id in traci.person.getIDList()
            traci.person.moveToXY(
                personID=agent_attr.id,
                edgeID="",
                x=float(front_bumper_pos[0]),
                y=float(front_bumper_pos[1]),
                angle=float(rad2degree(reference_state.heading)),
                keepRoute=0b010,
            )
        self.control_state_dict[agent_attr.id] = AgentControlState.STATIONARY

    def control_veh_behavior_at_intersection(self, node: net.node.Node) -> None:

        def _force_break(v_id: str):
            slow_down_duration = 0 if traci.vehicle.getSpeed(v_id) < 2 else traci.vehicle.getSpeed(v_id) / 4
            traci.vehicle.slowDown(v_id, 0, slow_down_duration)
            self.control_state_dict[v_id] = AgentControlState.FORCE_BREAK

        tls_id = node.getID()
        signal_states = traci.trafficlight.getRedYellowGreenState(tls_id)  # ['r', 'g', ...] # (#links,)
        all_controlled_links = traci.trafficlight.getControlledLinks(tls_id)  # (#links, #lanes per link)

        CLOSE_DISTANCE = 6
        LOW_SPEED = 1

        # for every link index
        for tl_link_idx, links in enumerate(all_controlled_links):
            tl_state_at_this_link = signal_states[tl_link_idx]

            # for every lane controlled by this linke index
            for from_lane_id, to_lane_id, via_lane_id in links:
                if not via_lane_id:
                    continue
                in_junction_vehicle_ids = traci.lane.getLastStepVehicleIDs(via_lane_id)
                incoming_vehicle_ids = traci.lane.getLastStepVehicleIDs(from_lane_id)

                # if this link is currently red, force vehicles near the stopping point to stop
                if tl_state_at_this_link == "r":

                    # if this lane includes a right turning lane, ignore it
                    outgoing_conns: list = self.sumonet.getLane(from_lane_id).getOutgoing()
                    if any(conn.getDirection() == "r" for conn in outgoing_conns):
                        continue

                    # control cars that have passed the stopping line
                    for v_id in in_junction_vehicle_ids:
                        curr_lane_pos: float = traci.vehicle.getLanePosition(v_id)
                        if curr_lane_pos < CLOSE_DISTANCE:
                            _force_break(v_id)

                    # control cars that are approaching the intersection
                    for v_id in incoming_vehicle_ids:

                        curr_lane_pos: float = traci.vehicle.getLanePosition(v_id)
                        lane_length = traci.lane.getLength(from_lane_id)
                        curr_speed = traci.vehicle.getSpeed(v_id)
                        acceleration = traci.vehicle.getAccel(v_id)

                        A_0 = 2.5
                        speed_threshold = np.sqrt(2 * A_0 * (lane_length - curr_lane_pos))
                        distance_threshold = (
                            (self.sumonet.getLane(from_lane_id).getSpeed()) ** 2 / (2 * A_0) / 2
                        )
                        if (curr_speed < LOW_SPEED) and lane_length - curr_lane_pos < distance_threshold:
                            _force_break(v_id)

                # or it is currently green, force vehicles near the stopping point to move
                elif tl_state_at_this_link in ["G", "g"]:
                    for v_id in in_junction_vehicle_ids:
                        curr_lane_pos: float = traci.vehicle.getLanePosition(v_id)
                        curr_speed: float = traci.vehicle.getSpeed(v_id)
                        if curr_lane_pos < CLOSE_DISTANCE and curr_speed < LOW_SPEED:
                            speed_up_duration = (traci.vehicle.getAllowedSpeed(v_id) - curr_speed) / 3
                            traci.vehicle.slowDown(
                                v_id,
                                traci.vehicle.getAllowedSpeed(v_id),
                                speed_up_duration,
                            )
                            self.control_state_dict[v_id] = AgentControlState.FORCE_GO

    """
    ############################### INFORMATION RETRIEVAL ###############################
    """

    def get_onroad_agents(
        self, agent_states: dict[str, list[AgentState]], remaining_agent_ids: list[int]
    ) -> dict[str, AgentState]:

        agents: dict[str, AgentState] = {}
        exist_agent_list = traci.vehicle.getIDList()

        for v_id in exist_agent_list:
            heading_in_sumo = degree2rad(traci.vehicle.getAngle(v_id))
            heading_in_waymo = wrap_radians(heading_in_sumo)
            front_bumper_pos: list[float] = list(traci.vehicle.getPosition(v_id))
            if len(front_bumper_pos) == 2:
                front_bumper_pos = front_bumper_pos + [agent_states[v_id][-1].get_position()[2]]
            center_position = front_bumper_to_center_position(
                front_bumper_pos,
                traci.vehicle.getLength(v_id),
                heading_in_waymo,
            )
            if v_id not in remaining_agent_ids:
                speed = traci.vehicle.getSpeed(v_id)
            else:
                speed = agent_states[v_id][-1].speed
                # for newly-generated vehicles, if it is offroad, SUMO return a speed of 0; we don't want this
            acceleration = traci.vehicle.getAcceleration(v_id)
            control_state = self.control_state_dict[v_id]

            agents[v_id] = AgentState(
                location=center_position,
                heading=heading_in_waymo,
                speed=speed,
                acceleration=acceleration,
                control_state=control_state,
            )

        exist_ped_list = traci.person.getIDList()

        for p_id in exist_ped_list:
            heading_in_sumo = degree2rad(traci.person.getAngle(p_id))
            heading_in_waymo = wrap_radians(heading_in_sumo)
            front_bumper_pos: list[float] = list(traci.person.getPosition(p_id))
            if len(front_bumper_pos) == 2:
                front_bumper_pos = front_bumper_pos + [agent_states[p_id][-1].get_position()[2]]
            center_position = front_bumper_to_center_position(
                front_bumper_pos,
                traci.person.getLength(p_id),
                heading_in_waymo,
            )
            if p_id not in remaining_agent_ids:
                speed = traci.person.getSpeed(p_id)
            else:
                speed = agent_states[p_id][-1].speed
            acceleration = traci.person.getAccel(p_id)
            control_state = self.control_state_dict[p_id]

            agents[p_id] = AgentState(
                location=center_position,
                heading=heading_in_waymo,
                speed=speed,
                acceleration=acceleration,
                control_state=control_state,
            )

        return agents

    def get_tls_list_from_SUMO(self) -> list[TrafficLightHead]:
        """
        Get a complete list of traffic light heads from SUMO at the last time step

        Returns a list of TrafficLightHead objects.
        """

        tl_head_list: list[TrafficLightHead] = []

        tls_id_list: list[str] = traci.trafficlight.getIDList()
        for tl_group_id in tls_id_list:
            controlled_links = traci.trafficlight.getControlledLinks(tl_group_id)
            for link_idx in range(len(controlled_links)):
                links = controlled_links[link_idx]
                for link_info in links:
                    front_lane, to_lane, via_lane = link_info
                    if not via_lane:
                        continue
                    via_lane_shape = traci.lane.getShape(via_lane)

                    # position
                    tl_head_pos = via_lane_shape[0]
                    # direction
                    tl_head_dir_in_sumo = np.array(via_lane_shape[1]) - np.array(via_lane_shape[0])
                    unit_length_tl_head_dir = list(tl_head_dir_in_sumo / np.linalg.norm(tl_head_dir_in_sumo))

                    tl_head_list.append(
                        TrafficLightHead(
                            id=str(len(tl_head_list)),
                            location=tl_head_pos,
                            tl_dir=unit_length_tl_head_dir,
                            state=TL_STATE_MAPPING[
                                traci.trafficlight.getRedYellowGreenState(tl_group_id)[link_idx]
                            ],
                        )
                    )
        return tl_head_list

    def _get_neighboring_lanes(self, position: tuple[float], r: float) -> list[net.lane.Lane]:

        neighboring_lanes: list[tuple[net.lane.Lane, float]] = self.sumonet.getNeighboringLanes(
            position[0], position[1], r=r
        )
        neighboring_lanes: list[net.lane.Lane] = [
            neighboring_lane[0] for neighboring_lane in neighboring_lanes
        ]
        return neighboring_lanes
