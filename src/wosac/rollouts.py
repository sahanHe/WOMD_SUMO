import numpy as np
import os

from waymo_open_dataset.protos import scenario_pb2, sim_agents_submission_pb2, sim_agents_metrics_pb2
from waymo_open_dataset.utils.sim_agents import submission_specs

from src.dynamic_behavior import SUMOBaseline


def validate_simagents(scenario, simulator: SUMOBaseline) -> bool:
    sim_agents: set[int] = set(map(int, simulator.agent_attrs.keys()))
    correct_sim_agents: set[int] = set(submission_specs.get_sim_agent_ids(scenario))

    return sim_agents == correct_sim_agents


def gen_one_joint_scene(simulator: SUMOBaseline, sim_cfg: dict = {}) -> sim_agents_submission_pb2.JointScene:
    agent_attrs, agent_states, _, _ = simulator.run_simulation(**sim_cfg)

    sim_trajectories: list[sim_agents_submission_pb2.SimulatedTrajectory] = []
    for agent_id in agent_attrs.keys():
        positions = [agent_state.get_position() for agent_state in agent_states[agent_id][11:]]
        center_x, center_y, center_z = zip(*positions)
        heading = (agent_state.heading for agent_state in agent_states[agent_id][11:])

        if np.any([center_x, center_y, center_z, heading] == None):
            raise ValueError("Found None in simulated trajectories")

        sim_trajectories.append(
            sim_agents_submission_pb2.SimulatedTrajectory(
                center_x=center_x,
                center_y=center_y,
                center_z=center_z,
                heading=heading,
                object_id=int(agent_id),
            )
        )

    return sim_agents_submission_pb2.JointScene(simulated_trajectories=sim_trajectories)


def gen_scenario_rollouts(
    scenario, baseline_cfg: dict = {}, sim_cfg: dict = {}, num_scenes=submission_specs.N_ROLLOUTS
) -> sim_agents_submission_pb2.ScenarioRollouts:
    simulator = SUMOBaseline(scenario, **baseline_cfg)

    if not validate_simagents(scenario, simulator):
        raise ValueError("Incorrect sim agents")

    joint_scenes: list[sim_agents_submission_pb2.JointScene] = []
    for _ in range(num_scenes):
        joint_scene = gen_one_joint_scene(simulator, sim_cfg)
        joint_scenes.append(joint_scene)


    rollouts = sim_agents_submission_pb2.ScenarioRollouts(
        joint_scenes=joint_scenes, scenario_id=scenario.scenario_id
    )
    if num_scenes == submission_specs.N_ROLLOUTS:
        submission_specs.validate_scenario_rollouts(rollouts, scenario)
    return rollouts

def metrics_to_dict(metrics: sim_agents_metrics_pb2.SimAgentMetrics) -> dict:
    return {
        "scenario_id": metrics.scenario_id,
        "metametric": metrics.metametric,
        "ADE": metrics.average_displacement_error,
        "linear_v": metrics.linear_speed_likelihood,
        "linear_a": metrics.linear_acceleration_likelihood,
        "angular_v": metrics.angular_speed_likelihood,
        "angular_a": metrics.angular_acceleration_likelihood,
        "distance_to_nearest_object": metrics.distance_to_nearest_object_likelihood,
        "collision_indication": metrics.collision_indication_likelihood,
        "time_to_collision": metrics.time_to_collision_likelihood,
        "distance_to_road_edge": metrics.distance_to_road_edge_likelihood,
        "offroad_indication": metrics.offroad_indication_likelihood,
        "min_ADE": metrics.min_average_displacement_error,
        "collision_rate": metrics.simulated_collision_rate,
        "offroad_rate": metrics.simulated_offroad_rate,
    }
