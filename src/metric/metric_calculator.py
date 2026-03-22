import sys
sys.path.append(".")
import numpy as np
import tensorflow as tf
from waymo_open_dataset.wdl_limited.sim_agents_metrics  import map_metric_features, interaction_features

from src.dynamic_behavior.agents import AgentAttr, AgentState, AgentType
from src.utils.pack_h5_womd import pack_scenario_raw

class MetricCalculator:
    def __init__(self, scenario, agent_attrs: dict, agent_states: dict,
                 current_time_index: int = 10,):

        self.scenario = scenario
        self.original_agent_attrs = agent_attrs
        self.original_agent_states = agent_states
        self.current_time_index = current_time_index
        
        self.episode = pack_scenario_raw(scenario)
        self.xmin, self.xmax, self.ymin, self.ymax = self.episode["map/boundary"].tolist()
        
        self.agent_attrs = {agent_id: attr for agent_id, attr in agent_attrs.items() 
                            if attr.type.value == 0}
        self.agent_states = {agent_id: agent_states[agent_id] 
                             for agent_id in self.agent_attrs.keys()}
        
        self._prepare_data()

    def _prepare_data(self):

        valid_agent_ids = list(self.agent_attrs.keys())
        self.N = len(valid_agent_ids)
        if self.N == 0:
            return
        self.T = len(next(iter(self.agent_states.values())))
        
        center_x = np.zeros((self.N, self.T))
        center_y = np.zeros((self.N, self.T))
        center_z = np.zeros((self.N, self.T))
        length = np.zeros((self.N, self.T))
        width = np.zeros((self.N, self.T))
        height = np.zeros((self.N, self.T))
        heading = np.zeros((self.N, self.T))
        
        for i, agent_id in enumerate(valid_agent_ids):
            attr = self.agent_attrs[agent_id]
            states = self.agent_states[agent_id]
            for t, state in enumerate(states):
                if state is None:
                    continue
                loc = state.location.to_list()
                center_x[i, t] = loc[0]
                center_y[i, t] = loc[1]
                center_z[i, t] = loc[2]
                length[i, t] = attr.size.length
                width[i, t] = attr.size.width
                height[i, t] = attr.size.height
                heading[i, t] = state.heading
                
        for i in range(self.N):
            for t in range(self.current_time_index, -1, -1):
                if self.agent_states[valid_agent_ids[i]][t] is None:
                    center_x[i, t] = center_x[i, t+1]
                    center_y[i, t] = center_y[i, t+1]
                    center_z[i, t] = center_z[i, t+1]
                    length[i, t] = length[i, t+1]
                    width[i, t] = width[i, t+1]
                    height[i, t] = height[i, t+1]
                    heading[i, t] = heading[i, t+1]
        
        valid = np.ones((self.N, self.T), dtype=bool)
        valid &= (center_x >= self.xmin) & (center_x <= self.xmax)
        valid &= (center_y >= self.ymin) & (center_y <= self.ymax)
        
        evaluated_object_mask = np.ones((self.N), dtype=bool)
        
        road_edges = []
        for map_feature in self.scenario.map_features:
            if map_feature.HasField('road_edge'):
                road_edges.append(map_feature.road_edge.polyline)
        
        self.center_x = tf.convert_to_tensor(center_x, dtype=tf.float32)
        self.center_y = tf.convert_to_tensor(center_y, dtype=tf.float32)
        self.center_z = tf.convert_to_tensor(center_z, dtype=tf.float32)
        self.length = tf.convert_to_tensor(length, dtype=tf.float32)
        self.width = tf.convert_to_tensor(width, dtype=tf.float32)
        self.height = tf.convert_to_tensor(height, dtype=tf.float32)
        self.heading = tf.convert_to_tensor(heading, dtype=tf.float32)
        self.valid = tf.convert_to_tensor(valid, dtype=tf.bool)
        self.evaluated_object_mask = tf.convert_to_tensor(evaluated_object_mask, dtype=tf.bool)
        self.road_edges = road_edges

    def compute_collision_metric(self):

        distances_to_objects = interaction_features.compute_distance_to_nearest_object(
            center_x=self.center_x,
            center_y=self.center_y,
            center_z=self.center_z,
            length=self.length,
            width=self.width,
            height=self.height,
            heading=self.heading,
            valid=self.valid,
            evaluated_object_mask=self.evaluated_object_mask
        )

        collision_in_previous = tf.reduce_any(distances_to_objects[:, :self.current_time_index] < 0, axis=1)
        eligible = tf.logical_not(collision_in_previous)
        
        eligible_distances = tf.boolean_mask(distances_to_objects[:, self.current_time_index:], eligible, axis=0)
        total_colliding_time_steps = tf.reduce_sum(tf.cast(eligible_distances < 0, tf.int32)).numpy()
        total_time_steps = int(tf.reduce_sum(tf.cast(eligible, tf.int32)).numpy()) * (self.T - self.current_time_index)

        collision_vehicle_count = tf.reduce_sum(
            tf.cast(tf.reduce_any(eligible_distances < 0, axis=1), tf.int32)
        ).numpy()
        
        return total_colliding_time_steps, total_time_steps, collision_vehicle_count

    def compute_offroad_metric(self):

        distances_to_road_edge_list = []
        for t in range(self.T):
            distances_t = map_metric_features.compute_distance_to_road_edge(
                center_x=self.center_x[:, t:t+1],
                center_y=self.center_y[:, t:t+1],
                center_z=self.center_z[:, t:t+1],
                length=self.length[:, t:t+1],
                width=self.width[:, t:t+1],
                height=self.height[:, t:t+1],
                heading=self.heading[:, t:t+1],
                valid=self.valid[:, t:t+1],
                evaluated_object_mask=self.evaluated_object_mask,
                road_edge_polylines=self.road_edges
            )  
            distances_to_road_edge_list.append(distances_t)
            
      
        distances_to_road_edge = tf.concat(distances_to_road_edge_list, axis=1)
        
     
        offroad_in_first = tf.reduce_any(distances_to_road_edge[:, :self.current_time_index] > 0, axis=1)
        onroad_mask = tf.logical_not(offroad_in_first)
        onroad_indices = tf.where(onroad_mask)[:, 0]
        
        eligible_offroad_distances = tf.gather(distances_to_road_edge, onroad_indices)
        total_offroad_time_steps = tf.reduce_sum(
            tf.cast(eligible_offroad_distances[:, self.current_time_index:] > 0, tf.int32)
        ).numpy()
        total_time_steps = int(tf.reduce_sum(tf.cast(onroad_mask, tf.int32)).numpy()) * (self.T - self.current_time_index)
       
        offroad_vehicle_count = tf.reduce_sum(
            tf.cast(tf.reduce_any(eligible_offroad_distances[:, self.current_time_index:] > 0, axis=1), tf.int32)
        ).numpy()
        
        return total_offroad_time_steps, total_time_steps, offroad_vehicle_count

    def get_metrics(self):

        colliding_steps, collision_total_steps, collision_vehicle_count = self.compute_collision_metric()
        offroad_steps, offroad_total_steps, offroad_vehicle_count = self.compute_offroad_metric()
        total_steps = self.N * self.T
        return {
            "total_colliding_time_steps": int(colliding_steps),
            "total_time_steps_collisioncount": int(collision_total_steps),
            "collision_vehicle_count": int(collision_vehicle_count),
            "total_offroad_time_steps": int(offroad_steps),
            "total_time_steps_offroadcount": int(offroad_total_steps),
            "offroad_vehicle_count": int(offroad_vehicle_count),
            "total_time_steps": int(total_steps),
        }


if __name__ == "__main__":
    
    
    scenario = None # replace this with the original scenario object obtained from WOMD dataset
    # replace these above with the simulation output from the SUMOBaseline simulator
    agent_attrs = None
    agent_states = None
    
    metric_calc = MetricCalculator(
        scenario,
        agent_attrs,
        agent_states,
        current_time_index=10,
    )
    if metric_calc.N == 0:
        exit(0)
    
    output_data = metric_calc.get_metrics()
    print(output_data)