import numpy as np
import copy
from typing import Optional, Dict, List, Tuple
import tensorflow as tf

from waymo_open_dataset.wdl_limited.sim_agents_metrics import map_metric_features
from waymo_open_dataset.utils import box_utils, geometry_utils

from .agents import AgentAttr, AgentState, AgentType
from src.static_context.utils import Pt


class RoadEdgeMapper:

    EXTREMELY_LARGE_DISTANCE = 1e10

    def __init__(
        self,
        scenario,
        distance_algorithm: str = "3d",
        mapping_const: float = 1,
        ignore_ped: bool = False,
    ):
        self.scenario = scenario
        self.distance_algorithm = distance_algorithm
        self.mapping_const = mapping_const
        self.ignore_ped = ignore_ped
        self._prepare_map_features()

    def _prepare_map_features(self) -> None:
        """
        从场景中提取所有 road_edge 信息，并转换为张量格式保存
        """
        road_edges = []
        for map_feature in self.scenario.map_features:
            if map_feature.HasField("road_edge"):
                road_edges.append(map_feature.road_edge.polyline)
        self.polylines_tensor = map_metric_features._tensorize_polylines(road_edges)
        self.is_polyline_cyclic = map_metric_features._check_polyline_cycles(road_edges)

    def perform_mapping(
        self,
        agent_attrs: Dict[str, AgentAttr],
        agent_states: Dict[str, List[AgentState]],
        stationary_agent_ids: List[str],
    ) -> Dict[str, List[AgentState]]:
        """
        对传入的 agent_states 进行离路映射处理

        :param agent_attrs: 各代理的属性字典
        :param agent_states: 各代理的状态列表字典（状态随时间变化）
        :param stationary_agent_ids: 静止代理的 ID 列表
        :return: 处理后的 agent_states 字典
        """
        new_agent_states = copy.deepcopy(agent_states)

        for v_id, attr in agent_attrs.items():
            # 跳过静止代理以及（根据设置）行人
            if v_id in stationary_agent_ids:
                continue
            if attr.type == AgentType.PEDESTRIAN and self.ignore_ped:
                continue
            if agent_states[v_id][0] is None:
                continue  ###check why is this happening

            print(f"Inspecting object id {v_id}")

            # 处理历史状态（前 11 帧）：如果历史状态中任一时刻离路，则不对该对象进行预测映射
            history_states = [state for state in agent_states[v_id][:11] if state is not None]
            if history_states:
                history_box = tf.convert_to_tensor(
                    [
                        [*(state.location.to_list()), *(attr.size.to_list()), state.heading]
                        for state in history_states
                    ],
                    dtype=tf.float32,
                )
                history_box_corners = box_utils.get_upright_3d_box_corners(history_box)[:, :4]
                flat_history_box_corners = tf.reshape(history_box_corners, (-1, 3))
                corner_distance, _ = self.compute_signed_distance_to_polylines(
                    xyzs=flat_history_box_corners,
                    polylines=self.polylines_tensor,
                    is_polyline_cyclic=self.is_polyline_cyclic,
                    z_stretch=3.0,
                )
                # 重塑为 (帧数, 4)
                corner_distance = tf.reshape(corner_distance, (-1, 4))
                max_distance_idx = tf.argmax(corner_distance, axis=1)
                max_distance = tf.gather(corner_distance, max_distance_idx, batch_dims=1)
                if tf.reduce_any(tf.greater(max_distance, 0)):
                    continue  # 历史状态中存在离路情况，跳过此对象

            # 处理预测状态（第 11 帧之后，共 80 帧）
            prediction_states = [state for state in agent_states[v_id][11:] if state is not None]
            if prediction_states:
                prediction_box = tf.convert_to_tensor(
                    [
                        [*(state.location.to_list()), *(attr.size.to_list()), state.heading]
                        for state in prediction_states
                    ],
                    dtype=tf.float32,
                )
                prediction_box_corners = box_utils.get_upright_3d_box_corners(prediction_box)[:, :4]
                flat_prediction_box_corners = tf.reshape(prediction_box_corners, (-1, 3))
                corner_distance, corresponding_vector = self.compute_signed_distance_to_polylines(
                    xyzs=flat_prediction_box_corners,
                    polylines=self.polylines_tensor,
                    is_polyline_cyclic=self.is_polyline_cyclic,
                    z_stretch=3.0,
                    algorithm=self.distance_algorithm,
                )
                corner_distance = tf.reshape(corner_distance, (-1, 4))
                corresponding_vector = tf.reshape(corresponding_vector, (-1, 4, 3))
                max_distance_idx = tf.argmax(corner_distance, axis=1)
                max_distance = tf.gather(corner_distance, max_distance_idx, batch_dims=1)
                corresponding_vector = tf.gather(corresponding_vector, max_distance_idx, batch_dims=1)

                # 针对 80 帧预测状态逐帧判断并进行映射
                for t in range(91):
                    if max_distance[t] > 0:
                        new_location = self._compute_mapped_location(
                            old_location=agent_states[v_id][t].location,
                            corresponding_vector=corresponding_vector[t],
                            max_distance=max_distance[t],
                        )
                        new_agent_states[v_id][t].location = new_location

        return new_agent_states

    def _compute_mapped_location(
        self,
        old_location: Pt,
        corresponding_vector: tf.Tensor,
        max_distance: tf.Tensor,
    ) -> Pt:
        """
        根据对应向量与最大距离计算新的位置

        :param old_location: 原始位置（Pt 对象）
        :param corresponding_vector: 对应的方向向量（Tensor）
        :param max_distance: 最大距离值（Tensor）
        :return: 计算后的新位置（Pt 对象），保持 z 轴不变
        """
        k = (max_distance + self.mapping_const) / tf.norm(corresponding_vector[:2])
        print(f"Mapping debug: vector={corresponding_vector}, distance_2d={max_distance}")
        old_xyz_list = old_location.to_list()
        new_xyz = tf.convert_to_tensor(old_xyz_list, dtype=tf.float32) - tf.cast(corresponding_vector * k, dtype=tf.float32)
        new_xyz_list = new_xyz.numpy().tolist()
        # 保持 z 轴不变
        new_xyz_list[2] = old_xyz_list[2]
        return Pt(*new_xyz_list)

    def compute_signed_distance_to_polylines(
        self,
        xyzs: tf.Tensor,
        polylines: tf.Tensor,
        is_polyline_cyclic: Optional[tf.Tensor] = None,
        z_stretch: float = 1.0,
        algorithm: str = "3d",
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        计算点集到多条 polyline 的带符号二维距离及对应向量

        :param xyzs: (num_points, 3) 点集坐标
        :param polylines: (num_polylines, num_segments+1, 4) 表示 polyline 的张量
        :param is_polyline_cyclic: 每条 polyline 是否闭合
        :param z_stretch: z 轴的伸缩因子
        :param algorithm: 距离计算算法，“3d” 或其他
        :return: (signed_distance, corresponding_vector)
        """
        num_points = xyzs.shape[0]
        tf.ensure_shape(xyzs, [num_points, 3])
        num_polylines = polylines.shape[0]
        num_segments = polylines.shape[1] - 1
        tf.ensure_shape(polylines, [num_polylines, num_segments + 1, 4])

        # shape: (num_polylines, num_segments+1)
        is_point_valid = tf.cast(polylines[:, :, 3], dtype=tf.bool)
        # shape: (num_polylines, num_segments)
        is_segment_valid = tf.logical_and(is_point_valid[:, :-1], is_point_valid[:, 1:])

        if is_polyline_cyclic is None:
            is_polyline_cyclic = tf.zeros([num_polylines], dtype=tf.bool)
        else:
            tf.ensure_shape(is_polyline_cyclic, [num_polylines])

        # Get distance to each segment.
        # shape: (num_points, num_polylines, num_segments, 3)
        xyz_starts = polylines[tf.newaxis, :, :-1, :3]
        xyz_ends = polylines[tf.newaxis, :, 1:, :3]
        start_to_point = xyzs[:, tf.newaxis, tf.newaxis, :3] - xyz_starts
        start_to_end = xyz_ends - xyz_starts

        # Relative coordinate of point projection on segment.
        # shape: (num_points, num_polylines, num_segments)
        rel_t = tf.math.divide_no_nan(
            geometry_utils.dot_product_2d(start_to_point[..., :2], start_to_end[..., :2]),
            geometry_utils.dot_product_2d(start_to_end[..., :2], start_to_end[..., :2]),
        )

        # Negative if point is on port side of segment, positive if point on
        # starboard side of segment.
        # shape: (num_points, num_polylines, num_segments)
        n = tf.sign(geometry_utils.cross_product_2d(start_to_point[..., :2], start_to_end[..., :2]))

        # Compute the absolute 3d distance to segment.
        # The vertical component is scaled by `z-stretch` to increase the separation
        # between different road altitudes.
        # shape: (num_points, num_polylines, num_segments, 3)
        segment_to_point = start_to_point - (start_to_end * tf.clip_by_value(rel_t, 0.0, 1.0)[..., tf.newaxis])
        # shape: (3)
        stretch_vector = tf.constant([1.0, 1.0, z_stretch], dtype=tf.float32)
        # shape: (num_points, num_polylines, num_segments)
        distance_to_segment_3d = tf.linalg.norm(
            segment_to_point * stretch_vector[tf.newaxis, tf.newaxis, tf.newaxis],
            axis=-1,
        )
        # Absolute planar distance to segment.
        # shape: (num_points, num_polylines, num_segments)
        distance_to_segment_2d = tf.linalg.norm(
            segment_to_point[..., :2],
            axis=-1,
        )

        # There are 3 cases:
        #   - if the point projection on the line falls within the segment, the sign
        #       of the distance is `n`.
        #   - if the point projection on the segment falls before the segment start,
        #       the sign of the distance depends on the convexity of the prior and
        #       nearest segments.
        #   - if the point projection on the segment falls after the segment end, the
        #       sign of the distance depends on the convexity of the nearest and next
        #       segments.

        # shape: (num_points, num_polylines, num_segments+2, 2)
        start_to_end_padded = tf.concat(
            [
                start_to_end[:, :, -1:, :2],
                start_to_end[..., :2],
                start_to_end[:, :, :1, :2],
            ],
            axis=-2,
        )
        # shape: (num_points, num_polylines, num_segments+1)
        is_locally_convex = tf.greater(
            geometry_utils.cross_product_2d(start_to_end_padded[:, :, :-1], start_to_end_padded[:, :, 1:]),
            0.0,
        )

        # Get shifted versions of `n` and `is_segment_valid`. If the polyline is
        # cyclic, the tensors are rolled, else they are padded with their edge value.
        # shape: (num_points, num_polylines, num_segments)
        n_prior = tf.concat(
            [
                tf.where(
                    is_polyline_cyclic[tf.newaxis, :, tf.newaxis],
                    n[:, :, -1:],
                    n[:, :, :1],
                ),
                n[:, :, :-1],
            ],
            axis=-1,
        )
        n_next = tf.concat(
            [
                n[:, :, 1:],
                tf.where(
                    is_polyline_cyclic[tf.newaxis, :, tf.newaxis],
                    n[:, :, :1],
                    n[:, :, -1:],
                ),
            ],
            axis=-1,
        )
        # shape: (num_polylines, num_segments)
        is_prior_segment_valid = tf.concat(
            [
                tf.where(
                    is_polyline_cyclic[:, tf.newaxis],
                    is_segment_valid[:, -1:],
                    is_segment_valid[:, :1],
                ),
                is_segment_valid[:, :-1],
            ],
            axis=-1,
        )
        is_next_segment_valid = tf.concat(
            [
                is_segment_valid[:, 1:],
                tf.where(
                    is_polyline_cyclic[:, tf.newaxis],
                    is_segment_valid[:, :1],
                    is_segment_valid[:, -1:],
                ),
            ],
            axis=-1,
        )

        # shape: (num_points, num_polylines, num_segments)
        sign_if_before = tf.where(
            is_locally_convex[:, :, :-1],
            tf.maximum(n, n_prior),
            tf.minimum(n, n_prior),
        )
        sign_if_after = tf.where(is_locally_convex[:, :, 1:], tf.maximum(n, n_next), tf.minimum(n, n_next))

        # shape: (num_points, num_polylines, num_segments)
        sign_to_segment = tf.where(
            (rel_t < 0.0) & is_prior_segment_valid,
            sign_if_before,
            tf.where((rel_t > 1.0) & is_next_segment_valid, sign_if_after, n),
        )

        # Flatten polylines together.
        # shape: (num_points, all_segments)
        distance_to_segment_3d = tf.reshape(distance_to_segment_3d, (num_points, num_polylines * num_segments))
        distance_to_segment_2d = tf.reshape(distance_to_segment_2d, (num_points, num_polylines * num_segments))
        sign_to_segment = tf.reshape(sign_to_segment, (num_points, num_polylines * num_segments))
        segment_to_point = tf.reshape(segment_to_point, (num_points, num_polylines * num_segments, 3))

        # Mask out invalid segments.
        # shape: (all_segments)
        is_segment_valid = tf.reshape(is_segment_valid, (num_polylines * num_segments))
        # shape: (num_points, all_segments)
        distance_to_segment_3d = tf.where(
            is_segment_valid[tf.newaxis],
            distance_to_segment_3d,
            RoadEdgeMapper.EXTREMELY_LARGE_DISTANCE,
        )
        distance_to_segment_2d = tf.where(
            is_segment_valid[tf.newaxis],
            distance_to_segment_2d,
            RoadEdgeMapper.EXTREMELY_LARGE_DISTANCE,
        )

        distance_to_segment_2d = tf.where(
            tf.less(tf.abs(segment_to_point[:,:,2]), 3),
            distance_to_segment_2d,
            RoadEdgeMapper.EXTREMELY_LARGE_DISTANCE
        )


        if algorithm == "3d":
            closest_segment_index = tf.argmin(distance_to_segment_3d, axis=-1)
        else:
            closest_segment_index = tf.argmin(distance_to_segment_2d, axis=-1)
        distance_sign = tf.gather(sign_to_segment, closest_segment_index, batch_dims=1)
        distance_2d = tf.gather(distance_to_segment_2d, closest_segment_index, batch_dims=1)
        corresponding_segments = tf.gather(segment_to_point, closest_segment_index, batch_dims=1)

        return distance_2d * distance_sign, corresponding_segments
