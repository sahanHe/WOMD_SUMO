import numpy as np
import cv2
from .scenario_visualizer import *
from src.dynamic_behavior.agents import TrafficLightHead, Agent, AgentRole

COLOR_YELLOWGREEN = (173, 255, 47)

class SUMOBaselineVisualizer(ScenarioVisualizer):

    def __init__(self, scenario,
    px_per_m: float = 10,
    video_size: int = 960,
    types_to_draw: Union[str, list[int]] = "lanecenters",
    shaded: bool = False,
    simplify_tls_color: bool = True,
    simplify_agent_color: bool = True,
    tls_style: str = "tlstop"):
        super().__init__(scenario, px_per_m, video_size, types_to_draw, shaded, simplify_tls_color, simplify_agent_color, tls_style)

        self.agent_control_state_style = [
            COLOR_PLUM_2,
            COLOR_CHOCOLATE_2,
            COLOR_SCARLET_RED_0,
            COLOR_YELLOWGREEN,
            COLOR_ORANGE_1
        ]

    def save_video(
        self,
        sim_time_buff_agents: list[list[Agent]],
        sim_time_buff_tls: list[list[TrafficLightHead]],
        base_dir: Union[str, None] = None,
        video_name: Union[str, None] = None,
        start_step: int = 0,
        end_step: Union[int, None] = None,
        with_ground_truth: bool = True
    ) -> None:

        img_buffer: list[np.ndarray] = []
        if end_step == None:
            end_step = self.episode["agent/valid"].shape[1]

        # every time step
        for t in range(start_step, end_step):
            step_image = self._plot_step_with_pred(t, sim_time_buff_agents[t], sim_time_buff_tls[t], with_ground_truth=with_ground_truth)
            img_buffer.append(step_image)

        # save it
        if not base_dir:
            base_dir = Path(f"outputs/{self.scenario_id}")
        base_dir = Path(base_dir)
        os.makedirs(base_dir, exist_ok=True)
        if not video_name:
            video_name = f"{self.scenario_id}.mp4"
        file_path = Path(base_dir) / video_name
        video_writer = cv2.VideoWriter(
            str(file_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10,
            (self.raster_map.shape[1], self.raster_map.shape[0]),
        )  # 10 is the fps
        for im in img_buffer:
            video_writer.write(im[:, :, ::-1])
        video_writer.release()

        return [img[:, :, ::-1] for img in img_buffer]

    def _plot_step_with_pred(
        self,
        t: int,
        agent_list: list[Agent],
        tl_list: list[TrafficLightHead],
        with_ground_truth: bool = True
    ):
        if with_ground_truth:
            step_image = self._plot_step(t, with_agents=True, with_tls=False, only_valid_agents=True)
        else:
            step_image = self.raster_map.copy()   
        for agent in agent_list: # for each simulation agent
            self._plot_step_agent(step_image, agent)
        for tl_agent in tl_list: # for each simulation tl head
            self._plot_step_tl(step_image, tl_agent)
        return step_image

    def _plot_step_agent(self, step_image: np.ndarray, agent: Agent):

        ag_pos = np.array([[agent.location.x, agent.location.y]])
        ag_yaw_bbox = np.array([agent.heading])
        bbox_gt = self._to_pixel(
            self._get_agent_bbox(
                np.array([True]),
                ag_pos,
                ag_yaw_bbox,
                np.array([[agent.size.length, agent.size.width]]),
            )
        )
        heading_start = self._to_pixel(ag_pos)
        heading_end = self._to_pixel(
            ag_pos
            + 1.5
            * np.stack([np.cos(ag_yaw_bbox[0]), np.sin(ag_yaw_bbox[0])], axis=-1)
        )
        agent_role = np.array([agent.role.value])
        for i in range(agent_role.shape[0]):
            role_color, control_color = self._get_agent_colors(agent)
            
            cv2.fillConvexPoly(step_image, bbox_gt[i], color=role_color)
            # if control_color:
            #     cv2.polylines(
            #         step_image,
            #         [bbox_gt[i].astype(int)],
            #         isClosed=True,
            #         color=control_color,
            #         thickness=5,
            #         lineType=cv2.LINE_AA,
            #     )
            cv2.arrowedLine(
                step_image,
                heading_start[i],
                heading_end[i],
                color=COLOR_WHITE,
                thickness=4,
                line_type=cv2.LINE_AA,
                tipLength=0.6,
            )

    def _get_agent_colors(self, agent:Agent):

        # filling color that indicates the role
        if agent.role == AgentRole.NORMAL:
            role_color = self.agent_style_default
        else:
            role_color = self.agent_role_style[agent.role.value]
        if self.simplified_agent_color: # and not role_color == COLOR_CYAN:
            role_color = self.agent_style_default

        # outline color that indicates that conrtol state
        if agent.control_state != None:
            control_color = self.agent_control_state_style[agent.control_state.value]
        else:
            control_color = None
        return role_color, control_color


    def _plot_step_tl(self, step_image: np.ndarray, tl_agent: TrafficLightHead):
        tl_state = tl_agent.state
        tl_pos = np.array([tl_agent.location.x, tl_agent.location.y])
        tl_dir = np.array([tl_agent.tl_dir[0], tl_agent.tl_dir[1]])
        stop_point = self._to_pixel(tl_pos)
        stop_point_end = self._to_pixel(tl_pos + 5 * tl_dir)
        if self.simplifed_tls_color:
            tl_state_mapping = {0:0, 1:4,2:5,3:6,4:4,5:5,6:6,7:4,8:5}
            tl_state = tl_state_mapping[tl_state]
        color = self.tl_style[tl_state]
        cv2.arrowedLine(
            step_image,
            stop_point,
            stop_point_end,
            color=color,
            thickness=4,
            line_type=cv2.LINE_AA,
            tipLength=0.3,
        )