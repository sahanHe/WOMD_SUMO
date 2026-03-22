import numpy as np
from enum import Enum
from typing import Union, Optional
from shapely.geometry import Polygon
from src.static_context.utils import Pt


def get_box_pts_from_center_heading(length, width, xc, yc, heading):
    def _rotate_pt(x, y, a):
        return np.cos(a) * x - np.sin(a) * y, np.sin(a) * x + np.cos(a) * y

    l, w = length / 2.0, width / 2.0

    ## box
    x1, y1 = l, w
    x2, y2 = l, -w
    x3, y3 = -l, -w
    x4, y4 = -l, w

    ## rotation
    a = heading
    x1_, y1_ = _rotate_pt(x1, y1, a)
    x2_, y2_ = _rotate_pt(x2, y2, a)
    x3_, y3_ = _rotate_pt(x3, y3, a)
    x4_, y4_ = _rotate_pt(x4, y4, a)

    ## translation
    pt1 = [x1_ + xc, y1_ + yc]
    pt2 = [x2_ + xc, y2_ + yc]
    pt3 = [x3_ + xc, y3_ + yc]
    pt4 = [x4_ + xc, y4_ + yc]

    return [pt1, pt2, pt3, pt4]


class Size3d:
    def __init__(self, length=None, width=None, height=None):
        self.length: float = length
        self.width: float = width
        self.height: float = height

    def to_list(self) -> list[float]:
        return [self.length, self.width, self.height]


class AgentType(Enum):
    VEHICLE = 0
    PEDESTRIAN = 1
    CYCLIST = 2

class AgentRole(Enum):
    NORMAL = -1
    SDC = 0
    INTEREST = 1
    PREDICT = 2

class AgentControlState(Enum):
    STATIONARY = 0
    SIMPLE_STATIONARY = 1
    FORCE_BREAK = 2
    FORCE_GO = 3
    SIMPLE_MOVE = 4


class AgentAttr:

    def __init__(
        self,
        id: str,
        size: Union[Size3d, list[float]],
        route: Optional[list[str]] = None,
        agent_type: Optional[AgentType] = AgentType.VEHICLE,
        agent_role: Optional[AgentRole] = AgentRole.NORMAL,
    ) -> None:
        self.id: str = id
        self.size: Size3d = size if type(size) == Size3d else Size3d(*size)
        self.route: list[str] = route
        self.type: AgentType = agent_type
        self.role: AgentRole = agent_role


class AgentState:

    def __init__(
        self,
        location: Union[Pt,list[float]] = [0, 0, 0],
        heading: float = None,
        speed: float = None,
        acceleration: float = None,
        yaw_rate: float = None,
        control_state: Optional[AgentControlState] = None,
    ) -> None:

        self.location: Pt = location if type(location) == Pt else Pt(*location)
        # object center position. In SUMO it needs to converted into the front bumper position
        self.heading: float = heading
        # rad east 0 and counter-clockwise. In SUMO it needs to converted into degrees between 0 and 360 with 0 at the top and going clockwise
        self.speed: float = speed
        self.acc: float = acceleration
        self.yaw_rate: float = yaw_rate
        self.control_state: AgentControlState = control_state

    def get_position(self) -> list[int]:
        return [self.location.x, self.location.y, self.location.z]


class Agent(AgentAttr, AgentState):
    def __init__(
        self,
        id: str,
        size: list[float] = [0, 0, 0],
        agent_type: Optional[AgentType] = AgentType.VEHICLE,
        agent_role: Optional[AgentRole] = AgentRole.NORMAL,
        route: list[str]=None,
        control_state: AgentControlState=None,
        location: list[float] = [0, 0, 0],
        heading: float = None,
        speed: float = None,
        acceleration: float = None,
        yaw_rate: float = None,
    ):
        AgentAttr.__init__(
            self, id, size, route=route, agent_type=agent_type, agent_role=agent_role,
        )
        AgentState.__init__(self, location, heading, speed, acceleration, yaw_rate, control_state)
        self.poly_box: Polygon = self.update_poly_box()  # The rectangle of the vehicle using shapely.Polygon

    def update_poly_box(self):
        realworld_4_vertices = get_box_pts_from_center_heading(
            length=self.size.length,
            width=self.size.width,
            xc=self.location.x,
            yc=self.location.y,
            heading=self.heading,
        )
        return Polygon(realworld_4_vertices)


class TrafficLightHead:
    def __init__(
        self,
        id,
        location: list[float] = [0, 0, 0],
        tl_dir: list[float] = [0, 0, 0],
        state: int = None
    ):
        "Configuration for traffic light heads"
        self.id = id  # traffic light head id, string
        self.location = Pt(*location)  # list [x,y,z]
        self.tl_dir: list[float] = tl_dir if len(tl_dir) == 3 else tl_dir + [0]  # list [dir_x, dir_y, dir_z]
        self.state: int = state

TL_STATE_MAPPING = {
    "r": 1,
    "y": 2,
    "G": 3,
    "g": 3,
    "s": 1,  # s in SUMO means'green right-turn arrow'. vehicles may pass the junction if no vehicle uses a higher priorised foe stream. They always stop before passing. This is only generated for junction type traffic_light_right_on_red.
}