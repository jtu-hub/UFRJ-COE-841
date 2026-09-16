from .geometry import Pose
from .maps import Map
import matplotlib.pyplot as plt

class RobotSensor:
    def __init__(self, rel_pos: Pose, robot_pose: Pose = Pose(0, 0, 0)):
        self.rel_pos = rel_pos
        self.robot_pose = robot_pose
        self.reading = None

    def updatePosition(self, robot_pose: Pose):
            self.robot_pose = robot_pose
    
    @property
    def abs_pos(self):
        return Pose(
            self.robot_pose.x + self.rel_pos.x * self.robot_pose.th.cos - self.rel_pos.y * self.robot_pose.th.sin,
            self.robot_pose.y + self.rel_pos.y * self.robot_pose.th.cos + self.rel_pos.x * self.robot_pose.th.sin,
            self.robot_pose.th + self.rel_pos.th
        )
    
    def getReading(self, m: Map, **kwargs):
        raise NotImplementedError()
    
    def draw(self, ax: plt.Axes, **kwargs):
        raise NotImplementedError()
