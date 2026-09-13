# %%
from geometry import Pose
from control import VelocityControl
from maps import Map
import matplotlib.pyplot as plt
from sensor import RobotSensor

class DynamicSystem():
    pass

class MotionModel:
    pass

class Robot(MotionModel):
    def __init__(self, pos: Pose, name = "Robot", sensors:list[RobotSensor] = [], collisionAvoidance = None):
        self.pos = pos
        self.name = name
        self.sensors = sensors
        self.has_new_readings = False
        self.collisionAvoidance = collisionAvoidance

    def applyControl(self, u: VelocityControl, **kwargs):
        if self.collisionAvoidance is not None and self.has_new_readings: u = self.collisionAvoidance(u)

        self.pos, u_eff = u.applyControl(self.pos,**kwargs)
        self.has_new_readings = False

        for sensor in self.sensors:
            sensor.updatePosition(self.pos)

        return u_eff

    def measure(self, m: Map, **kwargs):
        for sensor in self.sensors:
            sensor.getReading(m, **kwargs)
        self.has_new_readings = True

    def getCurrentSensorReadings(self):
        readings = []

        for sensor in self.sensors:
            readings.append(sensor.reading if self.has_new_readings else None)

        return readings

    def draw(self, ax: plt.Axes, r = 1, linewidths: list[int | float, int | float] = [1 , 1], colors: list[str, str] = ['blue', 'red'], linestyles: list[str, str] = ['-', '-'], alt_label: None | str = None, draw_sensors: bool = True, draw_sensor_readings=True, **kwargs):
        xx = self.pos.x + r * self.pos.th.cos
        yy = self.pos.y + r * self.pos.th.sin


        #draw orientation idicator (x,y) -> (xx,yy)
        ax.plot([self.pos.x,xx], [self.pos.y,yy], label=None, color=colors[0], linestyle=linestyles[0], linewidth=linewidths[0])

        #draw circle centered at robot position
        ax.add_patch(
            plt.Circle((self.pos.x, self.pos.y), r, color=colors[1], fill=False, linestyle=linestyles[1], label=self.name if alt_label is None else alt_label, linewidth=linewidths[1])
        )

        if draw_sensors:
            for sensor in self.sensors:
                sensor.draw(ax, **kwargs, draw_sensor_readings=self.has_new_readings)

        # Set the aspect of the plot to be equal
        ax.set_aspect('equal', adjustable='box')

        # Add a legend
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())

# %%
