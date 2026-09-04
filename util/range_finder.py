import matplotlib.pyplot as plt
import numpy as np

from geometry import Pose, Point, Angle
from maps import ObstacleMap, OccupancyGrid
from sensor import RobotSensor

class RayMeasurement:
    def __init__(self, r:float, origin: Pose):
        self.r = r
        self.origin = origin

    def draw(self, ax: plt.Axes, color='b', linestyle='--', **kwargs):

        x_r, y_r, th_r = self.origin.x, self.origin.y, self.origin.th
        
        x_f = x_r + self.r * th_r.cos
        y_f = y_r + self.r * th_r.sin
        
        ax.plot([self.origin.x, x_f],[self.origin.y, y_f], linestyle=linestyle, color=color)

    def __repr__(self):
        return f"DetectedFeature({self.r:.2f}, {self.phi}, {self.s})" 

class RangeFinder(RobotSensor):
    def __init__(self, rel_pos: Pose, sensor_range: float = 5., robot_pose: Pose = Pose(0, 0, 0), beam_fov: Angle = Angle.from_deg(1)):
        super().__init__(rel_pos, robot_pose=robot_pose)

        self.range = sensor_range
        self.beam_fov = beam_fov

    def getReading(self, m: ObstacleMap, detection_noise: bool = True, std_meas_noise = 0.02):
        z_hat = m.ray_cast(self.abs_pos, max_range=self.range)

        if detection_noise:
            z = z_hat + np.random.normal(0, std_meas_noise)
        else:
            z = z_hat

        self.reading = RayMeasurement(z, self.abs_pos)

        return self.reading

    def draw(self, ax: plt.Axes, color='b', linestyle='--', **kwargs):
        if self.reading is not None: self.reading.draw(ax, color=color, linestyle=linestyle, **kwargs)

    def pointInFOV(self, p: Point):
        dx, dy = p.x - self.abs_pos.x, p.y - self.abs_pos.y

        r = np.sqrt(dx**2 + dy**2)
        phi = Angle.atan2(dy, dx) - self.abs_pos.th

        right = Angle(-(self.beam_fov.rad / 2))
        left  = Angle( (self.beam_fov.rad / 2))

        if r < self.range and phi.is_between(right, left):
            return True
        else:
            return False


class RangeFinders(RobotSensor):
    def __init__(self, ray_origins: list[Pose], sensor_range: float = 10, robot_pose: Pose = Pose(0, 0, 0),beam_fov: Angle = Angle.from_deg(1)):
        super().__init__(robot_pose, robot_pose=robot_pose)
        self.z_rangers = [RangeFinder(origin, sensor_range=sensor_range, robot_pose=robot_pose, beam_fov=beam_fov) for origin in ray_origins]
        self.reading = []
        self.beam_fov = beam_fov

    def getReading(self, m: ObstacleMap, detection_noise: bool = True, std_meas_noise = 0.02):
        self.reading = []

        for ranger in self.z_rangers:
            self.reading.append(ranger.getReading(m, detection_noise=detection_noise, std_meas_noise=std_meas_noise))

        return self.reading

    def inverse_range_sensor_model(self, considered_z_rangers, p: Point, alpha: float = 0.1, beta: Angle = Angle.from_deg(1)):
        total = 0.0
        count = 0

        for idx in considered_z_rangers:
            sensor = self.z_rangers[idx]
            reading = self.reading[idx]

            dx = p.x - sensor.x
            dy = p.y - sensor.y

            r_hat = np.sqrt(dx**2 + dy**2)
            th_hat = Angle.atan2(dy, dx) - sensor.th

            if abs(th_hat.deg) > beta / 2 or r_hat > min(reading.r + alpha / 2, self.sensor_range):
                continue

            if reading.r < self.sensor_range and np.abs(r_hat - reading.r) < alpha / 2:
                total += OccupancyGrid.OCCUPIED
                count += 1
            elif r_hat <= reading.r:
                total += OccupancyGrid.FREE
                count += 1

        return np.clip(total / count, OccupancyGrid.FREE, OccupancyGrid.OCCUPIED) if count else OccupancyGrid.UNEXPLORED
            

    def pointInFOV(self, p: Point):
        sensor_idxs = []
        for idx, sensor in enumerate(self.z_rangers):
            if sensor.pointInFOV(p): sensor_idxs.append(idx)

        return sensor_idxs

    def update_occupancy_grid(self, m: OccupancyGrid):
        x_range, y_range = m.getCoordinateIterators()
        for x in x_range():
            for y in y_range():
                p = Point(x,y)
                sensors = self.pointInFOV(p)
                if sensors:
                    m.addExplored(p, self.inverse_range_sensor_model(sensors, p, beta=self.beam_fov))

    def draw(self, ax: plt.Axes, color='b', linestyle='--', **kwargs):
        for ranger in self.z_rangers: ranger.draw(ax, color=color, linestyle=linestyle, **kwargs)