import numpy as np

from pose import Pose
from maps import ObstacleMap
from ray_cast import ray_cast


class RangeFinder:
    def __init__(
        self,
        robot_pose: Pose,
        beam_angles: list[float],
        max_range: float = 10.0,
        sigma: float = 0.1
    ):
        self.robot_pose = robot_pose
        self.beam_angles = beam_angles
        self.max_range = max_range
        self.sigma = sigma

        self.reading = None

    def getReading(
        self,
        obstacle_map: ObstacleMap,
        detection_noise: bool = True
    ):
        measurements = []

        for beam_angle in self.beam_angles:

            # Direção absoluta do feixe
            angle = self.robot_pose.th.rad + beam_angle

            # Distância esperada pelo mapa
            z_hat = ray_cast(
                (self.robot_pose.x, self.robot_pose.y),
                angle,
                obstacle_map,
                max_range=self.max_range
            )

            # Simulação da medição real
            if detection_noise:
                z = z_hat + np.random.normal(0, self.sigma)
            else:
                z = z_hat

            # O sensor não pode retornar valores fora do alcance
            z = np.clip(z, 0.0, self.max_range)

            measurements.append(z)

        self.reading = measurements

        return self.reading