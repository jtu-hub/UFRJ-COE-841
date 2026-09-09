import matplotlib.pyplot as plt
import colorsys
import numpy as np
from typing import Dict

from geometry import Pose, Segment, Point

class Landmark:
    def __init__(self, pos: Pose, signature: any):
        self.pos = pos
        self.signature = signature

    def draw(self, ax: plt.Axes, color: str = 'red', set_label = False):
        ax.scatter(self.pos.x, self.pos.y, c=color, label = f"Landmark {self.signature}" if set_label else None)

class Map: pass

class LandmarkMap(Map):
    def __init__(self, landmarks: list[Landmark] | None = None, x_lims: tuple[float, float] = (-5, 5), y_lims: tuple[float, float] = (-5, 5)):
        super().__init__()

        if landmarks is None:
            landmarks = []

        self.landmarks = landmarks
        self.colors = []
        self.x_lims = x_lims
        self.y_lims = y_lims        

        self._generateLandmarkColors()

    def _generateLandmarkColors(self):
        n = len(self.landmarks)

        if n == 0: return

        self.colors = []

        for i in range(n):
            
            hue = (i + 1) / n
            saturation = (90 + np.random.rand() * 10) / 100
            value = (50 + np.random.rand() * 10) / 100 # lightness

            rgb = colorsys.hsv_to_rgb(hue, saturation, value)

            hex_color = f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"

            self.colors.append(hex_color)


    def draw(self, ax: plt.Axes, landmark_labels: bool = False):
        for l_idx, l in enumerate(self.landmarks):
            l.draw(ax, color = self.colors[l_idx], set_label=landmark_labels)

        ax.set_xlim(self.x_lims[0], self.x_lims[1])
        ax.set_ylim(self.y_lims[0], self.y_lims[1])


class ObstacleMap(Map):
    def __init__(
        self,
        segments: list[Segment] | None = None,
        x_lims: tuple[float, float] = (-5, 5),
        y_lims: tuple[float, float] = (-5, 5)
    ):
        super().__init__()

        if segments is None:
            segments = []

        self.segments = segments
        self.x_lims = x_lims
        self.y_lims = y_lims

    # =========================================================
    # Distance between a point and a segment
    # =========================================================

    @staticmethod
    def _point_to_segment_distance(
        point: tuple[float, float],
        segment: Segment
    ) -> float:

        px, py = point

        x1, y1 = segment.p1.x, segment.p1.y
        x2, y2 = segment.p2.x, segment.p2.y

        # Segment vector
        vx = x2 - x1
        vy = y2 - y1

        # Vector from p1 to the point
        wx = px - x1
        wy = py - y1

        segment_length_squared = vx**2 + vy**2

        # Degenerate segment: p1 == p2
        if np.isclose(segment_length_squared, 0.0):
            return np.sqrt(
                (px - x1)**2 +
                (py - y1)**2
            )

        # Projection of point onto the infinite line
        t = (
            wx * vx +
            wy * vy
        ) / segment_length_squared

        # Restrict projection to the segment
        t = np.clip(t, 0.0, 1.0)

        # Closest point on the segment
        closest_x = x1 + t * vx
        closest_y = y1 + t * vy

        # Euclidean distance
        return np.sqrt(
            (px - closest_x)**2 +
            (py - closest_y)**2
        )

    def ray_cast(
        self,
        ray_origin: Pose,
        max_range: float = np.inf
    ) -> float:
        
        min_distance = max_range

        for segment in self.segments:

            distance = Segment.ray_segment_intersection(
                ray_origin,
                segment
            )

            if distance is not None and distance < min_distance:
                min_distance = distance

        return min_distance
    # =========================================================
    # Distance from a point to the nearest obstacle
    # =========================================================

    def distance_to_nearest_obstacle(
        self,
        x: float,
        y: float
    ) -> float:

        if len(self.segments) == 0:
            return np.inf

        point = (x, y)

        distances = [
            self._point_to_segment_distance(
                point,
                segment
            )
            for segment in self.segments
        ]

        return min(distances)

    # =========================================================
    # Check whether a position is valid
    # =========================================================

    def is_valid_position(
        self,
        x: float,
        y: float,
        min_distance: float = 0.2
    ) -> bool:

        # Check map boundaries
        if x < self.x_lims[0] or x > self.x_lims[1]:
            return False

        if y < self.y_lims[0] or y > self.y_lims[1]:
            return False

        # Check distance to obstacles
        distance = self.distance_to_nearest_obstacle(
            x,
            y
        )

        return distance >= min_distance

    # =========================================================
    # Draw map
    # =========================================================

    def draw(
        self,
        ax: plt.Axes,
        color: str = 'black',
        linestyle: str = '-',
        linewidth: float = 1
    ):

        for segment in self.segments:

            segment.draw(
                ax,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth
            )

        ax.set_xlim(
            self.x_lims[0],
            self.x_lims[1]
        )

        ax.set_ylim(
            self.y_lims[0],
            self.y_lims[1]
        )

class DistanceField:

    def __init__(
        self,
        obstacle_map,
        resolution=0.05
    ):
        self.obstacle_map = obstacle_map
        self.resolution = resolution

        self.x_min, self.x_max = obstacle_map.x_lims
        self.y_min, self.y_max = obstacle_map.y_lims

        self.nx = int(
            np.ceil(
                (self.x_max - self.x_min)
                / resolution
            )
        ) + 1

        self.ny = int(
            np.ceil(
                (self.y_max - self.y_min)
                / resolution
            )
        ) + 1

        self.distances = np.zeros(
            (self.ny, self.nx)
        )

        self.build()

    def world_to_grid(self, x, y):

        ix = int(
            np.round(
                (x - self.x_min)
                / self.resolution
            )
        )

        iy = int(
            np.round(
                (y - self.y_min)
                / self.resolution
            )
        )

        return ix, iy

    def grid_to_world(self, ix, iy):

        x = (
            self.x_min
            + ix * self.resolution
        )

        y = (
            self.y_min
            + iy * self.resolution
        )

        return x, y

    def build(self):

        for iy in range(self.ny):

            for ix in range(self.nx):

                x, y = self.grid_to_world(
                    ix,
                    iy
                )

                self.distances[iy, ix] = (
                    self.obstacle_map
                    .distance_to_nearest_obstacle(x, y)
                )

    def distance_at(self, x, y):

        ix, iy = self.world_to_grid(x, y)

        if (
            ix < 0
            or ix >= self.nx
            or iy < 0
            or iy >= self.ny
        ):
            return np.inf

        return self.distances[iy, ix]

    def draw(self, ax):

        ax.imshow(
            self.distances,
            origin="lower",
            extent=[
                self.x_min,
                self.x_max,
                self.y_min,
                self.y_max
            ],
            aspect="equal"
        )

class OccupancyGrid:
    UNEXPLORED = 0.0
    FREE = -1.0
    OCCUPIED = 1.0

    def __init__(self, x_limits: tuple[float, float], y_limits: tuple[float, float], resolution: float):
        self.explored: Dict[Point, float] = {}
        self.x_lims = x_limits
        self.y_lims = y_limits
        self.resolution = resolution

    def addExplored(self, coordinates: Point, likelihood: float) -> None:
        if np.isclose(likelihood, self.UNEXPLORED): return

        if coordinates in self.explored:
            old_value = self.explored[coordinates]
            new_value = (old_value + likelihood) / 2
            if np.isclose(new_value, self.UNEXPLORED):
                del self.explored[coordinates]
            else:
                self.explored[coordinates] = new_value
        else:
            self.explored[coordinates] = likelihood

    def draw(self, ax: plt.Axes) -> None:
        for coord, likelihood in self.explored.items():
            if likelihood < self.UNEXPLORED:
                color = 'green'
                alpha = np.clip(likelihood / self.FREE, 0, 1)
            else:
                color = 'blue'
                alpha = np.clip(likelihood / self.OCCUPIED, 0, 1)
            ax.scatter(coord.x, coord.y, c=color, alpha=alpha, marker='s')
        ax.set_xlim(self.x_lims[0] - self.resolution/2, self.x_lims[1] + self.resolution/2)
        ax.set_ylim(self.y_lims[0] - self.resolution/2, self.y_lims[1] + self.resolution/2)  
        ax.set_aspect('equal', adjustable='box')    

    def drawImg(self, ax: plt.Axes):
        nx = int(((self.x_lims[1] + self.resolution) - self.x_lims[0]) / self.resolution)
        ny = int(((self.y_lims[1] + self.resolution) - self.y_lims[0]) / self.resolution)

        img = np.zeros((nx,ny))

        for coord, likelihood in self.explored.items():
            img[int((coord.y - self.y_lims[0]) / self.resolution), int((coord.x - self.y_lims[0]) / self.resolution)] = likelihood

        ax.imshow(img, cmap='Blues', extent=[self.x_lims[0], self.x_lims[1], self.y_lims[0], self.y_lims[1]], origin='lower')

