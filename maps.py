import matplotlib.pyplot as plt
import colorsys
import numpy as np

from pose import Pose

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


class Segment:
    def __init__(self, p1: tuple[float, float], p2: tuple[float, float]):
        self.p1 = p1
        self.p2 = p2

    def draw(self, ax: plt.Axes, color: str = 'black',
             linestyle: str = '-', linewidth: float = 2):
        ax.plot(
            [self.p1[0], self.p2[0]],
            [self.p1[1], self.p2[1]],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth
        )


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

        x1, y1 = segment.p1
        x2, y2 = segment.p2

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