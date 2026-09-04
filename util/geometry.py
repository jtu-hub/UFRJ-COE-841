import matplotlib.pyplot as plt
import numpy as np

class Radians(float):
    pass


class Degrees(float):
    pass


class Angle:
    def __init__(self, rad: Radians | float) -> None:
        self.rad: Radians = Radians(rad)

    def __add__(self, other: "Angle") -> "Angle":
        if not isinstance(other, Angle):
            raise ValueError(
                f"Adding variable of type {type(other)} to Angle"
            )

        new_val = self.rad + other.rad

        return Angle(Radians(new_val)).clip()

    def __sub__(self, other: "Angle") -> "Angle":
        if not isinstance(other, Angle):
            raise ValueError(
                f"Subtracting variable of type {type(other)} from Angle"
            )

        new_val = self.rad - other.rad

        return Angle(Radians(new_val)).clip()

    def __abs__(self) -> Radians:
        return Radians(np.abs(self.rad))

    @staticmethod
    def atan2(y: float, x: float) -> "Angle":
        return Angle(Radians(np.atan2(y, x)))

    @staticmethod
    def from_deg(deg: Degrees | float) -> "Angle":
        return Angle(Radians(deg * np.pi / 180))

    def clip(self) -> "Angle":
        self.rad = Radians(np.atan2(self.sin, self.cos))
        return self

    @property
    def deg(self) -> Degrees:
        return Degrees(self.rad * 180 / np.pi)

    @property
    def sin(self) -> float:
        return float(np.sin(self.rad))

    @property
    def cos(self) -> float:
        return float(np.cos(self.rad))

    def __repr__(self) -> str:
        return f"Angle({self.deg:.2f}°)"

    def is_between(
        self,
        low_bound: "Angle",
        high_bound: "Angle",
    ) -> bool:

        s = low_bound.rad
        e = high_bound.rad
        a = self.rad

        # shift interval to [0, 2pi]
        s_mod = (s + 2 * np.pi) % (2 * np.pi)
        e_mod = (e + 2 * np.pi) % (2 * np.pi)
        a_mod = (a + 2 * np.pi) % (2 * np.pi)

        if s_mod <= e_mod:
            return s_mod <= a_mod <= e_mod

        return a_mod >= s_mod or a_mod <= e_mod # interval wraps around 2pi

class Pose:
    def __init__(self, x, y, th: float | Degrees | Radians | Angle):
        self.x = x
        self.y = y

        if isinstance(th, Angle):
            self.th = th
        elif isinstance(th, Degrees):
            self.th = Angle.from_deg(th)
        else:
            self.th = Angle(th)

    def __add__(self, other):
        return Pose(
            self.x + other.x, 
            self.y + other.y, 
            self.th + other.th
        )
    
    def __repr__(self):
            return f"Pose({self.x:.2f}, {self.y:.2f}, {self.th})"

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    @staticmethod
    def from_tuple(coords: tuple[float, float]):
        return Point(coords[0], coords[1])

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return isinstance(other, Point) and self.x == other.x and self.y == other.y

class Segment:
    def __init__(self, p1: Point | tuple[float,float], p2: Point | tuple[float,float]):
        if isinstance(p1, tuple): p1 = Point.from_tuple(p1) 
        if isinstance(p2, tuple): p2 = Point.from_tuple(p2) 
        self.p1 = p1
        self.p2 = p2

    def draw(self, ax: plt.Axes, color: str = 'black',
             linestyle: str = '-', linewidth: float = 2):
        ax.plot(
            [self.p1.x, self.p2.x],
            [self.p1.y, self.p2.y],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth
        )

    @staticmethod
    def ray_segment_intersection(ray_origin: Pose, segment: 'Segment') -> float | None:

        dx = ray_origin.th.cos
        dy = ray_origin.th.sin

        x, y = ray_origin.x, ray_origin.y
        x1, y1 = segment.p1.x, segment.p1.y

        sx = segment.p2.x - x1
        sy = segment.p2.y - y1
        denominator = dx * sy - dy * sx

        if np.isclose(denominator, 0.0):
            return None

        qx = x1 - x
        qy = y1 - y

        t = (qx * sy - qy * sx) / denominator
        u = (qx * dy - qy * dx) / denominator

        if t < 0:
            return None

        if u < 0 or u > 1:
            return None

        return t
