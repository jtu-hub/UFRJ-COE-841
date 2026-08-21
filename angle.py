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