# %%
import numpy as np

class Radians(float):
    pass

class Degrees(float):
    pass

class Angle:
    def __init__(self, rad: Radians | float):
        self.rad = rad

    def __add__(self, other):
        if isinstance(other, Angle):
            new_val = self.rad + other.rad
        else:
            new_val = self.rad + other
            raise ValueError(f"Adding variable of type {type(other)} to Angle")

        return Angle(new_val).clip()

    def __sub__(self, other):
        if isinstance(other, Angle):
            new_val = self.rad - other.rad
        else:
            new_val = self.rad - other
            raise ValueError(f"Adding variable of type {type(other)} to Angle")

        return Angle(new_val).clip()

    def __abs__(self):
        return np.abs(self.rad)

    @staticmethod
    def atan2(y, x):
        return Angle(np.atan2(y, x))

    @staticmethod
    def from_deg(deg: Degrees | float):
        return Angle(deg * np.pi / 180)


    def clip(self):
        self.rad = np.atan2(self.sin, self.cos)

        return self

    @property
    def deg(self):
        return self.rad * 180 / np.pi

    @property
    def sin(self):
        return np.sin(self.rad)

    @property
    def cos(self):
        return np.cos(self.rad)

    def __repr__(self):
        return f"Angle({self.deg}°)"

