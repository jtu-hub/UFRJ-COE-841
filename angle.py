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
        return f"Angle({self.deg:.2f}°)"

    def is_between(self, low_bound: 'Angle', high_bound: 'Angle'):
        s = low_bound.rad
        e = high_bound.rad
        a = self.rad

        # shift interval to [0, 2pi]
        s_mod = (s + 2*np.pi) % (2*np.pi)
        e_mod = (e + 2*np.pi) % (2*np.pi)
        a_mod = (a + 2*np.pi) % (2*np.pi)

        if s_mod <= e_mod:
            return s_mod <= a_mod <= e_mod
        else:
            # interval wraps around 2pi
            return a_mod >= s_mod or a_mod <= e_mod
