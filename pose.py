from angle import *

class Pose:
    def __init__(self, x, y, th: float | Degrees | Radians | Angle, name: str = 'Robot'):
        self.x = x
        self.y = y

        if isinstance(th, Angle):
            self.th = th
        elif isinstance(th, Degrees):
            self.th = Angle.from_deg(th)
        else:
            self.th = Angle(th)

    def __repr__(self):
            return f"Pose({self.x}, {self.y}, {self.th})"