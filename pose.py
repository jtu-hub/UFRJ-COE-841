from angle import *

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