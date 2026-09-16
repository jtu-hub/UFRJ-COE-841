from .geometry import Pose, Angle
from .control import VelocityControl
from .distributions import ProbDistribution, GaussianDistribution

import numpy as np

class Odometry:
    def __init__(self, pos_start: Pose, pos_end: Pose):
        self.pos_s = pos_start
        self.pos_e = pos_end

    @property
    def dy(self): 
        return self.pos_e.y - self.pos_s.y
    
    @property
    def dx(self): 
        return self.pos_e.x - self.pos_s.x

    @property
    def  dth_1(self):
        return Angle.atan2(self.dy, self.dx) - self.pos_s.th

    @property
    def dr(self):
        return np.sqrt(self.dx**2 + self.dy**2)

    @property
    def dth_2(self):
        return self.pos_e.th - self.pos_s.th - self.dth_1

    @staticmethod
    def motionModel(hypothesis: 'Odometry', actual: 'Odometry', distribution: ProbDistribution = GaussianDistribution, alphas: tuple[float, float, float, float, float, float] = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1)):
        a1, a2, a3, a4, a5, a6 = alphas

        p1 = distribution(a1 * np.abs(actual.dth_1) + a2 * actual.dr).prob((actual.dth_1 - hypothesis.dth_1).rad)
        p2 = distribution(a3 * np.abs(actual.dth_1) + a4 * (np.abs(actual.dth_1) + np.abs(actual.dth_2))).prob(actual.dr - hypothesis.dr)
        p3 = distribution(a5 * np.abs(actual.dth_2) + a6 * actual.dr).prob((actual.dth_2 - hypothesis.dth_2).rad)

        return p1 * p2 * p3

    @staticmethod
    def sampleMotionModel(x0: Pose, odometry: 'Odometry', distribution: ProbDistribution = GaussianDistribution, alphas: tuple[float, float, float, float, float, float] = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1)):
        a1, a2, a3, a4, a5, a6 = alphas

        dth_1_h = (odometry.dth_1 - Angle(distribution(a1 * odometry.dth_1.rad ** 2 + a2 * odometry.dr **2).sample()))
        dr_h    = odometry.dr - distribution(a3 * odometry.dr ** 2 + a4 * (odometry.dth_1 + odometry.dth_2).rad).sample()
        dth_2_h = (odometry.dth_2 - Angle(distribution(a5 * odometry.dth_2.rad ** 2 + a6 * odometry.dr **2).sample()))

        return Pose(
            x0.x + dr_h * (x0.th + dth_1_h).cos,
            x0.y + dr_h * (x0.th + dth_1_h).sin,
            x0.th + dth_1_h + dth_2_h
        )


class CircularMotion:
    def __init__(self, x_start: Pose, x_end: Pose, dt: float):
        self.x0 = x_start
        self.x1 = x_end
        self.dt = dt

    def toVelocityControl(self):
        return VelocityControl(self.v, self.w, self.dt)
    
    @property
    def dx(self):
        return self.x1.x - self.x0.x

    @property
    def dy(self):
        return self.x1.y - self.x0.y
    
    @property
    def dth(self):
        return self.x1.th - self.x0.th

    @property
    def sum_x(self):
        return self.x1.x + self.x0.x

    @property
    def sum_y(self):
        return self.x1.y + self.x0.y
    
    @property
    def mu(self):
        if self.x0 == self.x1: return 0
        
        num = self.dx * self.x0.th.cos + self.dy * self.x0.th.sin 
        den = 2 * (self.dy * self.x0.th.cos - self.dx * self.x0.th.sin)

        return num / den 

    @property
    def c_rot(self):
        if self.x0 == self.x1: return (self.x0.x, self.x0.y)

        c_x = self.sum_x / 2 - self.mu * self.dy
        c_y = self.sum_y / 2 + self.mu * self.dx

        return (c_x, c_y)

    @property
    def r_rot(self):
        if self.x0 == self.x1: return 0

        c_x, c_y = self.c_rot

        return np.sqrt((self.x0.x - c_x)**2 + (self.x0.y - c_y)**2)

    @property
    def dth_rot(self):
        if self.x0 == self.x1: return 0

        c_x, c_y = self.c_rot

        cos_a = self.x1.x - c_x
        sin_a = self.x1.y - c_y
        cos_b = self.x0.x - c_x
        sin_b = self.x0.y - c_y
        dth_rot = Angle.atan2(sin_a * cos_b - cos_a * sin_b, cos_a * cos_b + sin_a * sin_b)

        return dth_rot
    
    @property
    def w(self):
        if self.x0 == self.x1: return 0

        w = self.dth_rot.rad / self.dt

        y_local = -self.dx * self.x0.th.sin + self.dy * self.x0.th.cos
        
        if y_local > 0:
            # Target is to the left
            if w < 0: 
                w = ( 2 * np.pi + self.dth_rot.rad ) / self.dt
        elif y_local < 0:
            # Target is to the right
            if w > 0: 
                w = (self.dth_rot.rad - 2 * np.pi) / self.dt
        else:
            # Target is directly ahead or behind
            pass

        return w

    @property
    def v(self):
        if self.x0 == self.x1: return 0

        v = self.w * self.r_rot

        y_local = -self.dx * self.x0.th.sin + self.dy * self.x0.th.cos
                
        if y_local > 0:
            # Target is to the left
            pass
        elif y_local < 0:
            # Target is to the right
            v = - self.w * self.r_rot
        else:
            # Target is directly ahead or behind
            pass
        
        return v

    @property
    def gamma(self):
        return self.dth.rad / self.dt - self.w

    @staticmethod
    def motionModel(hypothesis: 'CircularMotion', actual: VelocityControl, distribution: ProbDistribution = GaussianDistribution, alphas: tuple[float, float, float, float, float, float] = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1)):
        a1, a2, a3, a4, a5, a6 = alphas
        p1 = distribution(a1 * actual.v **2 + a2 * actual.w **2).prob(actual.v - hypothesis.v)
        p2 = distribution(a3 * actual.v **2 + a4 * actual.w **2).prob(actual.w - hypothesis.w)
        p3 = distribution(a5 * actual.v **2 + a6 * actual.w **2).prob(hypothesis.gamma)

        return p1 * p2 * p3

    @staticmethod
    def sampleMotionModel(x0: Pose, u: VelocityControl, distribution: ProbDistribution = GaussianDistribution, alphas: tuple[float, float, float, float, float, float] = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1)):
        a1, a2, a3, a4, a5, a6 = alphas

        v = u.v + distribution(a1 * u.v **2 + a2 * u.w **2).sample()
        w = u.w + distribution(a3 * u.v **2 + a4 * u.w **2).sample()
        gamma = distribution(a5 * u.v **2 + a6 * u.w **2).sample()

        r_rot = v / w
        th_rot = x0.th + Angle(w * u.dt)

        return Pose(
            x0.x - r_rot * (x0.th.sin - th_rot.sin),
            x0.y + r_rot * (x0.th.cos - th_rot.cos),
            th_rot + Angle(gamma * u.dt)
        )