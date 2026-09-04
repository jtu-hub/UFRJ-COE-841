from geometry import Pose, Angle

import numpy as np

class ControlInput:
   def applyControl(*args, **kwargs):
      raise NotImplemented

class OdometryControl(ControlInput):
    def __init__(self, th_1: Angle, r: float, th_2: Angle):
        super().__init__()
        self.th_1 = th_1
        self.r = r
        self.th_2 = th_2

    def copy(self):
       return OdometryControl(self.th_1, self.r, self.th_2)
    
    def applyControl(self, x0: Pose, motion_noise: bool = False, std_mot_noise: tuple[float, float, float] = (0.001, 0.002, 0.001)):
        if motion_noise:
            th1_eff = self.th_1 + np.random.normal(0, std_mot_noise[0])
            r_eff = self.r + np.random.normal(0, std_mot_noise[1])
            th2_eff = self.th_2 + np.random.normal(0, std_mot_noise[2])
    
            r_eff = r_eff if abs(r_eff) > 0. else 0.
            th1_eff = th1_eff if abs(th1_eff) > 0. else 0.
            th2_eff = th2_eff if abs(th2_eff) > 0. else 0.
    
            u_eff = OdometryControl(th1_eff, r_eff, th2_eff)
        else:
            u_eff = self.copy()
       
        theta_new = x0.th + u_eff.th_1
        x_new = x0.x + u_eff.r * theta_new.cos
        y_new = x0.y + u_eff.r * theta_new.sin
        theta_new += u_eff.th_2

        return Pose(x_new, y_new, theta_new), u_eff


class VelocityControl(ControlInput):
  def __init__(self, v : float, w : float, dt : float):
    super().__init__()
    self.dt= dt
    self.v = v
    self.w = w

  def __add__(self, other):
    v = self.v + other.v
    w = self.w + other.w
    dt = (self.dt + other.dt) / 2

    return VelocityControl(v,w,dt)

  def __radd__(self, other):
    return self.__add__(other)
  def __iadd__(self, other):
    return self.__add__(other)

  def copy(self):
    return VelocityControl(self.v, self.w, self.dt)

  
  def applyControl(self, x0: Pose, motion_noise: bool = False, std_mot_noise: tuple[float, float] = (0.002, 0.001)) -> tuple[Pose, 'VelocityControl']:
    if motion_noise:
        v_eff = self.v + np.random.normal(0, std_mot_noise[0])
        w_eff = self.w + np.random.normal(0, std_mot_noise[1])

        v_eff = v_eff if abs(v_eff) > 0. else 0.
        w_eff = w_eff if abs(w_eff) > 0. else 0.

        u_eff = VelocityControl(
            v_eff,
            w_eff,
            self.dt
        )
    else:
        u_eff = self.copy()

    if np.isclose(u_eff.w, 0.):
        #straight-line motion
        x_new = x0.x + u_eff.v * x0.th.cos * u_eff.dt
        y_new = x0.y + u_eff.v * x0.th.sin * u_eff.dt
        theta_new = x0.th
    else:
        #curved motion (unicycle model)
        theta_new = x0.th + u_eff.dth
        x_new = x0.x + u_eff.r * ( theta_new.sin - x0.th.sin)
        y_new = x0.y + u_eff.r * (-theta_new.cos + x0.th.cos)

    return Pose(x_new, y_new, theta_new), u_eff
  
  @property
  def dth(self):
    return Angle(self.w * self.dt)
  
  @property
  def dl(self):
    if np.isclose(self.w, 0.):
        return self.v * self.dt
    
    return self.r * float(self.dth)
  
  @property
  def r(self):
    if np.isclose(self.w, 0.):
        return np.inf
    
    return self.v / self.w
  
  def g_mat(self, x0: Pose):
    """
    Jacobian of transfer function, see applyControl(...) w.r.t. control 
    variables x0
    """
    #best estimate of angle before moovement
    sin_th = x0.th.sin
    cos_th = x0.th.cos
        
    if np.isclose(self.w, 0.):
        return np.array([
            [1, 0, -self.dl * sin_th],
            [0, 1,  self.dl * cos_th],
            [0, 0,          1       ]
        ])
    
    #final angle after control input
    sin_th_f = (x0.th + self.dth).sin
    cos_th_f = (x0.th + self.dth).cos

    d_sin = sin_th_f - sin_th
    d_cos = cos_th_f - cos_th

    return np.array([
        [1, 0, self.r * d_cos],
        [0, 1, self.r * d_sin],
        [0, 0,        1      ]
    ])
  
  def v_mat(self, x0: Pose):
    """
    Jacobian of transfer function, see applyControl(...) w.r.t. control 
    variables x0
    """
    #best estimate of angle before moovement
    sin_th = x0.th.sin
    cos_th = x0.th.cos
    
    if np.isclose(self.w, 0.):
        return np.array([
            [self.dt * cos_th, 0],
            [self.dt * sin_th, 0],
            [        0       , 0]
        ])
    
    #final angle after control input
    sin_th_f = (x0.th + self.dth).sin
    cos_th_f = (x0.th + self.dth).cos

    d_sin = sin_th_f - sin_th
    d_cos = cos_th_f - cos_th

    return np.array([
            [ d_sin / self.w, self.r * (cos_th_f * self.dt - d_sin / self.w)],
            [-d_cos / self.w, self.r * (sin_th_f * self.dt + d_cos / self.w)],
            [      0     ,                       self.dt                    ]
    ])
  
  def __str__(self):
     return f"VelocityControl({self.v}, {self.w}, {self.dt})"