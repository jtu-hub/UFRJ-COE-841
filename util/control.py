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
        theta_new = Angle(x0.th.rad + u_eff.dth.rad)
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

def generateConstantVelocityTrajectory(n_control_inputs: int, v: float, w: float, dt: float):
   return [VelocityControl(v, w, dt) for _ in range(n_control_inputs)]

def generateSquareTrajectory(dt=0.05, total_time=20.0, side_length=6.0, corner_radius=0.6):
    n_steps = int(total_time / dt)

    n_segments = 8  # 4 straights + 4 turns
    t_per_segment = total_time / n_segments

    v_straight = side_length / t_per_segment
    w_straight = 0.0
    n_straight = int(t_per_segment / dt)

    w_turn = (np.pi / 2) / t_per_segment
    v_turn = corner_radius * w_turn
    n_turn = int(t_per_segment / dt)

    controls = []

    for i in range(4):
        controls = [*controls, *generateConstantVelocityTrajectory(n_straight, v_straight, w_straight, dt)]
        controls = [*controls, *generateConstantVelocityTrajectory(n_turn, v_turn, w_turn, dt)]

    # Trim or pad to match total time exactly
    if len(controls) > n_steps:
        controls = controls[:n_steps]
    else:
        while len(controls) < n_steps:
            controls.append(VelocityControl(0, 0, dt))

    return controls


def generateRandomVelocityTrajectory(n_control_inputs: int, dt: float, max_v: float = 1, max_w: float = 0.5):

    """
    Generate a random, continuous trajectory for `n_control_inputs` steps.
    The trajectory is composed of `n_segments` (1 to `n_control_inputs // 3`),
    where each segment is constant, linearly increasing, or linearly decreasing.
    """
    n_segments = np.random.randint(1, n_control_inputs // 3 + 1)
    segment_lengths = np.random.randint(max(1, n_control_inputs // (n_segments + 1)), 1 + n_control_inputs // n_segments, size=n_segments)
    segment_lengths[-1] += n_control_inputs - np.sum(segment_lengths)  # Ensure total length is correct

    # Initialize trajectories
    v_trajectory = np.zeros(n_control_inputs)
    omega_trajectory = np.zeros(n_control_inputs)

    # Generate random segments for v and omega
    for trajectory in [v_trajectory, omega_trajectory]:
        start_idx = 0
        for i in range(n_segments):
            length = segment_lengths[i]
            end_idx = start_idx + length

            # Randomly choose segment type: 0=constant, 1=linear increasing, 2=linear decreasing
            segment_type = np.random.randint(0, 3)

            # Random start and end values for the segment
            if start_idx == 0:
                start_val = np.random.uniform(-1.0, 1.0)  # Random initial value
            else:
                start_val = trajectory[start_idx - 1]  # Ensure continuity

            if segment_type == 0:  # Constant
                end_val = start_val
            else:  # Linear (increasing or decreasing)
                slope = np.random.uniform(-0.5, 0.5)  # Random slope
                if segment_type == 1:  # Increasing
                    slope = abs(slope)
                else:  # Decreasing
                    slope = -abs(slope)
                end_val = start_val + slope * length

            # Fill the segment
            if length == 1:
                trajectory[start_idx:end_idx] = start_val
            else:
                trajectory[start_idx:end_idx] = np.linspace(start_val, end_val, length)

            start_idx = end_idx

    omega_trajectory -= np.mean(omega_trajectory)
    v_trajectory -= np.mean(v_trajectory)
    us = []
    for v,w in zip(v_trajectory, omega_trajectory):
        if abs(v) >= max_v:
            v = np.sign(v) * max_v

        if abs(w) >= max_w:
            w = np.sign(w) * max_w

        us.append(VelocityControl(v,w,dt))

    return us