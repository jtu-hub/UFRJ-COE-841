import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Arc

from geometry import Pose, Angle
from sensor import RobotSensor
from maps import Landmark, LandmarkMap

class DetectedFeature:
    def __init__(self, r, phi, signature):
        self.r = r
        self.phi = phi
        self.s = signature

    def draw(self, ax: plt.Axes, measurement_origin: Pose, color='b', linestyle='--', **kwargs):

        x_r, y_r, th_r = measurement_origin.x, measurement_origin.y, measurement_origin.th
        
        x_f = x_r + self.r * (self.phi + th_r).cos
        y_f = y_r + self.r * (self.phi + th_r).sin
        
        ax.plot([measurement_origin.x, x_f],[measurement_origin.y, y_f], linestyle=linestyle, color=color)

    def __repr__(self):
        return f"DetectedFeature({self.r:.2f}, {self.phi}, {self.s})"

class Camera(RobotSensor):
    def __init__(self, rel_pos: Pose, field_of_veiw: Angle = Angle.from_deg(80.), sensor_range: float = 5., robot_pose: Pose = Pose(0, 0, 0)):
        super().__init__(rel_pos, robot_pose=robot_pose)

        self.fov = field_of_veiw
        self.range = sensor_range

    def readingToRobotFrame(self, reading: DetectedFeature):
        x_s, y_s, th_s = self.abs_pos.x, self.abs_pos.y, self.abs_pos.th
        x_r, y_r, th_r = self.robot_pose.x, self.robot_pose.y, self.robot_pose.th
        r_sf, phi_sf = reading.r, reading.phi

        x_f = x_s + r_sf * (th_s + phi_sf).cos
        y_f = y_s + r_sf * (th_s + phi_sf).sin

        dx = x_f - x_r
        dy = y_f - y_r
        r_rf = np.hypot(dx, dy)
        phi_rf = Angle.atan2(dy, dx) - th_r

        return DetectedFeature(r_rf, phi_rf, reading.s)
    
    def detectLandmark(self, landmark: Landmark, detection_prob: bool = False, detection_noise: bool= False, std_meas_noise: float = 0.2, **kwargs) -> tuple[bool, float]:
        dx, dy = landmark.pos.x - self.abs_pos.x, landmark.pos.y - self.abs_pos.y

        r = np.sqrt(dx**2 + dy**2)
        phi = Angle.atan2(dy, dx) - self.abs_pos.th

        right = Angle(-(self.fov.rad / 2))
        left  = Angle( (self.fov.rad / 2))

        if r < self.range and phi.is_between(right, left):
            d_phi1 = abs(phi - right)
            d_phi2 = abs(phi - left)
            
            p_detect = 1 - 0.25 * (d_phi1 + d_phi2) / self.fov.rad
            
            #probability of detection: 1 at center, linear fall off
            is_detected = not detection_prob or np.random.random() < p_detect 
        else:
            is_detected = False

        r_noisy = r + np.random.normal(0, std_meas_noise) if is_detected else None
        phi_noisy = phi + Angle(np.random.normal(0, std_meas_noise) / np.pi) if is_detected else None

        return (is_detected, r_noisy, phi_noisy) if detection_noise else (is_detected, r, phi)

    def getReading(self, m: LandmarkMap, std_meas_noise: float = 0.1, detection_prob: bool = False, detection_noise: bool= True, **kwargs) -> any:
        if isinstance(m, LandmarkMap):
            detected_landmarks = []
            correspondences = []
            for c in range(len(m.landmarks)):
                l = m.landmarks[c]

                is_in_fov, r, phi = self.detectLandmark(l, std_meas_noise=std_meas_noise, detection_prob=detection_prob, detection_noise=detection_noise, **kwargs)
            
                if is_in_fov:
                    correspondences.append(c)
                    detected_landmarks.append(
                        self.readingToRobotFrame(DetectedFeature(r, phi, l.signature))
                    )
        
            self.reading = {
                "features": detected_landmarks, 
                "correspondences": correspondences
            }

            return self.reading
        
        self.reading = None
        return None
  
    def draw(self, ax: plt.Axes, color='g', linestyle='--', draw_sensor_readings: bool = False, **kwargs) -> None:
        x, y, theta = self.abs_pos.x, self.abs_pos.y, self.abs_pos.th
        half_fov = Angle(self.fov.rad / 2)

        #calculate endpoints of the cone sides
        left_angle = theta + half_fov
        right_angle = theta - half_fov

        x_left = x + self.range * left_angle.cos
        y_left = y + self.range * left_angle.sin

        x_right = x + self.range * right_angle.cos
        y_right = y + self.range * right_angle.sin

        #draw the cone as a triangle
        ax.plot([x, x_left], [y, y_left], linestyle=linestyle, color=color, **kwargs)
        ax.plot([x, x_right], [y, y_right], linestyle=linestyle, color=color, **kwargs)

        if draw_sensor_readings and self.reading is not None:
            for f in self.reading['features']:
                f.draw(ax, self.robot_pose)

        if self.range == np.inf: return

        #draw the circular arc at the bottom
        arc = Arc(
            (x, y),  # Center of the arc
            2 * self.range,  # Width
            2 * self.range,  # Height
            theta1=(right_angle).deg,
            theta2=(left_angle).deg,
            linestyle=linestyle,
            color=color,
            **kwargs
        )
        ax.add_patch(arc)
