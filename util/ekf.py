import numpy as np
import matplotlib.pyplot as plt

from motion_models import CircularMotion
from control import VelocityControl
from camera import DetectedFeature
from geometry import Angle, Pose
from robot import Robot
from distributions import GaussianDistribution

class PositionEstimate:
    def __init__(self, mu=None, sigma0_x=0.1, sigma0_y=0.1, sigma0_th=np.pi / 12, r_mat=None, q_mat=None):
        if mu is None: mu = np.zeros((3, 1))
        if r_mat is None: r_mat = np.eye(3) * 0.01
        if q_mat is None: q_mat = np.eye(3) * 0.001

        self.mu = np.array(mu, dtype=float, copy=True)
        self.sigma_pos = np.diag(np.array([sigma0_x, sigma0_y, sigma0_th],dtype=float) ** 2)
        self.r_mat = np.array(r_mat, dtype=float, copy=True)
        self.q_mat = np.array(q_mat, dtype=float, copy=True)

    def updateEstimateAfterMovement(self, motion: CircularMotion):
        u = motion.toVelocityControl()

        mu_old = Pose.from_array(self.mu)
        g_mat = u.g_mat(mu_old)

        self.mu, _ = u.applyControl(mu_old, motion_noise=False)
        self.mu = self.mu.as_array
        self.sigma_pos = g_mat @ self.sigma_pos @ g_mat.T + self.r_mat

        return g_mat

    def updateEstimateAfterMeasurement(self, h_mat_x, k_mat_x, dz):
        self.mu += k_mat_x @ dz
        self.mu[2,0] = Angle(self.mu[2,0]).clip().rad
        temp_mat = (np.eye(3) - k_mat_x @ h_mat_x)
        self.sigma_pos = temp_mat @ self.sigma_pos @ temp_mat.T + k_mat_x @ self.q_mat @ k_mat_x.T

    def draw(self, ax: plt.Axes, **kwargs):
        Robot(Pose.from_array(self.mu), name="Estimate").draw(ax, r=0.1, linestyles=['--','--'])
        GaussianDistribution.plotGaussian2D(ax, self.mu[:2,:],self.sigma_pos[:2,:2], n_sigmas=3, draw_all_sigma=False, draw_mu=False)


class LandmarkEstimate:
    def __init__(self, detected_feature: DetectedFeature, mu_pose_estimate=None, sigma0_x=0.1, sigma0_y=0.1, sigma0_s=1e-6, q_mat=None, sigma_pos=None):
        if mu_pose_estimate is None: mu_pose_estimate = np.zeros((3, 1))
        if q_mat is None: q_mat = np.eye(3) * 0.001

        self.q_mat = np.array(q_mat, dtype=float, copy=True)

        mu_pose_estimate = np.array(mu_pose_estimate, dtype=float, copy=False)
        th = Angle(mu_pose_estimate[2,0])

        self.mu = np.array([
            mu_pose_estimate[0,0] + detected_feature.dx(th), 
            mu_pose_estimate[1,0] + detected_feature.dy(th),
            detected_feature.s
        ]).reshape((3,1))

        if sigma_pos is None:
            sigma_pos = np.diag(np.array([sigma0_x, sigma0_y, np.pi / 12], dtype=float) ** 2)

        sigma_pos = np.array(sigma_pos, dtype=float, copy=False)

        h_lm_x, h_lm_z = self.h_mat_init(detected_feature, th)

        self.sigma_lm = (h_lm_x @ sigma_pos @ h_lm_x.T + h_lm_z @ self.q_mat @ h_lm_z.T)
        self.sigma_lm_x = (sigma_pos @ h_lm_x.T)

        self.sigma0_x = sigma0_x
        self.sigma0_y = sigma0_y
        self.sigma0_s = sigma0_s

    def updateEstimate(self, detected_feature: DetectedFeature, pos_est: PositionEstimate, num_tol=1e-6):
        delta = self.mu - pos_est.mu

        q = max(num_tol, delta[0,0]**2 + delta[1,0]**2)
        r_hat   = np.sqrt(q)
        phi_hat = Angle.atan2(delta[1,0], delta[0,0]) - Angle(pos_est.mu[2,0])
        s_hat   = self.mu[2,0]

        h_mat_x, h_mat_lm = self.h_mat(delta[0,0], delta[1,0], q, r_hat, num_tol=num_tol)

        innovation_cov = (
            h_mat_x @ pos_est.sigma_pos @ h_mat_x.T + 
            h_mat_x @ self.sigma_lm_x @ h_mat_lm.T + 
            h_mat_lm @ self.sigma_lm_x.T @ h_mat_x.T + 
            h_mat_lm @ self.sigma_lm @ h_mat_lm.T +
            self.q_mat
        )

        gain_x = pos_est.sigma_pos @ h_mat_x.T + self.sigma_lm_x @ h_mat_lm.T

        gain_lm = self.sigma_lm_x.T @ h_mat_x.T + self.sigma_lm @ h_mat_lm.T

        k_mat_x = np.linalg.solve(innovation_cov, gain_x.T ).T
        k_mat_lm = np.linalg.solve(innovation_cov, gain_lm.T).T

        dz = np.array([
            detected_feature.r - r_hat,
            (detected_feature.phi - phi_hat).rad,
            detected_feature.s - s_hat
        ]).reshape((3, 1))

        pos_est.mu += k_mat_x @ dz
        pos_est.mu[2, 0] = Angle(pos_est.mu[2, 0]).clip().rad

        self.mu += k_mat_lm @ dz

        sigma_joint = np.block([
            [pos_est.sigma_pos, self.sigma_lm_x],
            [self.sigma_lm_x.T, self.sigma_lm  ]
        ])

        h_joint = np.hstack([
            h_mat_x,
            h_mat_lm
        ])

        k_joint = np.vstack([k_mat_x, k_mat_lm])

        joseph_temp = np.eye(6) - k_joint @ h_joint

        sigma_joint = (
            joseph_temp @ sigma_joint @ joseph_temp.T + 
            k_joint @ self.q_mat @ k_joint.T
        )
        #make numerically fully simmetric
        sigma_joint = (sigma_joint + sigma_joint.T) / 2.0

        # Extract the updated covariance blocks.
        pos_est.sigma_pos = sigma_joint[:3, :3]
        self.sigma_lm_x   = sigma_joint[:3, 3:]
        self.sigma_lm     = sigma_joint[3:, 3:]

    def h_mat(self, dx, dy, q, r, num_tol=1e-6):
        h_mat_x = np.array([
            [-r * dx, -r * dy,  0],
            [  -dy  ,    dx  , -q],
            [   0   ,    0   ,  0]
        ])

        h_mat_lm = np.array([
            [r * dx, r * dy, 0],
            [ -dy  ,   dx  , 0],
            [  0   ,    0  , q]
        ])

        den = max(q, num_tol)

        return h_mat_x / den, h_mat_lm / den

    def h_mat_init(self, detected_feature: DetectedFeature, mu_pos_est_th: Angle):
        cos_a = (detected_feature.phi + mu_pos_est_th).cos
        sin_a = (detected_feature.phi + mu_pos_est_th).sin
        r = detected_feature.r

        h_lm_x = np.array([
            [1.0, 0.0, -r * sin_a],
            [0.0, 1.0,  r * cos_a],
            [0.0, 0.0,  0.0]
        ])

        h_lm_z = np.array([
            [cos_a, -r * sin_a, 0.0],
            [sin_a,  r * cos_a, 0.0],
            [0.0,         0.0,          1.0]
        ])

        return h_lm_x, h_lm_z
    
    def draw(self, ax: plt.Axes, color='b', **kwargs):
        ax.add_patch(
            plt.Circle((self.mu[0,0], self.mu[1,0]), .1, color=color, fill=False, linestyle='-', linewidth=2, label="$\\mu$")
        )
        GaussianDistribution.plotGaussian2D(ax, self.mu[:2,:],self.sigma_lm[:2,:2], color=color, n_sigmas=3, draw_all_sigma=False, draw_mu=False)


class EKFSlamKnownCorrespondences:
    def __init__(self, mu=None, **kwargs):
        if mu is None: mu = np.zeros((3, 1))

        self.pos_est = PositionEstimate(mu, **kwargs)
        self.lm_ests = {}

    def update(self, motion: CircularMotion, detected_features: list[DetectedFeature], num_tol=1e-6):
        g_mat = self.pos_est.updateEstimateAfterMovement(motion)

        for landmark in self.lm_ests.values():
            landmark.sigma_lm_x = (g_mat @ landmark.sigma_lm_x)

        for detected_feature in detected_features:
            if detected_feature.s not in self.lm_ests:
                self.lm_ests[detected_feature.s] = LandmarkEstimate(detected_feature, self.pos_est.mu, q_mat=self.pos_est.q_mat, sigma_pos=self.pos_est.sigma_pos)

            self.lm_ests[detected_feature.s].updateEstimate(detected_feature, self.pos_est, num_tol=num_tol)

    def draw(self, ax: plt.Axes, lm_colors, **kwargs):
        for lm in self.lm_ests:
            self.lm_ests[lm].draw(ax, color=lm_colors[lm])

        self.pos_est.draw(ax)

