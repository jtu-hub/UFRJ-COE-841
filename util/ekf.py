import numpy as np

from control import VelocityControl
from camera import DetectedFeature
from geometry import Angle, Pose

class PositionEstimate:
    def __init__(self, mu = np.zeros((3,1)), sigma0_x = 1e-6, sigma0_y = 1e-6, sigma0_th = 1e-6, r_mat = np.eye(3) * 1e-6):

        self.mu = mu
        self.sigma_mat = np.diagonal(np.array([sigma0_x, sigma0_y, sigma0_th]))
        self.r_mat = r_mat

    def updateEstimateAfterMovement(self, u: VelocityControl):
        mu_old = Pose.from_array(self.mu)
        g_mat = u.g_mat(mu_old)

        self.mu = u.applyControl(mu_old, motion_noise=False).as_array()
        self.sigma_mat = g_mat @ self.sigma_mat @ g_mat.T + self.r_mat

    def updateEstimateAfterMeasurement(self, h_mat_x, k_mat_x, dz):
        self.mu += k_mat_x @ dz
        self.sigma_mat = (np.eye(3) - k_mat_x @ h_mat_x) @ self.sigma_mat


class LandmarkEstimate:
    def __init__(self, detected_feature: DetectedFeature, mu_pose_estimate = np.zeros((3,1)), sigma0_x = 1e-6, sigma0_y = 1e-6, sigma0_s = 1e-6, q_mat = np.eye(3) * 1e-6):

        self.mu = np.array([
            mu_pose_estimate[0,0] + detected_feature.dx(Angle(mu_pose_estimate[2,0])),
            mu_pose_estimate[1,0] + detected_feature.dy(Angle(mu_pose_estimate[2,0])),
            detected_feature.s
        ]).reshape((3,1))

        self.sigma_mat = np.diagonal(np.array([sigma0_x, sigma0_y, sigma0_s]))
        self.q_mat = q_mat

    def updateEstimate(self, detected_feature: DetectedFeature, pos_est: PositionEstimate, num_tol = 1e-6):
        delta = self.mu - pos_est.mu

        q = max(num_tol, delta[0,0]**2 + delta[1,0]**2)
        r_hat   = np.sqrt(q)
        phi_hat = Angle.atan2(delta[1,0], delta[0,0]) - Angle(pos_est.mu[2,0])
        s_hat   = self.mu[2,0]

        z_hat = np.array([r_hat, phi_hat, s_hat]).reshape((3,1))

        h_mat_x, h_mat_lm = self.h_mat(delta[0,0], delta[1,0], q, r_hat, num_tol=num_tol)

        temp_mat = np.linalg.pinv(h_mat_x @ pos_est.sigma_mat @ h_mat_x.T + h_mat_lm @ self.sigma_mat @ h_mat_lm.T + self.q_mat)

        k_mat_x  = pos_est.sigma_mat @ h_mat_x @ temp_mat
        k_mat_lm = self.sigma_mat @ h_mat_lm @ temp_mat

        dz = detected_feature.as_array() - z_hat
        self.mu += k_mat_lm @ dz
        self.sigma_mat = (np.eye(3) - k_mat_lm @ h_mat_lm) @ self.sigma_mat

        return h_mat_x, k_mat_x, dz


    def h_mat(self, dx, dy, q, r, num_tol = 1e-6):
        h_mat_x = np.array([
            [-r * dx, -r * dy,  0],
            [   dy  ,   -dx  , -q],
            [   0   ,    0   ,  0]
        ])

        h_mat_lm = np.array([
            [r * dx, r * dy, 0],
            [ -dy  ,   dx  , 0],
            [  0   ,    0  , q]
        ])
        
        return h_mat_x / max(q, num_tol), h_mat_lm / max(q, num_tol)


class EKFSlamKnownCorrespondences:
    def __init__(self, mu = np.zeros((3,1)), sigma0_x = 1e-6, sigma0_y = 1e-6, sigma0_th = 1e-6, r_mat = np.eye(3) * 1e-6):
        self.pos_est = PositionEstimate(mu, sigma0_x, sigma0_y, sigma0_th, r_mat)

        self.lm_ests = {}

    def update(self, u: VelocityControl, detected_features: list[DetectedFeature], num_tol = 1e-6):

        self.pos_est.updateEstimateAfterMovement(u)

        for detected_feature in detected_features:
            if detected_feature.s not in self.lm_ests:
                self.lm_ests[detected_feature.s] = LandmarkEstimate(detected_feature, self.pos_est.mu)
            
            h_mat_x, k_mat_x, dz = self.lm_ests[detected_feature.s].updateEstimate(detected_feature, self.pos_est, num_tol = num_tol)
            self.pos_est.updateEstimateAfterMeasurement(h_mat_x, k_mat_x, dz)