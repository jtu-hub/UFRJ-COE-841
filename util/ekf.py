import numpy as np
import matplotlib.pyplot as plt
import warnings

from motion_models import CircularMotion
from control import VelocityControl
from camera import DetectedFeature
from geometry import Angle, Pose
from robot import Robot
from distributions import GaussianDistribution

class EKFSlamKnownCorrespondences:
    def __init__(self, n_landmarks, r_mat = np.eye(3) * 0.001, q_mat = np.diag([0.5, 0.5])):
        self.n_lm = n_landmarks
        self.n_lm_detected = 0
        self.correspondence = {}

        self.dim_m_est  = 2 # [mu_m_x, mu_m_y]
        self.dim_pos_est = 3 # [mu_x, mu_y, mu_th]
        self.mu    = np.zeros((self.dim_pos_est + self.n_lm * self.dim_m_est,1))
        self.sigma = np.zeros((self.dim_pos_est + self.n_lm * self.dim_m_est, self.dim_pos_est + self.n_lm * self.dim_m_est))
        self.r_mat = r_mat.copy()
        self.q_mat = q_mat.copy()

    @property
    def mu_x(self):
        return self.mu[:self.dim_pos_est].copy()

    @property
    def mu_x_x(self):
        return self.mu_x[0,0].copy()
    
    @property
    def mu_x_y(self):
        return self.mu_x[1,0].copy()

    @property
    def mu_x_th(self):
        return Angle(self.mu_x[2,0].copy()).clip()
    
    @property
    def mu_m(self):
        return self.mu[self.dim_pos_est:].copy()
    
    @property
    def sigma_x(self):
        return self.sigma[:self.dim_pos_est, :self.dim_pos_est].copy()

    @property
    def sigma_m(self):        
        return self.sigma[self.dim_pos_est:, self.dim_pos_est:].copy()

    @property
    def sigma_x_m(self):
        return self.sigma[:self.dim_pos_est, self.dim_pos_est:].copy()
        
    @property
    def sigma_m_x(self):
        return self.sigma[self.dim_pos_est:, :self.dim_pos_est].copy()

    @mu_x.setter
    def mu_x(self, value):
        self.mu[:self.dim_pos_est] = value

    @mu_m.setter
    def mu_m(self, value):
        self.mu[self.dim_pos_est:] = value

    @sigma_x.setter
    def sigma_x(self, value):
        self.sigma[:self.dim_pos_est, :self.dim_pos_est] = value

    @sigma_m.setter
    def sigma_m(self, value):        
        self.sigma[self.dim_pos_est:, self.dim_pos_est:] = value

    @sigma_x_m.setter
    def sigma_x_m(self, value):
        self.sigma[:self.dim_pos_est, self.dim_pos_est:] = value
        
    @sigma_m_x.setter
    def sigma_m_x(self, value):
        self.sigma[self.dim_pos_est:, :self.dim_pos_est] = value
    
    def get_mu_m_i(self, idx):
        if idx >= self.n_lm: 
            raise IndexError(f"Trying to access mu_m for landmark index {idx}, above actual size {self.n_lm}")
        
        i_start = self.dim_pos_est + idx * self.dim_m_est
        i_end = i_start + self.dim_m_est
        return self.mu[i_start:i_end].copy()
    
    def get_sigma_m_i(self, idx):
        if idx >= self.n_lm: 
            raise IndexError(f"Trying to access sigma_m for landmark index {idx}, above actual size {self.n_lm}")
        
        i_start = self.dim_pos_est + idx * self.dim_m_est
        i_end = i_start + self.dim_m_est
        return self.sigma[i_start:i_end, i_start:i_end].copy()

    def get_sigma_m_i_x(self, idx):
        if idx >= self.n_lm: 
            raise IndexError(f"Trying to access sigma_m_x for landmark index {idx}, above actual size {self.n_lm}")
        
        i_start = self.dim_pos_est + idx * self.dim_m_est
        i_end = i_start + self.dim_m_est
        return self.sigma[i_start:i_end, :self.dim_pos_est].copy()

    def get_sigma_x_m_i(self, idx):
        if idx >= self.n_lm: 
            raise IndexError(f"Trying to access sigma_x_m for landmark index {idx}, above actual size {self.n_lm}")
        
        i_start = self.dim_pos_est + idx * self.dim_m_est
        i_end = i_start + self.dim_m_est
        return self.sigma[:self.dim_pos_est, i_start:i_end].copy()

    def set_mu_m_i(self, idx, value):
        if idx >= self.n_lm:
            raise IndexError(f"Trying to access mu_m for landmark index {idx}, above actual size {self.n_lm}")
        
        i_start = self.dim_pos_est + idx * self.dim_m_est
        i_end = i_start + self.dim_m_est
        self.mu[i_start:i_end] = value.copy()

    def set_sigma_m_i(self, idx, value):
        if idx >= self.n_lm:
            raise IndexError(f"Trying to access sigma_m for landmark index {idx}, above actual size {self.n_lm}")
        
        i_start = self.dim_pos_est + idx * self.dim_m_est
        i_end = i_start + self.dim_m_est
        self.sigma[i_start:i_end, i_start:i_end] = value.copy()

    def set_sigma_m_i_x(self, idx, value):
        if idx >= self.n_lm:
            raise IndexError(f"Trying to access sigma_m_x for landmark index {idx}, above actual size {self.n_lm}")
        
        i_start = self.dim_pos_est + idx * self.dim_m_est
        i_end = i_start + self.dim_m_est
        self.sigma[i_start:i_end, :self.dim_pos_est] = value.copy()

    def set_sigma_x_m_i(self, idx, value):
        if idx >= self.n_lm:
            raise IndexError(f"Trying to access sigma_x_m for landmark index {idx}, above actual size {self.n_lm}")
        
        i_start = self.dim_pos_est + idx * self.dim_m_est
        i_end = i_start + self.dim_m_est
        self.sigma[:self.dim_pos_est, i_start:i_end] = value.copy()

    def updatePositionEstimate(self, motion: CircularMotion | VelocityControl):
        if isinstance(motion, CircularMotion):
            u = motion.toVelocityControl()
        else: 
            u = motion.copy()

        mu_old = Pose.from_array(self.mu_x)
        g_mat = u.g_mat(mu_old)

        #propagate mu through system dynamics
        self.mu_x = u.applyControl(mu_old)[0].as_array

        g_sxx = g_mat @ self.sigma_x @ g_mat.T
        g_sxx = (g_sxx + g_sxx.T) / 2.0
        g_sxm = g_mat @ self.sigma_x_m

        self.sigma_x = g_sxx + self.r_mat
        self.sigma_x_m = g_sxm
        self.sigma_m_x = g_sxm.T
        #sigma_m_m stays invariant in the position update

    def h_mat_init(self, r, alpha):
        #alpha being the absolute angle of r, usually mu_x_th + z.phi
        h_mat_x_init = np.array([
            [1, 0, -r * alpha.sin],
            [0, 1,  r * alpha.cos]
        ])

        h_mat_m_init = np.array([
            [alpha.cos, -r * alpha.sin],
            [alpha.sin,  r * alpha.cos]
        ])

        return h_mat_x_init, h_mat_m_init

    def initializeNewLandmark(self, z: DetectedFeature):
        if self.n_lm_detected >= self.n_lm:
            warnings.warn(f"Detcted feature will be ignored!\n---> Detected a new feature {z} when all {self.n_lm} available landmark estimates are already allocated to another landmark")
            return False
            
        self.correspondence[z.s] = self.n_lm_detected

        mu_m_init = np.array([
            self.mu_x_x + z.dx(self.mu_x_th), 
            self.mu_x_y + z.dy(self.mu_x_th)
        ]).reshape((2,1))

        h_mat_x_init, h_mat_m_init = self.h_mat_init(z.r, z.phi + self.mu_x_th)
        sigma_m_init = h_mat_x_init @ self.sigma_x @ h_mat_x_init.T + h_mat_m_init @ self.q_mat @ h_mat_m_init.T
        sigma_m_init = (sigma_m_init + sigma_m_init.T) / 2

        self.set_mu_m_i(self.n_lm_detected, mu_m_init)
        self.set_sigma_m_i(self.n_lm_detected, sigma_m_init)
        self.n_lm_detected += 1

        return True

    def updateLandmarkEstimates(self, z: DetectedFeature):
        if z.s not in self.correspondence:
            if not self.initializeNewLandmark(z): return
    
        i= self.correspondence[z.s]

        delta = self.get_mu_m_i(i) - np.array([self.mu_x_x, self.mu_x_y]).reshape((2,1))
        q = float(delta.T @ delta)

        r_hat   = np.sqrt(q)
        phi_hat = Angle.atan2(delta[1,0], delta[0,0]) - self.mu_x_th
        z_hat = np.array([r_hat, phi_hat.rad]).reshape((2,1))

        dz = (z.as_array[:2] - z_hat)
        dz[1,0] = Angle(dz[1,0]).clip().rad

        h_mat_x, h_mat_m = self.h_mat(delta[0,0].copy(), delta[1,0].copy(), q, r_hat)

        hsh = (
            h_mat_x @ self.sigma_x            @ h_mat_x.T +
            h_mat_m @ self.get_sigma_m_i_x(i) @ h_mat_x.T + #TODO: can be optimized, the term below is the transpose of this one
            h_mat_x @ self.get_sigma_x_m_i(i) @ h_mat_m.T + 
            h_mat_m @ self.get_sigma_m_i(i)   @ h_mat_m.T +
            self.q_mat
        )

        # sh = [sh_x, sh_0, ...sh_N]^T, 
        # - sh_x = sigma_xx     @ h_x.T + sigma_x{m_i}     @ h_m.T; 
        # - sh_k = sigma_{m_k}x @ h_x.T + sigma_{m_k}{m_i} @ h_m.T; for k \in {0, ..., N-1}
        i_start = self.dim_pos_est + i * self.dim_m_est
        sh = (
            self.sigma[:, :self.dim_pos_est]               @ h_mat_x.T + #[sigma_xx     sigma_{m_k}x    ].T @ h_mat_x.T
            self.sigma[:, i_start: i_start+self.dim_m_est] @ h_mat_m.T   #[sigma_x{m_i} sigma_{m_k}{m_i}].T @ h_mat_m.T
        )

        k = np.linalg.solve(hsh, sh.T).T
        k_hx = k @ h_mat_x
        k_hm = k @ h_mat_m

        mu_corr = k @ dz
        self.mu += mu_corr

        s_corr  = np.eye(self.mu.shape[0])
        s_corr[:, :self.dim_pos_est]               -= k_hx
        s_corr[:, i_start: i_start+self.dim_m_est] -= k_hm
        self.sigma = s_corr @ self.sigma
        self.sigma = (self.sigma + self.sigma.T) / 2


    def h_mat(self, dx, dy, q, r, num_tol=1e-6):
        h_mat_x = np.array([
            [-r * dx, -r * dy,  0],
            [   dy  ,   -dx  , -q],
        ])

        h_mat_lm = np.array([
            [r * dx, r * dy],
            [ -dy  ,   dx  ]
        ])

        den = max(q, num_tol)

        return h_mat_x / den, h_mat_lm / den    

    def update(self, motion: CircularMotion | VelocityControl, detected_features: list[DetectedFeature] | None):
        self.updatePositionEstimate(motion)

        if detected_features:
            for feature in detected_features:
                self.updateLandmarkEstimates(feature)

    def draw(self, ax: plt.Axes, lm_colors, **kwargs):
        GaussianDistribution.plotGaussian2D(
            ax, 
            self.mu_x[:2], self.sigma_x[:2,:2],
            n_sigmas= 3, draw_all_sigma= False, draw_mu=False
        )

        Robot(Pose.from_array(self.mu_x), name="Estimate").draw(ax, r=0.3, linestyles=['--','--'])

        for s, i in self.correspondence.items():
            mu = self.get_mu_m_i(i)
            GaussianDistribution.plotGaussian2D(
                ax,
                mu, self.get_sigma_m_i(i),
                n_sigmas= 3, draw_all_sigma= False, draw_mu=False, color=lm_colors[s]
            )

            ax.add_patch(
                plt.Circle((mu[0,0], mu[1,0]), .1, color=lm_colors[s], fill=False, linestyle='-', linewidth=2, label=f"$\\mu_{s}$")
            )