from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from geometry import Angle, Pose
from control import VelocityControl
from motion_models import CircularMotion
from camera import DetectedFeature


@dataclass
class LandmarkEstimate:
    """Estimated state of one landmark inside a FastSLAM particle."""

    mu: np.ndarray
    sigma: np.ndarray
    seen_count: int = 1

    def __post_init__(self):
        self.mu = np.asarray(self.mu, dtype=float).reshape(2, 1)
        self.sigma = np.asarray(self.sigma, dtype=float).reshape(2, 2)


@dataclass
class Particle:
    """One FastSLAM particle: robot pose + independent landmark EKFs."""

    pose: Pose
    landmarks: dict[int, LandmarkEstimate] = field(default_factory=dict)
    weight: float = 1.0
    associations: list[int] = field(default_factory=list)


class FastSLAM:
    """
    FastSLAM 1.0 with maximum-likelihood data association.

    Parameters
    ----------
    n_particles:
        Number of particles.
    r_mat:
        Motion-noise covariance in [v, w] parameter space is not used
        directly here; motion sampling uses VelocityControl.applyControl().
        Kept as an optional configuration value for future extensions.
    q_mat:
        Measurement covariance for [range, bearing].
    p0:
        Likelihood assigned to the new-landmark hypothesis.
    """

    def __init__(
        self,
        n_particles: int = 100,
        r_mat: np.ndarray | None = None,
        q_mat: np.ndarray | None = None,
        p0: float = 1e-3,
    ):
        if n_particles <= 0:
            raise ValueError("n_particles must be positive.")

        if q_mat is None:
            q_mat = np.diag([0.05**2, np.deg2rad(1.0)**2])

        self.n_particles = n_particles
        self.r_mat = None if r_mat is None else np.asarray(r_mat, dtype=float)
        self.q_mat = np.asarray(q_mat, dtype=float).reshape(2, 2)
        self.p0 = float(p0)

        self.particles: list[Particle] = []

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, pose: Pose) -> None:
        """Initialize all particles at the supplied initial pose."""
        self.particles = [
            Particle(
                pose=Pose(pose.x, pose.y, pose.th),
                weight=1.0 / self.n_particles,
            )
            for _ in range(self.n_particles)
        ]

    # ------------------------------------------------------------------
    # Motion model
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_motion(
        pose: Pose,
        motion,
        motion_noise: bool = True,
    ) -> Pose:
        """
        Sample a new pose using the project's existing motion model.

        The project motion classes already implement the noisy control
        model, so FastSLAM reuses them rather than duplicating kinematics.
        """
        if isinstance(motion, CircularMotion):
            motion = motion.toVelocityControl()

        if not isinstance(motion, VelocityControl):
            raise TypeError(
                "FastSLAM expects VelocityControl or CircularMotion motion."
            )

        new_pose, _ = motion.applyControl(
            pose,
            motion_noise=motion_noise,
        )

        return new_pose

    # ------------------------------------------------------------------
    # Measurement model
    # ------------------------------------------------------------------

    @staticmethod
    def predict_measurement(
        pose: Pose,
        landmark_mu: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Predict [range, bearing] for a landmark.

        Returns
        -------
        z_hat:
            Predicted measurement, shape (2, 1).
        delta:
            Landmark position relative to robot, shape (2, 1).
        q:
            Squared range.
        """
        landmark_mu = np.asarray(landmark_mu, dtype=float).reshape(2, 1)

        delta = landmark_mu - np.array(
            [pose.x, pose.y],
            dtype=float,
        ).reshape(2, 1)

        q = float((delta.T @ delta).item())

        if np.isclose(q, 0.0):
            raise ValueError(
                "Cannot predict a bearing for a landmark at the robot pose."
            )

        r_hat = np.sqrt(q)

        phi_hat = (
            Angle.atan2(delta[1, 0], delta[0, 0])
            - pose.th
        )

        z_hat = np.array(
            [r_hat, phi_hat.rad],
            dtype=float,
        ).reshape(2, 1)

        return z_hat, delta, q

    @staticmethod
    def measurement_jacobian(
        delta: np.ndarray,
        q: float,
        r_hat: float | None = None,
    ) -> np.ndarray:
        """
        Jacobian H_j = dh/dmu_j of the range/bearing measurement model.
        """
        delta = np.asarray(delta, dtype=float).reshape(2, 1)

        if np.isclose(q, 0.0):
            raise ValueError("Measurement Jacobian is undefined for q = 0.")

        if r_hat is None:
            r_hat = np.sqrt(q)

        dx = float(delta[0, 0])
        dy = float(delta[1, 0])

        return np.array(
            [
                [r_hat * dx / q, r_hat * dy / q],
                [-dy / q, dx / q],
            ],
            dtype=float,
        )

    @staticmethod
    def _innovation(z: DetectedFeature, z_hat: np.ndarray) -> np.ndarray:
        """Compute z - z_hat and wrap the bearing residual."""
        innovation = np.array(
            [
                z.r - z_hat[0, 0],
                z.phi.rad - z_hat[1, 0],
            ],
            dtype=float,
        ).reshape(2, 1)

        innovation[1, 0] = Angle(innovation[1, 0]).clip().rad

        return innovation

    # ------------------------------------------------------------------
    # Likelihood / data association
    # ------------------------------------------------------------------

    def measurement_likelihood(
        self,
        z: DetectedFeature,
        landmark: LandmarkEstimate,
        pose: Pose,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Compute p(z | landmark, pose).

        Returns
        -------
        likelihood:
            Gaussian likelihood.
        z_hat:
            Predicted measurement.
        innovation_cov:
            Q_j / Psi_j, i.e. H Sigma H^T + Q.
        """
        z_hat, delta, q = self.predict_measurement(
            pose,
            landmark.mu,
        )

        r_hat = np.sqrt(q)
        h = self.measurement_jacobian(
            delta,
            q,
            r_hat,
        )

        innovation_cov = (
            h @ landmark.sigma @ h.T
            + self.q_mat
        )

        innovation_cov = (
            innovation_cov + innovation_cov.T
        ) / 2.0

        innovation = self._innovation(z, z_hat)

        try:
            solved = np.linalg.solve(
                innovation_cov,
                innovation,
            )

            mahalanobis_sq = float(
                (innovation.T @ solved).item()
            )

            sign, logdet = np.linalg.slogdet(
                innovation_cov
            )

            sign = float(np.asarray(sign).item())
            logdet = float(np.asarray(logdet).item())

            if sign <= 0:
                return 0.0, z_hat, innovation_cov

            log_likelihood = (
                -0.5 * mahalanobis_sq
                -0.5 * logdet
                -np.log(2.0 * np.pi)
            )

            likelihood = float(
                np.exp(np.clip(log_likelihood, -745.0, 700.0))
            )

        except np.linalg.LinAlgError:
            likelihood = 0.0

        return likelihood, z_hat, innovation_cov

    def associate(
        self,
        z: DetectedFeature,
        particle: Particle,
    ) -> tuple[int | None, float, dict]:
        """
        Associate one observation with the maximum-likelihood landmark.

        Returns
        -------
        correspondence:
            Landmark index, or None when the new-landmark hypothesis wins.
        weight:
            Maximum likelihood used for the particle weight.
        diagnostics:
            Likelihoods and predicted measurements for later experiments.
        """
        likelihoods = {}
        predictions = {}
        innovation_covariances = {}

        for idx, landmark in particle.landmarks.items():
            likelihood, z_hat, psi = self.measurement_likelihood(
                z,
                landmark,
                particle.pose,
            )

            likelihoods[idx] = likelihood
            predictions[idx] = z_hat
            innovation_covariances[idx] = psi

        new_landmark_idx = (
            max(particle.landmarks.keys(), default=-1) + 1
        )

        likelihoods[new_landmark_idx] = self.p0

        best_idx = max(
            likelihoods,
            key=likelihoods.get,
        )

        diagnostics = {
            "likelihoods": likelihoods,
            "predictions": predictions,
            "innovation_covariances": innovation_covariances,
            "new_landmark_index": new_landmark_idx,
        }

        if best_idx == new_landmark_idx:
            return None, self.p0, diagnostics

        return best_idx, likelihoods[best_idx], diagnostics

    # ------------------------------------------------------------------
    # Landmark initialization
    # ------------------------------------------------------------------

    @staticmethod
    def initialize_landmark(
        pose: Pose,
        z: DetectedFeature,
        q_mat: np.ndarray,
    ) -> LandmarkEstimate:
        """
        Initialize a new landmark from a range/bearing measurement.

        m_x = x + r cos(theta + phi)
        m_y = y + r sin(theta + phi)

        The covariance is obtained by transforming measurement noise
        into Cartesian landmark coordinates.
        """
        alpha = pose.th + z.phi

        mu = np.array(
            [
                pose.x + z.r * alpha.cos,
                pose.y + z.r * alpha.sin,
            ],
            dtype=float,
        ).reshape(2, 1)

        # Jacobian of Cartesian landmark position with respect to
        # measurement [r, phi].
        g_z = np.array(
            [
                [alpha.cos, -z.r * alpha.sin],
                [alpha.sin,  z.r * alpha.cos],
            ],
            dtype=float,
        )

        sigma = g_z @ q_mat @ g_z.T
        sigma = (sigma + sigma.T) / 2.0

        return LandmarkEstimate(
            mu=mu,
            sigma=sigma,
            seen_count=1,
        )

    # ------------------------------------------------------------------
    # Landmark EKF update
    # ------------------------------------------------------------------

    def update_landmark(
        self,
        landmark: LandmarkEstimate,
        pose: Pose,
        z: DetectedFeature,
    ) -> dict:
        """
        Perform the EKF update for one landmark.

        Returns diagnostics useful for notebook experiments.
        """
        z_hat, delta, q = self.predict_measurement(
            pose,
            landmark.mu,
        )

        r_hat = np.sqrt(q)

        h = self.measurement_jacobian(
            delta,
            q,
            r_hat,
        )

        psi = (
            h @ landmark.sigma @ h.T
            + self.q_mat
        )

        psi = (psi + psi.T) / 2.0

        innovation = self._innovation(z, z_hat)

        try:
            kalman_gain = (
                landmark.sigma
                @ h.T
                @ np.linalg.inv(psi)
            )
        except np.linalg.LinAlgError:
            kalman_gain = (
                landmark.sigma
                @ h.T
                @ np.linalg.pinv(psi)
            )

        landmark.mu = (
            landmark.mu
            + kalman_gain @ innovation
        )

        identity = np.eye(2)

        landmark.sigma = (
            (identity - kalman_gain @ h)
            @ landmark.sigma
        )

        landmark.sigma = (
            landmark.sigma
            + landmark.sigma.T
        ) / 2.0

        landmark.seen_count += 1

        return {
            "z_hat": z_hat,
            "innovation": innovation,
            "H": h,
            "Psi": psi,
            "K": kalman_gain,
        }

    # ------------------------------------------------------------------
    # One FastSLAM particle update
    # ------------------------------------------------------------------

    def update_particle(
        self,
        particle: Particle,
        motion,
        detected_features: list[DetectedFeature] | None,
        motion_noise: bool = True,
    ) -> dict:
        """
        Apply one FastSLAM update to a single particle.
        """
        particle.pose = self._sample_motion(
            particle.pose,
            motion,
            motion_noise=motion_noise,
        )

        particle.associations = []

        diagnostics = {
            "associations": [],
            "observation_diagnostics": [],
        }

        if detected_features is None:
            particle.weight = 1.0
            return diagnostics

        if len(detected_features) == 0:
            particle.weight = 1.0
            return diagnostics

        log_weight = 0.0

        for z in detected_features:

            correspondence, likelihood, association_diag = self.associate(
                z,
                particle,
            )

            diagnostics["observation_diagnostics"].append(
                association_diag
            )

            # The algorithm from the reference pseudocode uses the
            # maximum likelihood as the particle importance factor.
            # Use a small floor to avoid log(0).
            log_weight += np.log(
                max(likelihood, np.finfo(float).tiny)
            )

            if correspondence is None:
                landmark_idx = max(
                    particle.landmarks.keys(),
                    default=-1,
                ) + 1

                particle.landmarks[landmark_idx] = (
                    self.initialize_landmark(
                        particle.pose,
                        z,
                        self.q_mat,
                    )
                )

                particle.associations.append(landmark_idx)

                diagnostics["associations"].append(
                    {
                        "estimated": landmark_idx,
                        "is_new": True,
                        "likelihood": likelihood,
                    }
                )

            else:
                landmark = particle.landmarks[correspondence]

                update_diag = self.update_landmark(
                    landmark,
                    particle.pose,
                    z,
                )

                particle.associations.append(correspondence)

                diagnostics["associations"].append(
                    {
                        "estimated": correspondence,
                        "is_new": False,
                        "likelihood": likelihood,
                        "update": update_diag,
                    }
                )

        particle.weight = float(
            np.exp(
                np.clip(log_weight, -745.0, 700.0)
            )
        )

        return diagnostics

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def normalized_weights(self) -> np.ndarray:
        """Return normalized particle weights."""
        weights = np.array(
            [p.weight for p in self.particles],
            dtype=float,
        )

        if len(weights) == 0:
            raise RuntimeError(
                "FastSLAM has not been initialized."
            )

        if (
            not np.all(np.isfinite(weights))
            or np.sum(weights) <= 0.0
        ):
            return np.ones(len(weights)) / len(weights)

        return weights / np.sum(weights)

    def effective_particle_number(self) -> float:
        """N_eff = 1 / sum(w_k^2)."""
        weights = self.normalized_weights()
        return float(1.0 / np.sum(weights**2))

    def resample(self) -> np.ndarray:
        """
        Resample particles according to their normalized weights.

        Returns the selected indices, useful for notebook diagnostics.
        """
        weights = self.normalized_weights()

        indices = np.random.choice(
            len(self.particles),
            size=self.n_particles,
            replace=True,
            p=weights,
        )

        old_particles = self.particles

        self.particles = [
            deepcopy(old_particles[idx])
            for idx in indices
        ]

        uniform_weight = 1.0 / self.n_particles

        for particle in self.particles:
            particle.weight = uniform_weight

        return indices

    # ------------------------------------------------------------------
    # Full FastSLAM update
    # ------------------------------------------------------------------

    def update(
        self,
        motion,
        detected_features: list[DetectedFeature] | None,
        motion_noise: bool = True,
        resample: bool = True,
    ) -> dict:
        """
        Execute one complete FastSLAM 1.0 iteration.

        Returns diagnostics for experiments and plots.
        """
        if len(self.particles) == 0:
            raise RuntimeError(
                "Call initialize() before update()."
            )

        diagnostics = {
            "particle_diagnostics": [],
            "weights_before_resampling": None,
            "normalized_weights": None,
            "effective_particle_number": None,
            "resampled_indices": None,
        }

        for particle in self.particles:
            particle_diag = self.update_particle(
                particle,
                motion,
                detected_features,
                motion_noise=motion_noise,
            )

            diagnostics["particle_diagnostics"].append(
                particle_diag
            )

        weights = np.array(
            [p.weight for p in self.particles],
            dtype=float,
        )

        diagnostics["weights_before_resampling"] = weights.copy()

        if (
            np.any(np.isfinite(weights))
            and np.sum(np.where(np.isfinite(weights), weights, 0.0)) > 0
        ):
            normalized = self.normalized_weights()
        else:
            normalized = np.ones(
                self.n_particles
            ) / self.n_particles

        diagnostics["normalized_weights"] = normalized.copy()
        diagnostics["effective_particle_number"] = float(
            1.0 / np.sum(normalized**2)
        )

        if resample:
            diagnostics["resampled_indices"] = self.resample()

        return diagnostics

    # ------------------------------------------------------------------
    # Convenience accessors for experiments
    # ------------------------------------------------------------------

    @property
    def best_particle(self) -> Particle:
        """Particle with maximum weight before/after resampling."""
        if len(self.particles) == 0:
            raise RuntimeError(
                "FastSLAM has not been initialized."
            )

        return max(
            self.particles,
            key=lambda p: p.weight,
        )

    @property
    def poses(self) -> list[Pose]:
        """Current particle poses."""
        return [p.pose.copy() for p in self.particles]

    def get_landmark_estimates(self, particle_index: int = 0) -> dict:
        """Return the estimated landmarks of one particle."""
        particle = self.particles[particle_index]
        return {
            idx: {
                "mu": landmark.mu.copy(),
                "sigma": landmark.sigma.copy(),
                "seen_count": landmark.seen_count,
            }
            for idx, landmark in particle.landmarks.items()
        }
