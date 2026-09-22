from __future__ import annotations

import numpy as np

from .geometry import Angle, Pose
from .control import VelocityControl
from .motion_models import CircularMotion
from .camera import DetectedFeature
from .fastslam import FastSLAM, Particle, LandmarkEstimate


class FastSLAM2(FastSLAM):
    """
    FastSLAM 2.0 with maximum-likelihood data association.

    The only conceptual difference with respect to FastSLAM 1.0
    (:class:`.fastslam.FastSLAM`) is the proposal distribution used to
    sample each particle's pose. FastSLAM 1.0 samples purely from the
    motion model and lets the measurement only re-weight the particle.
    FastSLAM 2.0 instead folds the current observations into the
    proposal itself: the pose is sampled from a Gaussian that combines
    the motion-model prediction with every observed, previously-seen
    landmark, so particles concentrate where the measurements agree
    with the map. This yields a lower-variance estimator and typically
    needs far fewer particles for the same accuracy.

    Everything else (landmark initialization, the per-landmark EKF
    update, resampling, diagnostics accessors, ...) is inherited
    unchanged from :class:`.fastslam.FastSLAM`.
    """

    # Extra Jacobian needed for the improved proposal
    @staticmethod
    def pose_jacobian(
        delta: np.ndarray,
        q: float,
        r_hat: float | None = None,
    ) -> np.ndarray:
        """
        Jacobian G_x = dh/dx of the range/bearing measurement model
        with respect to the robot pose (x, y, theta).
        """
        delta = np.asarray(delta, dtype=float).reshape(2, 1)

        if np.isclose(q, 0.0):
            raise ValueError("Pose Jacobian is undefined for q = 0.")

        if r_hat is None:
            r_hat = np.sqrt(q)

        dx = float(delta[0, 0])
        dy = float(delta[1, 0])

        return np.array(
            [
                [-r_hat * dx / q, -r_hat * dy / q, 0.0],
                [dy / q, -dx / q, -1.0],
            ],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Motion-noise covariance in pose space
    # ------------------------------------------------------------------

    @staticmethod
    def _as_velocity_control(motion) -> VelocityControl:
        if isinstance(motion, CircularMotion):
            motion = motion.toVelocityControl()

        if not isinstance(motion, VelocityControl):
            raise TypeError(
                "FastSLAM2 expects VelocityControl or CircularMotion motion."
            )

        return motion

    @staticmethod
    def _motion_covariance(
        pose: Pose,
        motion: VelocityControl,
        std_mot_noise: tuple[float, float],
    ) -> np.ndarray:
        """
        R_t: covariance of the (x, y, theta) prediction induced by the
        control-space noise (std_v, std_w), obtained by propagating the
        control-noise covariance through the motion model's Jacobian.

        A tiny jitter is added on the diagonal so that R_t stays
        invertible even along straight-line segments (w = 0), where the
        analytic Jacobian assigns zero heading uncertainty.
        """
        v_mat = motion.v_mat(pose)

        m_mat = np.diag(
            [std_mot_noise[0] ** 2, std_mot_noise[1] ** 2]
        )

        r_t = v_mat @ m_mat @ v_mat.T
        r_t = (r_t + r_t.T) / 2.0
        r_t += np.eye(3) * 1e-9

        return r_t

    @staticmethod
    def _log_gaussian(innovation: np.ndarray, cov: np.ndarray) -> float:
        """log N(innovation; 0, cov), with a graceful singular fallback."""
        cov = (cov + cov.T) / 2.0

        try:
            solved = np.linalg.solve(cov, innovation)

            mahalanobis_sq = float((innovation.T @ solved).item())

            sign, logdet = np.linalg.slogdet(cov)
            sign = float(np.asarray(sign).item())
            logdet = float(np.asarray(logdet).item())

            if sign <= 0:
                return -np.inf

            return (
                -0.5 * mahalanobis_sq
                - 0.5 * logdet
                - np.log(2.0 * np.pi)
            )

        except np.linalg.LinAlgError:
            return -np.inf

    # ------------------------------------------------------------------
    # One FastSLAM 2.0 particle update
    # ------------------------------------------------------------------

    def update_particle(
        self,
        particle: Particle,
        motion,
        detected_features: list[DetectedFeature] | None,
        motion_noise: bool = True,
        std_mot_noise: tuple[float, float] = (0.2, 0.1),
    ) -> dict:
        """
        Apply one FastSLAM 2.0 update to a single particle.
        """
        prior_pose = particle.pose
        vel_motion = self._as_velocity_control(motion)

        diagnostics = {
            "associations": [],
            "observation_diagnostics": [],
        }

        if not detected_features:
            particle.pose = self._sample_motion(
                prior_pose,
                vel_motion,
                motion_noise=motion_noise,
                std_mot_noise=std_mot_noise,
            )
            particle.associations = []
            particle.weight = 1.0
            return diagnostics

        # Linearization point: the noise-free motion prediction.
        mean_pose = self._sample_motion(
            prior_pose,
            vel_motion,
            motion_noise=False,
        )
        mean_vec = mean_pose.as_array

        r_t = self._motion_covariance(
            prior_pose,
            vel_motion,
            std_mot_noise,
        )

        # ------------------------------------------------------------
        # Data association at the linearization point.
        # ------------------------------------------------------------
        probe_particle = Particle(pose=mean_pose, landmarks=particle.landmarks)

        known = []
        new_features = []

        for z in detected_features:
            correspondence, likelihood, assoc_diag = self.associate(
                z,
                probe_particle,
            )

            diagnostics["observation_diagnostics"].append(assoc_diag)

            if correspondence is None:
                new_features.append(z)
                continue

            landmark = particle.landmarks[correspondence]

            z_hat, delta, q = self.predict_measurement(
                mean_pose,
                landmark.mu,
            )

            r_hat = np.sqrt(q)

            g_x = self.pose_jacobian(delta, q, r_hat)
            g_m = self.measurement_jacobian(delta, q, r_hat)

            q_tilde = g_m @ landmark.sigma @ g_m.T + self.q_mat
            q_tilde = (q_tilde + q_tilde.T) / 2.0

            innovation = self._innovation(z, z_hat)

            known.append(
                {
                    "idx": correspondence,
                    "landmark": landmark,
                    "z": z,
                    "z_hat": z_hat,
                    "g_x": g_x,
                    "q_tilde": q_tilde,
                    "innovation": innovation,
                    "likelihood": likelihood,
                }
            )

        # ------------------------------------------------------------
        # Improved proposal: fuse the motion prediction with every
        # observed, previously-seen landmark (information form).
        # ------------------------------------------------------------
        log_weight = 0.0

        if known:
            try:
                r_inv = np.linalg.inv(r_t)
            except np.linalg.LinAlgError:
                r_inv = np.linalg.pinv(r_t)

            omega = r_inv.copy()
            xi = r_inv @ mean_vec

            for obs in known:
                try:
                    q_inv = np.linalg.inv(obs["q_tilde"])
                except np.linalg.LinAlgError:
                    q_inv = np.linalg.pinv(obs["q_tilde"])

                g_x = obs["g_x"]

                omega += g_x.T @ q_inv @ g_x
                xi += g_x.T @ q_inv @ (
                    obs["innovation"] + g_x @ mean_vec
                )

                l_mat = g_x @ r_t @ g_x.T + obs["q_tilde"]

                log_weight += self._log_gaussian(
                    obs["innovation"],
                    l_mat,
                )

            try:
                sigma_prop = np.linalg.inv(omega)
            except np.linalg.LinAlgError:
                sigma_prop = np.linalg.pinv(omega)

            sigma_prop = (sigma_prop + sigma_prop.T) / 2.0
            mu_prop = (sigma_prop @ xi).flatten()

            sampled = np.random.multivariate_normal(mu_prop, sigma_prop)

            new_pose = Pose(sampled[0], sampled[1], Angle(sampled[2]))

        else:
            # No previously-seen landmark was re-observed: fall back to
            # sampling directly from the (noisy) motion model, exactly
            # as FastSLAM 1.0 does.
            new_pose = self._sample_motion(
                prior_pose,
                vel_motion,
                motion_noise=motion_noise,
                std_mot_noise=std_mot_noise,
            )

        particle.pose = new_pose
        particle.associations = []

        # ------------------------------------------------------------
        # EKF landmark updates / initializations at the sampled pose.
        # ------------------------------------------------------------
        for obs in known:
            update_diag = self.update_landmark(
                obs["landmark"],
                new_pose,
                obs["z"],
            )

            particle.associations.append(obs["idx"])

            diagnostics["associations"].append(
                {
                    "estimated": obs["idx"],
                    "is_new": False,
                    "likelihood": obs["likelihood"],
                    "update": update_diag,
                }
            )

        for z in new_features:
            landmark_idx = max(
                particle.landmarks.keys(),
                default=-1,
            ) + 1

            particle.landmarks[landmark_idx] = self.initialize_landmark(
                new_pose,
                z,
                self.q_mat,
            )

            particle.associations.append(landmark_idx)

            log_weight += np.log(max(self.p0, np.finfo(float).tiny))

            diagnostics["associations"].append(
                {
                    "estimated": landmark_idx,
                    "is_new": True,
                    "likelihood": self.p0,
                }
            )

        particle.weight = float(
            np.exp(np.clip(log_weight, -745.0, 700.0))
        )

        return diagnostics
