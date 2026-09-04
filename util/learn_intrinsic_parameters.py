import numpy as np

from geometry import Pose, Angle

from beam_range_finder import (
    p_hit,
    p_short,
    p_max,
    p_rand
)


# ============================================================
# Synthetic dataset generation
# ============================================================

def sample_short_measurement(
    z_hat: float,
    lambda_short: float,
    rng: np.random.Generator
) -> float:
    """
    Generate a sample from the truncated exponential
    distribution used by the short component.

    The distribution is defined over:

        0 <= z <= z_hat
    """

    if z_hat <= 0:
        return 0.0

    if lambda_short <= 0:
        raise ValueError(
            "lambda_short must be greater than zero."
        )

    u = rng.uniform(0.0, 1.0)

    return (
        -np.log(
            1.0
            - u * (
                1.0
                - np.exp(-lambda_short * z_hat)
            )
        )
        / lambda_short
    )

def generate_random_poses(
    obstacle_map,
    n_poses,
    min_distance=0.2,
    seed=42,
    max_attempts=10000
):
    rng = np.random.default_rng(seed)

    poses = []

    x_min, x_max = obstacle_map.x_lims
    y_min, y_max = obstacle_map.y_lims
    attempts = 0
    while len(poses) < n_poses:
        if attempts >= max_attempts:
            raise RuntimeError(
                "Unable to generate the requested number of valid poses."
            )
        attempts += 1

        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)

        if not obstacle_map.is_valid_position(
            x,
            y,
            min_distance
        ):
            continue

        theta = rng.uniform(
            -np.pi,
            np.pi
        )

        poses.append(
            Pose(
                x,
                y,
                Angle(theta)
            )
        )

    return poses

def generate_synthetic_dataset(
    poses,
    obstacle_map,
    beam_angles,
    z_max=10.0,
    z_hit=0.7,
    z_short=0.1,
    z_max_weight=0.1,
    z_rand=0.1,
    sigma_hit=0.2,
    lambda_short=1.0,
    seed=None
):
    """
    Generate synthetic range measurements.

    For each pose and each beam:

        1 - pose + map
        2 - ray casting
        3 - z_hat
        4 - choose component
        5 - generate z

    Returns
    -------
    scans:
        Real/simulated sensor measurements.

    expected:
        Expected measurements obtained by ray casting.

    components:
        Component used to generate each measurement.
        This is useful for validating the learning algorithm.
    """

    rng = np.random.default_rng(seed)

    weights = np.array([
        z_hit,
        z_short,
        z_max_weight,
        z_rand
    ], dtype=float)

    if np.any(weights < 0):
        raise ValueError(
            "Mixture weights must be non-negative."
        )

    if weights.sum() <= 0:
        raise ValueError(
            "At least one mixture weight must be positive."
        )

    # Normalize weights in case they do not sum exactly to 1.
    weights /= weights.sum()

    component_names = [
        "hit",
        "short",
        "max",
        "rand"
    ]

    scans = []
    expected = []
    components = []

    for pose in poses:

        scan = []
        scan_expected = []
        scan_components = []

        for beam_angle in beam_angles:

            # Expected measurement
            z_hat = obstacle_map.ray_cast(
                Pose(pose.x, pose.y, pose.th.rad + beam_angle),
                max_range=z_max
            )

            scan_expected.append(z_hat)

            # Select the component that generates the reading
            component = rng.choice(
                component_names,
                p=weights
            )

            # Generate measurement
            if component == "hit":

                z = z_hat + rng.normal(
                    0.0,
                    sigma_hit
                )

                z = np.clip(
                    z,
                    0.0,
                    z_max
                )

            elif component == "short":

                z = sample_short_measurement(
                    z_hat=z_hat,
                    lambda_short=lambda_short,
                    rng=rng
                )

            elif component == "max":

                z = z_max

            else:  # rand

                z = rng.uniform(
                    0.0,
                    z_max
                )

            scan.append(z)
            scan_components.append(component)

        scans.append(scan)
        expected.append(scan_expected)
        components.append(scan_components)

    return scans, expected, components


# E-step
def beam_responsibilities(
    z,
    z_hat,
    z_max,
    z_hit,
    z_short,
    z_max_weight,
    z_rand,
    sigma_hit,
    lambda_short
):
    """
    E-step.

    Calculate the responsibility of each mixture component
    for the measurement z.

    Returns
    -------
    (e_hit, e_short, e_max, e_rand)

    where each value represents the responsibility of the
    corresponding component.
    """

    # Weighted likelihood of each component
    likelihood_hit = (
        z_hit
        * p_hit(
            z,
            z_hat,
            sigma_hit
        )
    )

    likelihood_short = (
        z_short
        * p_short(
            z,
            z_hat,
            lambda_short
        )
    )

    likelihood_max = (
        z_max_weight
        * p_max(
            z,
            z_max
        )
    )

    likelihood_rand = (
        z_rand
        * p_rand(
            z,
            z_max
        )
    )

    # Total likelihood
    total = (
        likelihood_hit
        + likelihood_short
        + likelihood_max
        + likelihood_rand
    )

    # Avoid division by zero.
    if total <= 0 or not np.isfinite(total):

        return (
            0.0,
            0.0,
            0.0,
            0.0
        )

    # Responsibilities
    return (
        likelihood_hit / total,
        likelihood_short / total,
        likelihood_max / total,
        likelihood_rand / total
    )


# ============================================================
# M-step
# ============================================================

def update_parameters(
    z_values,
    z_hat_values,
    responsibilities,
    old_parameters
):
    """
    M-step.

    Update:

        z_hit
        z_short
        z_max_weight
        z_rand
        sigma_hit
        lambda_short

    using the responsibilities obtained in the E-step.
    """

    n = len(z_values)
    if n == 0:
        raise ValueError("Dataset vazio.")

    # Effective number of observations assigned to each model
    hit_weight = sum(
        r[0] for r in responsibilities
    )

    short_weight = sum(
        r[1] for r in responsibilities
    )

    max_weight = sum(
        r[2] for r in responsibilities
    )

    rand_weight = sum(
        r[3] for r in responsibilities
    )

    # Mixture weights
    z_hit = hit_weight / n
    z_short = short_weight / n
    z_max_weight = max_weight / n
    z_rand = rand_weight / n

    # sigma_hit
    if hit_weight > 1e-12:

        weighted_squared_error = sum(
            r[0] * (z - z_hat) ** 2
            for z, z_hat, r in zip(
                z_values,
                z_hat_values,
                responsibilities
            )
        )

        sigma_hit = np.sqrt(
            weighted_squared_error / hit_weight
        )

        # Avoid a zero standard deviation, which would make
        # the Gaussian problematic in the next iteration.
        sigma_hit = max(
            sigma_hit,
            1e-6
        )

    else:
        # If no measurement is attributed to hit,
        # preserve the previous value.
        sigma_hit = old_parameters["sigma_hit"]

    # lambda_short
    weighted_short_distance = sum(
        r[1] * z
        for z, r in zip(
            z_values,
            responsibilities
        )
    )

    if weighted_short_distance > 1e-12:

        lambda_short = (
            short_weight
            / weighted_short_distance
        )

        lambda_short = max(
            lambda_short,
            1e-6
        )

    else:

        # If no measurement is attributed to short,
        # preserve the previous value.
        lambda_short = old_parameters["lambda_short"]

    return {
        "z_hit": z_hit,
        "z_short": z_short,
        "z_max_weight": z_max_weight,
        "z_rand": z_rand,
        "sigma_hit": sigma_hit,
        "lambda_short": lambda_short
    }


# EM algorithm
def learn_intrinsic_parameters(
    scans,
    poses,
    obstacle_map,
    beam_angles,
    z_max=10.0,
    z_hit=0.7,
    z_short=0.1,
    z_max_weight=0.1,
    z_rand=0.1,
    sigma_hit=0.2,
    lambda_short=1.0,
    max_iterations=100,
    tolerance=1e-5
):
    """
    Learn the intrinsic parameters of the Beam Range Finder
    model using an Expectation-Maximization procedure.

    Parameters
    ----------
    scans:
        List of sensor scans.

    poses:
        Pose associated with each scan.

    obstacle_map:
        Map used to calculate expected measurements.

    beam_angles:
        Relative angle of each sensor beam.

    z_max:
        Maximum sensor range.

    z_hit, z_short, z_max_weight, z_rand:
        Initial mixture weights.

    sigma_hit:
        Initial standard deviation of the hit model.

    lambda_short:
        Initial parameter of the short model.

    max_iterations:
        Maximum number of EM iterations.

    tolerance:
        Convergence threshold.

    Returns
    -------
    parameters:
        Dictionary containing the learned parameters.
    """

    # Basic validation
    if len(scans) != len(poses):
        raise ValueError(
            "scans and poses must have the same length."
        )

    if len(scans) == 0:
        raise ValueError(
            "The dataset cannot be empty."
        )

    if len(beam_angles) == 0:
        raise ValueError(
            "beam_angles cannot be empty."
        )

    if sigma_hit <= 0:
        raise ValueError(
            "sigma_hit must be greater than zero."
        )

    if lambda_short <= 0:
        raise ValueError(
            "lambda_short must be greater than zero."
        )

    # Initial parameters
    parameters = {
        "z_hit": z_hit,
        "z_short": z_short,
        "z_max_weight": z_max_weight,
        "z_rand": z_rand,
        "sigma_hit": sigma_hit,
        "lambda_short": lambda_short
    }

    # EM iterations
    for iteration in range(max_iterations):

        # E-STEP
        z_values = []
        z_hat_values = []
        responsibilities = []

        for scan, pose in zip(scans, poses):

            if len(scan) != len(beam_angles):
                raise ValueError(
                    "Each scan must have a measurement "
                    "for each beam_angle."
                )

            for k, z in enumerate(scan):
                # Expected measurement from ray casting
                z_hat = obstacle_map.ray_cast(
                    Pose(pose.x, pose.y, pose.th.rad + beam_angles[k]),
                    max_range=z_max
                )

                # Responsibility calculation
                e = beam_responsibilities(
                    z=z,
                    z_hat=z_hat,
                    z_max=z_max,

                    z_hit=parameters["z_hit"],
                    z_short=parameters["z_short"],
                    z_max_weight=parameters["z_max_weight"],
                    z_rand=parameters["z_rand"],

                    sigma_hit=parameters["sigma_hit"],
                    lambda_short=parameters["lambda_short"]
                )

                z_values.append(z)
                z_hat_values.append(z_hat)
                responsibilities.append(e)

        # M-STEP
        new_parameters = update_parameters(
            z_values=z_values,
            z_hat_values=z_hat_values,
            responsibilities=responsibilities,
            old_parameters=parameters
        )

        # Check convergence
        old_values = np.array([
            parameters["z_hit"],
            parameters["z_short"],
            parameters["z_max_weight"],
            parameters["z_rand"],
            parameters["sigma_hit"],
            parameters["lambda_short"]
        ])

        new_values = np.array([
            new_parameters["z_hit"],
            new_parameters["z_short"],
            new_parameters["z_max_weight"],
            new_parameters["z_rand"],
            new_parameters["sigma_hit"],
            new_parameters["lambda_short"]
        ])

        difference = np.max(
            np.abs(new_values - old_values)
        )

        parameters = new_parameters

        if difference < tolerance:

            print(
                f"Convergence reached at iteration "
                f"{iteration + 1}."
            )

            break

    else:

        print(
            f"Maximum of {max_iterations} iterations "
            f"reached."
        )

    return parameters