import numpy as np
from maps import DistanceField


# ============================================================
# Measurement endpoint
# ============================================================

def measurement_endpoint(
    robot_pose,
    measurement: float,
    beam_angle: float
) -> tuple[float, float]:
    """
    Calculate the endpoint of a range measurement in world
    coordinates.

    The sensor is assumed to be located at the robot position.
    """

    theta = robot_pose.th.rad + beam_angle

    x = (
        robot_pose.x
        + measurement * np.cos(theta)
    )

    y = (
        robot_pose.y
        + measurement * np.sin(theta)
    )

    return x, y


# ============================================================
# Likelihood Field probability
# ============================================================

def likelihood_field_probability(
    distance: float,
    z_max: float,
    z_hit: float,
    z_rand: float,
    sigma_hit: float
) -> float:
    """
    Calculate the probability of a measurement according to
    the Likelihood Field model.

    The hit component is modeled as a Gaussian centered at
    zero distance from the nearest obstacle.

    The random component is uniformly distributed over the
    sensor range.
    """

    if sigma_hit <= 0:
        raise ValueError(
            "sigma_hit must be greater than zero."
        )

    if z_max <= 0:
        raise ValueError(
            "z_max must be greater than zero."
        )

    # Gaussian centered at distance = 0
    p_hit = (
        1.0
        / (np.sqrt(2.0 * np.pi) * sigma_hit)
        * np.exp(
            -0.5
            * (distance / sigma_hit) ** 2
        )
    )

    # Uniform random measurement
    p_rand = 1.0 / z_max

    probability = (
        z_hit * p_hit
        + z_rand * p_rand
    )

    return probability


# ============================================================
# Likelihood Field Range Finder Model
# ============================================================

def likelihood_field_range_finder_model(
    z,
    x,
    distance_field: DistanceField,
    z_max: float,
    z_hit: float,
    z_rand: float,
    sigma_hit: float,
    beam_angles
) -> float:
    """
    Calculate the likelihood of a complete laser scan
    according to the Likelihood Field Range Finder Model.

    Parameters
    ----------
    z:
        Laser measurements.

    x:
        Robot pose.

    distance_field:
        Precomputed distance field of the map.

    z_max:
        Maximum sensor range.

    z_hit:
        Weight of the hit component.

    z_rand:
        Weight of the random component.

    sigma_hit:
        Standard deviation of the hit model.

    beam_angles:
        Relative angle of each laser beam.

    Returns
    -------
    float
        Likelihood of the complete scan.
    """

    if len(z) != len(beam_angles):
        raise ValueError(
            "z and beam_angles must have the same length."
        )

    q = 1.0

    for k in range(len(z)):

        if z[k] >= z_max:
            continue

        x_z, y_z = measurement_endpoint(
            robot_pose=x,
            measurement=z[k],
            beam_angle=beam_angles[k]
        )

        distance = distance_field.distance_at(
            x_z,
            y_z
        )

        probability = likelihood_field_probability(
            distance=distance,
            z_max=z_max,
            z_hit=z_hit,
            z_rand=z_rand,
            sigma_hit=sigma_hit
        )

        q *= probability   # Combine beam probabilities


    return q