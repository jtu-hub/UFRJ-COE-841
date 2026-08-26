from distributions import GaussianDistribution
from ray_cast import ray_cast
import numpy as np


def p_hit(z, z_hat, sigma_hit):
    error = z - z_hat

    return GaussianDistribution(
        sigma_hit ** 2
    ).prob(error)


def p_short(z, z_hat, lambda_short):
    if z < 0 or z > z_hat:
        return 0.0

    eta = 1.0 / (
        1.0 - np.exp(-lambda_short * z_hat)
    )

    return eta * lambda_short * np.exp(
        -lambda_short * z
    )

def p_max(z, z_max):
    if np.isclose(z, z_max):
        return 1.0

    return 0.0

def p_rand(z, z_max):
    if 0 <= z <= z_max:
        return 1.0 / z_max

    return 0.0


def beam_measurement_probability(
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
    p = (
        z_hit * p_hit(z, z_hat, sigma_hit)
        + z_short * p_short(z, z_hat, lambda_short)
        + z_max_weight * p_max(z, z_max)
        + z_rand * p_rand(z, z_max)
    )

    return p

def beam_range_finder_model(
    z,
    x,
    obstacle_map,
    z_max,
    z_hit,
    z_short,
    z_max_weight,
    z_rand,
    sigma_hit,
    lambda_short,
    beam_angles
):
    q = 1.0

    for k in range(len(z)):

        z_hat = ray_cast(
            (x.x, x.y),
            x.th.rad + beam_angles[k],
            obstacle_map,
            max_range=z_max
        )

        p = beam_measurement_probability(
            z=z[k],
            z_hat=z_hat,
            z_max=z_max,
            z_hit=z_hit,
            z_short=z_short,
            z_max_weight=z_max_weight,
            z_rand=z_rand,
            sigma_hit=sigma_hit,
            lambda_short=lambda_short
        )

        q *= p

    return q