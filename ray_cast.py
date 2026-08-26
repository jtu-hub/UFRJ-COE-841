import numpy as np
from maps import ObstacleMap, Segment


def ray_segment_intersection(
    ray_origin: tuple[float, float],
    ray_angle: float,
    segment: Segment
) -> float | None:

    dx = np.cos(ray_angle)
    dy = np.sin(ray_angle)

    x, y = ray_origin
    x1, y1 = segment.p1

    sx = segment.p2[0] - x1
    sy = segment.p2[1] - y1
    denominator = dx * sy - dy * sx

    if np.isclose(denominator, 0.0):
        return None

    qx = x1 - x
    qy = y1 - y

    t = (qx * sy - qy * sx) / denominator
    u = (qx * dy - qy * dx) / denominator

    if t < 0:
        return None

    if u < 0 or u > 1:
        return None

    return t


def ray_cast(
    robot_pos: tuple[float, float],
    ray_angle: float,
    obstacle_map: ObstacleMap,
    max_range: float = np.inf
) -> float:
    
    min_distance = max_range

    for segment in obstacle_map.segments:

        distance = ray_segment_intersection(
            robot_pos,
            ray_angle,
            segment
        )

        if distance is not None and distance < min_distance:
            min_distance = distance

    return min_distance