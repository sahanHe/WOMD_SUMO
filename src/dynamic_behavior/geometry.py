import math
import numpy as np

def wrap_radians(theta):
    """Wrap angle to the range [-pi, pi]"""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def degree2rad(degree):
    # Check if the input is a scalar
    is_scalar = np.isscalar(degree)
    if is_scalar:
        degree = np.array([degree])

    # Adjust the angle to the range [0, 360] with 0 on the right (east)
    adjusted_degree = (90 - degree) % 360

    # Convert to radians in the range [-pi, pi]
    radian = np.radians(adjusted_degree)
    radian[radian > np.pi] -= 2 * np.pi

    # If the input was a scalar, convert the output to a scalar
    if is_scalar:
        radian = radian.item()

    return radian


def rad2degree(theta):
    # Convert to degrees in the range [-180, 180] with 0 on the right
    theta_degrees = math.degrees(theta)

    # Adjust to the range [0, 360] with 0 at the top
    theta_degrees = (90.0 - theta_degrees) % 360.0

    return theta_degrees


def center_to_front_bumper_position(center: list[float], length:float, theta: float) -> list[float]:
    # Calculate the offset of the front bumper from the center
    offset_x = (length / 2) * math.cos(theta)
    offset_y = (length / 2) * math.sin(theta)

    # Add the offset to the center position to get the position of the front bumper
    front_bumper_position = list(center)
    front_bumper_position[0] = center[0] + offset_x
    front_bumper_position[1] = center[1] + offset_y
    return front_bumper_position


def front_bumper_to_center_position(front_bumper: list[float], length: float, theta: float) -> list[float]:
    offset_x = (length / 2) * math.cos(theta)
    offset_y = (length / 2) * math.sin(theta)

    # Add the offset to the front bumper position to get the position of the center position
    center_position = list(front_bumper)
    center_position[0] = front_bumper[0] - offset_x
    center_position[1] = front_bumper[1] - offset_y
    return center_position