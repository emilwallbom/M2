"""Small, ROS-independent trajectory sampling helper."""


def sample_reference(velocities, steering_angles, sample_time, elapsed):
    """Return the time-indexed velocity and steering reference."""
    index = max(0, int(elapsed / sample_time))
    if index >= len(velocities):
        return 0.0, 0.0, True
    steering = steering_angles[index] if index < len(steering_angles) else 0.0
    return float(velocities[index]), float(steering), False
