from low_level_controller.trajectory_sampling import sample_reference


def test_samples_reference_and_stops_after_trajectory():
    velocities = [0.0, 0.2, 0.3, 0.0]
    steering = [0.1, 0.2, 0.0]

    assert sample_reference(velocities, steering, 0.1, 0.15) == (
        0.2, 0.2, False
    )
    assert sample_reference(velocities, steering, 0.1, 0.35) == (
        0.0, 0.0, False
    )
    assert sample_reference(velocities, steering, 0.1, 0.45) == (
        0.0, 0.0, True
    )
