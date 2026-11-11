"""Vehicle dynamics, independent of ROS.

TODO: define state/input vectors and units, vehicle parameters, discrete
dynamics, and any linearization needed by the chosen MPC formulation.
Use the platform's vehicle_parameters.yaml as the source of vehicle constants.
"""


def predict_next_state(state, control, dt):
    """Predict one time step using the vehicle model (to be implemented)."""
    raise NotImplementedError("Implement the discrete vehicle model")
