"""Reference trajectory preparation, independent of ROS.

TODO: define a numeric reference representation, nearest-point selection,
path resampling over the prediction horizon, and a speed profile.
The ROS node must transform incoming paths into the state frame first.
"""


def build_reference(state, path, horizon, dt):
    """Build the horizon reference from a numeric path (to be implemented)."""
    raise NotImplementedError("Implement reference trajectory generation")
