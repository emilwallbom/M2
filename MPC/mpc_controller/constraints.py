"""State and input constraints, independent of ROS.

TODO: define speed, steering, acceleration and rate limits, and any track or
obstacle constraints. Distinguish physical inputs from normalized actuator
efforts; the vehicle's throttle command is not a target speed in m/s.
"""


def build_constraints(states, controls):
    """Construct constraints for the selected solver (to be implemented)."""
    raise NotImplementedError("Implement the MPC constraints")
