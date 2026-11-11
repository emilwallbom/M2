"""Objective terms, independent of ROS.

TODO: define tracking, control effort, control variation, and terminal costs;
choose and document units and weights after state/input definitions are fixed.
"""


def tracking_cost(states, controls, reference):
    """Evaluate or construct the MPC objective (to be implemented)."""
    raise NotImplementedError("Implement the MPC objective")
