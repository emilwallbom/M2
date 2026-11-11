"""MPC optimization entry point, independent of ROS.

TODO: select a solver, combine dynamics/costs/constraints, implement warm
starting, and define a result including solver status and the first input.
Handle infeasibility and solve-time limits before connecting actuator output.
"""


class MPCController:
    """Own the optimization problem and persistent solver state."""

    def solve(self, state, reference):
        """Return a control solution for the current state and reference."""
        raise NotImplementedError("Implement the MPC optimization problem")
