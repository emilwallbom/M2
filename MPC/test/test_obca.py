import numpy as np

from mpc_controller.controller import TrajectoryOptimizer
from mpc_controller.model import footprint_halfspaces, rotation, translation
from mpc_controller.obstacles import axis_aligned_rectangle, trajectory_clearance
from mpc_controller.reference import TrajectoryGuess


def test_obca_duals_certify_full_vehicle_clearance():
    obstacle = axis_aligned_rectangle(1.0, 0.0, 0.2, 0.2)
    optimizer = TrajectoryOptimizer(obstacles=(obstacle,))
    state = np.zeros(4)
    guess = TrajectoryGuess(
        states=np.vstack((state, state)), controls=np.zeros((1, 2)), sample_time=0.1
    )
    result = optimizer.solve(state, state, guess)
    assert result.success, result.status

    G, g = footprint_halfspaces(optimizer.params)
    lambdas = result.obstacle_lambdas[0]
    mus = result.vehicle_mus[0]
    for index, predicted_state in enumerate(result.states):
        world_normal = obstacle.A.T @ lambdas[index]
        separation = -g @ mus[index] + (
            obstacle.A @ translation(predicted_state, optimizer.params) - obstacle.b
        ) @ lambdas[index]
        coupling = G.T @ mus[index] + rotation(predicted_state).T @ world_normal
        assert separation >= optimizer.config.minimum_clearance - 1e-6
        assert np.linalg.norm(coupling) <= 1e-6
        assert np.linalg.norm(world_normal) <= 1.0 + 1e-6

    clearance = trajectory_clearance(result.states, (obstacle,), optimizer.params)
    assert np.min(clearance) >= optimizer.config.minimum_clearance - 1e-5
