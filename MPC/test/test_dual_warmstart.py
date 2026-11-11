import numpy as np
import pytest

from mpc_controller.controller import OptimizerConfig, TrajectoryOptimizer
from mpc_controller.dual_warmstart import compute_dual_warmstart
from mpc_controller.model import (
    VehicleParams, footprint_halfspaces, rotation, translation,
)
from mpc_controller.obstacles import ConvexPolytope, axis_aligned_rectangle

# Unit square of half-size 1 at the origin, and a car of half-length 2 and
# half-width 1 whose body origin is its centre (like the real footprint).
ROWS = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
OBSTACLE = ConvexPolytope(ROWS, np.ones(4))
G, g = ROWS, np.array([2.0, 2.0, 1.0, 1.0])
S = np.sqrt(0.5)


def run(x, y, phi):
    lam, mu, dist, collides = compute_dual_warmstart([x], [y], [phi], [OBSTACLE], G, g)
    return lam[0][0], mu[0][0], dist[0][0], collides[0][0]


@pytest.mark.parametrize(
    "pose, distance, lam, mu",
    [
        ((5, 0, 0), 2.0, [1, 0, 0, 0], [0, 1, 0, 0]),
        ((4, 3, 0), np.sqrt(2), [S, 0, S, 0], [0, S, 0, S]),
        ((5, 0, np.pi / 2), 3.0, [1, 0, 0, 0], [0, 0, 1, 0]),
    ],
)
def test_known_geometries(pose, distance, lam, mu):
    lam_k, mu_k, dist, collides = run(*pose)
    assert dist == pytest.approx(distance, abs=1e-5)
    np.testing.assert_allclose(lam_k, lam, atol=1e-4)
    np.testing.assert_allclose(mu_k, mu, atol=1e-4)
    assert not collides


def test_overlap_is_flagged():
    _, _, dist, collides = run(1.0, 0.0, 0.0)
    assert collides and dist < 1e-6


def test_all_constraints_hold_on_random_poses():
    rng = np.random.default_rng(0)
    for _ in range(30):
        x, y = rng.uniform(-6, 6, 2)
        phi = rng.uniform(-np.pi, np.pi)
        lam, mu, dist, _ = run(x, y, phi)
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        n = OBSTACLE.A.T @ lam
        s = -g @ mu + (OBSTACLE.A @ [x, y] - OBSTACLE.b) @ lam
        assert s == pytest.approx(dist, abs=1e-6)  # (1) with d = -dist
        assert np.linalg.norm(G.T @ mu + R.T @ n) <= 1e-6  # (2)
        assert np.linalg.norm(n) <= 1 + 1e-6  # (3)
        assert np.all(lam >= 0) and np.all(mu >= 0)  # (4)


def _frozen_legacy(optimizer, states):
    """Verbatim copy of the original heuristic, to prove 'legacy' is unchanged."""
    G, g = footprint_halfspaces(optimizer.params)
    lambda_guesses, mu_guesses = [], []
    for obstacle in optimizer.obstacles:
        lambdas = np.zeros((len(states), len(obstacle.A)))
        mus = np.zeros((len(states), len(G)))
        for si, state in enumerate(states):
            rotated = rotation(state)
            translated = translation(state, optimizer.params)
            best = -np.inf
            for fi, normal in enumerate(obstacle.A):
                lv = np.zeros(len(obstacle.A))
                lv[fi] = 1.0 / np.linalg.norm(normal)
                bn = -rotated.T @ (obstacle.A.T @ lv)
                mv = np.array([max(bn[0], 0.0), max(-bn[0], 0.0),
                               max(bn[1], 0.0), max(-bn[1], 0.0)])
                sep = -g @ mv + (obstacle.A @ translated - obstacle.b) @ lv
                if sep > best:
                    best, lambdas[si], mus[si] = sep, lv, mv
        lambda_guesses.append(lambdas)
        mu_guesses.append(mus)
    return tuple(lambda_guesses), tuple(mu_guesses)


def test_legacy_option_is_identical_to_old_code():
    obstacles = (axis_aligned_rectangle(1.5, 0.0, 0.38, 0.30),)
    states = np.array([[0.0, 0.0, 0.0, 0.0], [0.7, 0.4, 0.5, 0.2], [1.5, 0.1, -0.3, 0.0]])
    optimizer = TrajectoryOptimizer(OptimizerConfig(dual_init="legacy"), obstacles=obstacles)
    new_l, new_m = optimizer._dual_initial_guess(states)
    old_l, old_m = _frozen_legacy(optimizer, states)
    for a, b in zip(new_l + new_m, old_l + old_m):
        np.testing.assert_array_equal(a, b)


def test_default_is_optimal_separation_and_unknown_value_raises():
    assert OptimizerConfig().dual_init == "optimal_separation"
    with pytest.raises(ValueError, match="dual_init"):
        OptimizerConfig(dual_init="nonsense")


def test_colliding_steps_fall_back_to_legacy_values(caplog):
    obstacles = (axis_aligned_rectangle(0.0, 0.0, 0.4, 0.4),)
    states = np.array([[0.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]])  # step 0 overlaps
    new = TrajectoryOptimizer(obstacles=obstacles)
    old = TrajectoryOptimizer(OptimizerConfig(dual_init="legacy"), obstacles=obstacles)
    with caplog.at_level("WARNING"):
        lam, mu = new._dual_initial_guess(states)
    assert "step 0" in caplog.text
    legacy_lam, legacy_mu = old._dual_initial_guess(states)
    np.testing.assert_array_equal(lam[0][0], legacy_lam[0][0])
    np.testing.assert_array_equal(mu[0][0], legacy_mu[0][0])
    # Step 1 is clear of the obstacle, so it keeps the optimal-separation values.
    G, g = footprint_halfspaces(new.params)
    centers = np.array([translation(s, new.params) for s in states])
    direct, _, _, _ = compute_dual_warmstart(
        centers[:, 0], centers[:, 1], states[:, 2], obstacles, G, g
    )
    np.testing.assert_array_equal(lam[0][1], direct[0][1])
