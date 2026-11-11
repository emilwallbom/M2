"""Paper-faithful, warm-started OBCA trajectory optimization.

Hybrid A* supplies only initial values and the horizon length. The nonlinear
program is free to create a different trajectory between fixed start and goal
states while minimizing equation (20) of Zhang et al.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from time import perf_counter

import casadi as ca
import numpy as np

from .constraints import build_constraints
from .costs import ObjectiveWeights, trajectory_objective
from .dual_warmstart import compute_dual_warmstart
from .model import (
    NU,
    NX,
    VehicleParams,
    footprint_halfspaces,
    rotation,
    step,
    translation,
)
from .obstacles import ConvexPolytope
from .reference import TrajectoryGuess

logger = logging.getLogger(__name__)

# Allowed values of OptimizerConfig.dual_init.
DUAL_INIT_METHODS = ("optimal_separation", "legacy")


@dataclass(frozen=True)
class OptimizerConfig:
    """Numerical settings for the one-shot trajectory optimization."""

    minimum_sample_time: float = 0.05
    maximum_sample_time: float = 0.35
    minimum_clearance: float = 0.05
    solver_max_iterations: int = 500
    solver_max_cpu_time: float = 10.0
    solver_tolerance: float = 1e-5
    # How lambda and mu get their initial values for IPOPT:
    #   "optimal_separation": solve a small convex problem per (step, obstacle)
    #                         for the best separating line of the guess
    #                         (dual_warmstart.py). Default.
    #   "legacy":             the original heuristic (_legacy_dual_initial_guess).
    # Only the initial values change; the optimization problem is the same.
    dual_init: str = "optimal_separation"

    def __post_init__(self):
        if self.dual_init not in DUAL_INIT_METHODS:
            raise ValueError(
                f"dual_init must be one of {DUAL_INIT_METHODS}, got {self.dual_init!r}"
            )
        if not (0.0 < self.minimum_sample_time < self.maximum_sample_time):
            raise ValueError("sample-time bounds must satisfy 0 < min < max")
        if not np.isfinite(self.minimum_clearance) or self.minimum_clearance < 0.0:
            raise ValueError("minimum_clearance must be finite and nonnegative")
        if self.solver_max_iterations < 1 or self.solver_max_cpu_time <= 0.0:
            raise ValueError("solver iteration and time limits must be positive")


@dataclass(frozen=True)
class TrajectoryResult:
    """Complete trajectory passed from the optimizer to the low-level group."""

    success: bool
    states: np.ndarray
    controls: np.ndarray
    sample_time: float
    total_time: float
    solve_time: float
    status: str
    iterations: int
    objective: float
    time_cost: float
    input_cost: float
    input_rate_cost: float
    obstacle_lambdas: tuple[np.ndarray, ...]
    vehicle_mus: tuple[np.ndarray, ...]

    @property
    def control(self) -> np.ndarray:
        """First input, retained only as a convenience for existing callers."""
        return self.controls[0].copy() if len(self.controls) else np.zeros(NU)


def _validate_state(value, name: str) -> np.ndarray:
    state = np.asarray(value, dtype=float)
    if state.shape != (NX,) or not np.all(np.isfinite(state)):
        raise ValueError(f"{name} must be a finite vector with shape ({NX},)")
    return state


class TrajectoryOptimizer:
    """Solve the paper's finite-horizon optimal-control problem once."""

    def __init__(
        self,
        config: OptimizerConfig | None = None,
        params: VehicleParams | None = None,
        weights: ObjectiveWeights | None = None,
        obstacles: tuple[ConvexPolytope, ...] = (),
    ):
        self.config = config or OptimizerConfig()
        self.params = params or VehicleParams()
        self.weights = weights or ObjectiveWeights()
        self.obstacles = tuple(obstacles)

    def _dual_initial_guess(self, states: np.ndarray):
        """Initial lambda/mu for the given state guess; method set by dual_init."""
        # I read the knob here, the single place where the dual guess is made.
        if self.config.dual_init == "legacy":
            return self._legacy_dual_initial_guess(states)

        G, g = footprint_halfspaces(self.params)
        # The collision constraints use t = translation(state), the footprint
        # CENTRE (the state itself is the rear axle), so I use the same here.
        centers = np.array([translation(state, self.params) for state in states])
        lambdas, mus, _, collides = compute_dual_warmstart(
            centers[:, 0], centers[:, 1], states[:, 2], self.obstacles, G, g
        )
        if any(mask.any() for mask in collides):
            # Where the guess already touches an obstacle, optimal separation
            # has no usable answer (lambda, mu ~ 0), so I fall back to the
            # legacy values for exactly those (k, m).
            legacy_lambdas, legacy_mus = self._legacy_dual_initial_guess(states)
            for m, mask in enumerate(collides):
                for k in np.flatnonzero(mask):
                    logger.warning(
                        "initial guess touches obstacle %d at step %d; "
                        "using legacy dual initialization there", m, k,
                    )
                    lambdas[m][k] = legacy_lambdas[m][k]
                    mus[m][k] = legacy_mus[m][k]
        return tuple(lambdas), tuple(mus)

    def _legacy_dual_initial_guess(self, states: np.ndarray):
        G, g = footprint_halfspaces(self.params)
        lambda_guesses = []
        mu_guesses = []
        for obstacle in self.obstacles:
            lambdas = np.zeros((len(states), len(obstacle.A)))
            mus = np.zeros((len(states), len(G)))
            for state_index, state in enumerate(states):
                rotated = rotation(state)
                translated = translation(state, self.params)
                best_separation = -np.inf
                for facet_index, normal in enumerate(obstacle.A):
                    normal_length = np.linalg.norm(normal)
                    lambda_value = np.zeros(len(obstacle.A))
                    lambda_value[facet_index] = 1.0 / normal_length
                    world_normal = obstacle.A.T @ lambda_value
                    body_normal = -rotated.T @ world_normal
                    mu_value = np.array(
                        [max(body_normal[0], 0.0), max(-body_normal[0], 0.0),
                         max(body_normal[1], 0.0), max(-body_normal[1], 0.0)]
                    )
                    separation = -g @ mu_value + (
                        obstacle.A @ translated - obstacle.b
                    ) @ lambda_value
                    if separation > best_separation:
                        best_separation = separation
                        lambdas[state_index] = lambda_value
                        mus[state_index] = mu_value
            lambda_guesses.append(lambdas)
            mu_guesses.append(mus)
        return tuple(lambda_guesses), tuple(mu_guesses)

    def _failure(self, horizon: int, elapsed: float, status: str) -> TrajectoryResult:
        return TrajectoryResult(
            False,
            np.full((horizon + 1, NX), np.nan),
            np.full((horizon, NU), np.nan),
            np.nan,
            np.nan,
            elapsed,
            status,
            0,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            (),
            (),
        )

    def solve(
        self,
        start_state,
        goal_state,
        warm_start: TrajectoryGuess,
        previous_control=None,
    ) -> TrajectoryResult:
        """Optimize a trajectory; ``warm_start`` never enters the objective."""
        start = _validate_state(start_state, "start_state")
        goal = _validate_state(goal_state, "goal_state")
        if not isinstance(warm_start, TrajectoryGuess):
            raise TypeError("warm_start must be a TrajectoryGuess")
        previous = np.zeros(NU) if previous_control is None else np.asarray(previous_control, dtype=float)
        if previous.shape != (NU,) or not np.all(np.isfinite(previous)):
            raise ValueError(f"previous_control must have shape ({NU},)")

        horizon = warm_start.horizon
        opti = ca.Opti()
        states = opti.variable(horizon + 1, NX)
        controls = opti.variable(horizon, NU)
        sample_time = opti.variable()

        opti.subject_to(states[0, :].T == ca.DM(start))
        opti.subject_to(states[-1, :].T == ca.DM(goal))
        opti.subject_to(
            opti.bounded(
                self.config.minimum_sample_time,
                sample_time,
                self.config.maximum_sample_time,
            )
        )
        for index in range(horizon):
            predicted = step(
                states[index, :].T,
                controls[index, :].T,
                sample_time,
                self.params,
                xp=ca,
            )
            opti.subject_to(states[index + 1, :].T == predicted)

        objective = trajectory_objective(
            controls,
            sample_time,
            previous,
            weights=self.weights,
            xp=ca,
        )
        opti.minimize(objective)

        ordinary = build_constraints(
            states, controls, previous, sample_time, self.params, xp=ca
        )
        opti.subject_to(opti.bounded(ordinary.lower, ordinary.values, ordinary.upper))

        G_array, g_array = footprint_halfspaces(self.params)
        G, g = ca.DM(G_array), ca.DM(g_array)
        obstacle_lambdas = []
        vehicle_mus = []
        for obstacle in self.obstacles:
            A, b = ca.DM(obstacle.A), ca.DM(obstacle.b)
            lambdas = opti.variable(horizon + 1, len(obstacle.A))
            mus = opti.variable(horizon + 1, len(G_array))
            opti.subject_to(ca.vec(lambdas) >= 0.0)
            opti.subject_to(ca.vec(mus) >= 0.0)
            for index in range(horizon + 1):
                state = states[index, :].T
                lambda_value = lambdas[index, :].T
                mu_value = mus[index, :].T
                rotated = rotation(state, xp=ca)
                translated = translation(state, self.params, xp=ca)
                world_normal = A.T @ lambda_value
                separation = -ca.dot(g, mu_value) + ca.dot(
                    A @ translated - b, lambda_value
                )
                opti.subject_to(separation >= self.config.minimum_clearance)
                opti.subject_to(G.T @ mu_value + rotated.T @ world_normal == 0.0)
                opti.subject_to(ca.sumsqr(world_normal) <= 1.0)
            obstacle_lambdas.append(lambdas)
            vehicle_mus.append(mus)

        opti.solver(
            "ipopt",
            {"expand": True, "print_time": False},
            {
                "print_level": 0,
                "sb": "yes",
                "max_iter": self.config.solver_max_iterations,
                "max_cpu_time": self.config.solver_max_cpu_time,
                "tol": self.config.solver_tolerance,
                "acceptable_tol": 10.0 * self.config.solver_tolerance,
            },
        )

        initial_states = warm_start.states.copy()
        initial_states[0] = start
        initial_states[-1] = goal
        initial_time = np.clip(
            warm_start.sample_time,
            self.config.minimum_sample_time,
            self.config.maximum_sample_time,
        )
        opti.set_initial(states, initial_states)
        opti.set_initial(controls, warm_start.controls)
        opti.set_initial(sample_time, initial_time)
        initial_lambdas, initial_mus = self._dual_initial_guess(initial_states)
        for variable, values in zip(obstacle_lambdas, initial_lambdas):
            opti.set_initial(variable, values)
        for variable, values in zip(vehicle_mus, initial_mus):
            opti.set_initial(variable, values)

        started = perf_counter()
        try:
            solution = opti.solve()
        except RuntimeError as error:
            elapsed = perf_counter() - started
            try:
                status = str(opti.stats().get("return_status", error))
            except RuntimeError:
                status = str(error).splitlines()[0]
            return self._failure(horizon, elapsed, status)
        elapsed = perf_counter() - started

        solved_states = np.asarray(solution.value(states), dtype=float).reshape(
            horizon + 1, NX
        )
        solved_controls = np.asarray(solution.value(controls), dtype=float).reshape(
            horizon, NU
        )
        solved_time = float(solution.value(sample_time))
        stats = solution.stats()
        success = bool(stats.get("success", False)) and np.all(np.isfinite(solved_states))
        if not success:
            return self._failure(horizon, elapsed, str(stats.get("return_status", "unknown")))

        deltas = np.diff(np.vstack((previous, solved_controls)), axis=0) / solved_time
        input_cost = float(np.sum(
            self.weights.steering * solved_controls[:, 0] ** 2
            + self.weights.acceleration * solved_controls[:, 1] ** 2
        ))
        rate_cost = float(np.sum(
            self.weights.steering_rate * deltas[:, 0] ** 2
            + self.weights.acceleration_rate * deltas[:, 1] ** 2
        ))
        time_cost = float(self.weights.time * horizon * solved_time)
        return TrajectoryResult(
            True,
            solved_states,
            solved_controls,
            solved_time,
            horizon * solved_time,
            elapsed,
            str(stats.get("return_status", "unknown")),
            int(stats.get("iter_count", 0)),
            float(solution.value(objective)),
            time_cost,
            input_cost,
            rate_cost,
            tuple(
                np.asarray(solution.value(v), dtype=float).reshape(
                    horizon + 1, v.shape[1]
                )
                for v in obstacle_lambdas
            ),
            tuple(
                np.asarray(solution.value(v), dtype=float).reshape(
                    horizon + 1, v.shape[1]
                )
                for v in vehicle_mus
            ),
        )


# Existing project code uses MPC naming, although this component is the
# paper's one-shot trajectory optimizer rather than a path-tracking controller.
MPCConfig = OptimizerConfig
MPCResult = TrajectoryResult
MPCController = TrajectoryOptimizer
