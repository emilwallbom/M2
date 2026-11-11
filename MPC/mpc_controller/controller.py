"""Reusable CasADi/IPOPT nonlinear MPC, independent of ROS."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import casadi as ca
import numpy as np

from .constraints import build_constraints
from .costs import MPCWeights, tracking_cost
from .model import NU, NX, VehicleParams, step
from .reference import ReferenceTrajectory


@dataclass(frozen=True)
class MPCConfig:
    """Prediction and nonlinear-solver settings."""

    horizon: int = 30
    dt: float = 0.1
    solver_max_iterations: int = 100
    solver_max_cpu_time: float = 0.08
    solver_tolerance: float = 1e-5

    def __post_init__(self):
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if self.solver_max_iterations < 1:
            raise ValueError("solver_max_iterations must be positive")
        if not np.isfinite(self.solver_max_cpu_time) or self.solver_max_cpu_time <= 0:
            raise ValueError("solver_max_cpu_time must be finite and positive")


@dataclass(frozen=True)
class MPCResult:
    """One solve result; callers must check ``success`` before using control."""

    success: bool
    control: np.ndarray
    states: np.ndarray
    controls: np.ndarray
    solve_time: float
    status: str
    iterations: int
    objective: float


class MPCController:
    """Own one compiled optimization problem and its shifted warm start."""

    def __init__(
        self,
        config: MPCConfig | None = None,
        params: VehicleParams | None = None,
        weights: MPCWeights | None = None,
    ):
        self.config = config or MPCConfig()
        self.params = params or VehicleParams()
        self.weights = weights or MPCWeights()
        self._warm_states: np.ndarray | None = None
        self._warm_controls: np.ndarray | None = None
        self._build_problem()

    def _build_problem(self):
        horizon = self.config.horizon
        opti = ca.Opti()
        states = opti.variable(horizon + 1, NX)
        controls = opti.variable(horizon, NU)
        initial_state = opti.parameter(NX)
        reference = opti.parameter(horizon + 1, NX)
        previous_control = opti.parameter(NU)

        opti.subject_to(states[0, :].T == initial_state)
        for index in range(horizon):
            predicted = step(
                states[index, :].T,
                controls[index, :].T,
                self.config.dt,
                self.params,
                xp=ca,
            )
            opti.subject_to(states[index + 1, :].T == predicted)

        objective = tracking_cost(
            states,
            controls,
            reference,
            previous_control,
            dt=self.config.dt,
            weights=self.weights,
            xp=ca,
        )
        opti.minimize(objective)

        constraints = build_constraints(
            states,
            controls,
            previous_control,
            self.config.dt,
            self.params,
            xp=ca,
        )
        opti.subject_to(
            opti.bounded(constraints.lower, constraints.values, constraints.upper)
        )

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

        self._opti = opti
        self._states = states
        self._controls = controls
        self._initial_state = initial_state
        self._reference = reference
        self._previous_control = previous_control
        self._objective = objective

    def reset_warm_start(self):
        """Discard the previous solution, for example after receiving a new path."""
        self._warm_states = None
        self._warm_controls = None

    def _validate_inputs(self, state, reference, previous_control):
        state = np.asarray(state, dtype=float)
        previous_control = np.asarray(previous_control, dtype=float)
        reference_states = (
            reference.states
            if isinstance(reference, ReferenceTrajectory)
            else np.asarray(reference, dtype=float)
        )
        reference_states = np.asarray(reference_states, dtype=float)
        expected_reference_shape = (self.config.horizon + 1, NX)
        if state.shape != (NX,) or not np.all(np.isfinite(state)):
            raise ValueError(f"state must be a finite vector with shape ({NX},)")
        if previous_control.shape != (NU,) or not np.all(np.isfinite(previous_control)):
            raise ValueError(
                f"previous_control must be a finite vector with shape ({NU},)"
            )
        if reference_states.shape != expected_reference_shape:
            raise ValueError(
                f"reference must have shape {expected_reference_shape}, "
                f"got {reference_states.shape}"
            )
        if not np.all(np.isfinite(reference_states)):
            raise ValueError("reference must contain only finite values")
        return state, reference_states, previous_control

    def _initial_guess(self, reference_states):
        if self._warm_states is None or self._warm_controls is None:
            return reference_states, np.zeros((self.config.horizon, NU))
        return self._warm_states, self._warm_controls

    def _failure_result(self, elapsed: float, status: str) -> MPCResult:
        states = np.full((self.config.horizon + 1, NX), np.nan)
        controls = np.full((self.config.horizon, NU), np.nan)
        return MPCResult(
            False,
            np.zeros(NU),
            states,
            controls,
            elapsed,
            status,
            0,
            np.inf,
        )

    def solve(
        self,
        state,
        reference,
        previous_control=None,
    ) -> MPCResult:
        """Solve one horizon and return its first physical input ``[delta, a]``."""
        if previous_control is None:
            previous_control = np.zeros(NU)
        state, reference_states, previous_control = self._validate_inputs(
            state, reference, previous_control
        )
        initial_states, initial_controls = self._initial_guess(reference_states)

        self._opti.set_value(self._initial_state, state)
        self._opti.set_value(self._reference, reference_states)
        self._opti.set_value(self._previous_control, previous_control)
        self._opti.set_initial(self._states, initial_states)
        self._opti.set_initial(self._controls, initial_controls)

        started = perf_counter()
        try:
            solution = self._opti.solve()
        except RuntimeError as error:
            elapsed = perf_counter() - started
            try:
                status = str(self._opti.stats().get("return_status", error))
            except RuntimeError:
                status = str(error).splitlines()[0]
            return self._failure_result(elapsed, status)
        elapsed = perf_counter() - started

        solved_states = np.asarray(solution.value(self._states), dtype=float)
        solved_controls = np.asarray(solution.value(self._controls), dtype=float)
        stats = solution.stats()
        success = bool(stats.get("success", False))
        status = str(stats.get("return_status", "unknown"))
        iterations = int(stats.get("iter_count", 0))
        objective = float(solution.value(self._objective))
        if not (
            success
            and np.all(np.isfinite(solved_states))
            and np.all(np.isfinite(solved_controls))
        ):
            return self._failure_result(elapsed, status)

        self._warm_states = np.vstack((solved_states[1:], solved_states[-1:]))
        self._warm_controls = np.vstack(
            (solved_controls[1:], solved_controls[-1:])
        )
        return MPCResult(
            True,
            solved_controls[0].copy(),
            solved_states,
            solved_controls,
            elapsed,
            status,
            iterations,
            objective,
        )
