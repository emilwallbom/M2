# OBCA trajectory optimizer

This ROS 2 Python package implements the autonomous-parking optimization in
Zhang, Liniger, and Borrelli, *Optimization-Based Collision Avoidance*. Despite
the project directory name, this component is a one-shot trajectory planner,
not the low-level path tracker.

The optimization variables are the complete state sequence
`x = [X, Y, yaw, velocity]`, the input sequence
`u = [steering, acceleration]`, the OBCA dual variables, and the common sample
time `Topt`. The start and goal states are fixed. Forward-Euler bicycle
dynamics, vehicle/input limits, steering-rate limits, and the full rectangular
vehicle collision constraints from equation (15) are enforced.

The equation (20) objective is

```text
q * N * Topt + sum(u[k]' R u[k] + du[k]' R_rate du[k])
du[k] = (u[k] - u[k-1]) / Topt
```

There is no path-tracking or state-reference term. Hybrid A* supplies only:

- the horizon length `N`;
- initial state values for IPOPT;
- inferred initial input values for IPOPT.

The solver is free to move every intermediate state away from that guess. The
returned trajectory is intended for the other group's low-level controller.

| File | Responsibility |
| --- | --- |
| `mpc_controller/model.py` | Rear-axle bicycle dynamics and vehicle rectangle |
| `mpc_controller/reference.py` | Hybrid A* path to `TrajectoryGuess` conversion |
| `mpc_controller/costs.py` | Equation (20) time/input/input-rate objective |
| `mpc_controller/constraints.py` | Speed, input, and steering-rate limits |
| `mpc_controller/obstacles.py` | Convex obstacle geometry and clearance checks |
| `mpc_controller/controller.py` | CasADi/IPOPT problem with equation (15) OBCA |
| `mpc_controller/demo.py` | One-shot hardware-free solve and diagnostic plot |

## Python interface

Hybrid A* should provide rows `[x_m, y_m, yaw_rad, direction]`, where direction
is `+1` for forward or `-1` for reverse. A three-column forward-only path is
also accepted.

```python
from mpc_controller.controller import TrajectoryOptimizer
from mpc_controller.reference import build_initial_guess

guess = build_initial_guess(hybrid_astar_path, sample_time=0.15)
optimizer = TrajectoryOptimizer(obstacles=tuple_of_convex_polytopes)
result = optimizer.solve(guess.states[0], guess.states[-1], guess)

assert result.success
states = result.states          # (N+1, 4)
inputs = result.controls        # (N, 2)
Topt = result.sample_time
```

`sample_time=0.15` in `build_initial_guess` is an initial value. `Topt` is a
decision variable bounded by `OptimizerConfig`, so maneuver duration is not
fixed before planning.

## Standalone demonstration

From the repository root:

```bash
python -m pip install casadi matplotlib numpy PyYAML
PYTHONPATH=MPC python -m mpc_controller.demo
```

The demo uses a smooth, collision-free path as a stand-in for the other group's
Hybrid A* output. It solves the optimization once and writes
`MPC/results/mpc_solution.png`. The plot shows the initial guess and optimized
trajectory separately, rectangular vehicle footprints, independently computed
clearance, optimized velocity and inputs, and the equation (20) cost terms.

Run the tests with:

```bash
PYTHONPATH=MPC python -m pytest MPC/test -q
```

## ROS integration boundary

`ros_node.py` remains a non-publishing interface scaffold. It must eventually
receive a complete Hybrid A* path and obstacle polytopes, run this optimizer,
and publish the optimized state/input/time sequence to the low-level group.
Mapping physical steering and acceleration to CasCAR actuator effort belongs to
that low-level controller, not this optimizer.
