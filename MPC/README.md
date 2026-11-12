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
| `mpc_controller/hybrid_astar.py` | Forward/reverse geometric warm-start search |
| `mpc_controller/model.py` | Rear-axle bicycle dynamics and vehicle rectangle |
| `mpc_controller/reference.py` | Hybrid A* path to `TrajectoryGuess` conversion |
| `mpc_controller/costs.py` | Equation (20) time/input/input-rate objective |
| `mpc_controller/constraints.py` | Speed, input, and steering-rate limits |
| `mpc_controller/obstacles.py` | Convex obstacle geometry and clearance checks |
| `mpc_controller/controller.py` | CasADi/IPOPT problem with equation (15) OBCA |
| `mpc_controller/demo.py` | One-shot hardware-free solve and diagnostic plot |
| `mpc_controller/ros_node.py` | Goal-driven A* -> OBCA ROS pipeline |

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

## ROS pipeline

Build and start the planner in the project's ROS 2 Jazzy environment:

```bash
colcon build --symlink-install --packages-up-to mpc_controller
source install/setup.bash
ros2 launch mpc_controller mpc.launch.py agent_name:=panther
```

The node waits for odometry and a goal. On each new goal it runs Hybrid A*,
uses the resulting forward/reverse path to initialize IPOPT, runs OBCA once,
and publishes the complete result.

| Direction | Relative topic | Message |
| --- | --- | --- |
| Subscribe | `odom` | `nav_msgs/msg/Odometry` |
| Subscribe | `goal_pose` | `geometry_msgs/msg/PoseStamped` |
| Publish | `warm_start_path` | `nav_msgs/msg/Path` |
| Publish | `optimized_path` | `nav_msgs/msg/Path` |
| Publish | `optimized_trajectory` | `cascar_interfaces/msg/OptimizedTrajectory` |

`OptimizedTrajectory` contains the common optimized sample time, all poses and
velocities, and the steering/acceleration values for each interval. This is the
interface intended for the low-level controller. Physical actuator conversion
remains the low-level group's responsibility.

`config/mpc.yaml` supplies one shared obstacle list as repeated lower-left
`[x, y, width, height]` values. Hybrid A* uses inflated versions of these
rectangles for its coarse search and OBCA uses their exact convex half-space
representations. The default list is empty and must be populated from the
actual map before obstacle-aware operation.

## Full simulated chain

Build the workspace and launch the repeatable straight-line scenario:

```bash
colcon build --symlink-install --packages-up-to mpc_controller
source install/setup.bash
ros2 launch mpc_controller full_chain.launch.py agent_name:=panther
```

This launch explicitly selects the simulator and disables Qualisys. After
odometry starts, the scenario publishes a goal 1.5 m ahead. Hybrid A* creates
the warm start, OBCA publishes `optimized_trajectory`, and the low-level node
time-samples its velocity and steering arrays into `actuator_command`. All
topics share the `/panther` namespace. Useful checks in another terminal are:

```bash
ros2 topic echo /panther/optimized_trajectory --once
ros2 topic echo /panther/actuator_command
```

## Physical parking mission

With the Raspberry Pi hardware drivers running and visible on the same ROS
domain, the Linux project PC can start its complete side of the stack with:

```bash
ros2 launch mpc_controller hardware_parking.launch.py agent_name:=tiger
```

The launcher starts vehicle state estimation with Qualisys, RViz, Hybrid A*/
OBCA, the low-level controller, and a one-shot mission. It waits for odometry,
a stationary vehicle, and an eight-second startup delay. The default target is
expressed in the vehicle's starting frame: 0.8 m
behind, 0.4 m left, with the same final heading. Hardware throttle effort is
limited to 0.2 by default.

Mission geometry and the effort limit can be changed at launch:

```bash
ros2 launch mpc_controller hardware_parking.launch.py \
  agent_name:=tiger forward_m:=-0.6 left_m:=0.35 \
  yaw_delta_rad:=0.0 max_throttle_output:=0.15
```

Use `qualisys_used:=false` only when running without the Qualisys system.

### Physical slalom from `scene_demo.py`

The plotted three-wall slalom has a dedicated hardware launcher:

```bash
ros2 launch mpc_controller hardware_slalom.launch.py agent_name:=tiger
```

Place the rear axle at `(-0.5, 0.0)` with heading `0` in the Qualisys odometry
frame. The goal is `(5.5, 0.0, 0)`. Place axis-aligned barriers using these
lower-left positions and sizes in metres:

| Barrier | Lower-left `(x, y)` | Size `(width, height)` |
| --- | --- | --- |
| 1 | `(1.0, -1.5)` | `(0.3, 1.8)` |
| 2 | `(2.6, -0.3)` | `(0.3, 1.8)` |
| 3 | `(4.2, -1.5)` | `(0.3, 1.8)` |

The same values from `config/slalom.yaml` initialize Hybrid A* and OBCA. The
launcher sends the goal after eight seconds and limits normalized throttle to
`0.2` by default.

### Paper T-junction maneuver

The paper's constrained T-junction is recreated at Cascar scale. The narrow
upper chamber forces the characteristic forward/reverse maneuver before the
car exits to the left. Run the standalone plotted solve with:

```bash
PYTHONPATH=MPC python -m mpc_controller.scene_demo --scene paper_tjunction
```

The result is written to `MPC/test/scene_paper_tjunction.png`. For hardware,
place the rear axle at `(-1.5, 1.30)` facing `+X` and run:

```bash
ros2 launch mpc_controller hardware_paper_tjunction.launch.py agent_name:=tiger
```

The goal rear-axle pose is `(0.0, 0.35, pi/2)` inside the vertical stem. The
obstacle footprints are:

| Obstacle | Lower-left `(x, y)` | Size `(width, height)` |
| --- | --- | --- |
| Lower left | `(-3.0, 0.0)` | `(2.675, 1.0)` |
| Lower right | `(0.325, 0.0)` | `(2.675, 1.0)` |
| Top wall | `(-3.0, 1.6)` | `(6.0, 0.2)` |
