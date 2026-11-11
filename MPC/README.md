# MPC

Python ROS 2 package `mpc_controller`. This is a development scaffold: the
optimization and vehicle model are deliberately unimplemented. The node can
receive messages but publishes no actuator commands.

| File | Work to implement |
| --- | --- |
| `mpc_controller/model.py` | State/input definitions and discrete vehicle dynamics |
| `mpc_controller/reference.py` | Path sampling and reference trajectory |
| `mpc_controller/costs.py` | Tracking, input and terminal costs |
| `mpc_controller/constraints.py` | Vehicle, actuator and environment limits |
| `mpc_controller/controller.py` | Solver setup, optimization and solve status |
| `mpc_controller/ros_node.py` | ROS input conversion, control loop and command output |
| `config/mpc.yaml` | ROS parameters (currently only the timer rate) |
| `launch/mpc.launch.py` | Node startup and vehicle namespace |

Keep the numerical control components independent of ROS so they can later be
tested using recorded data or standalone simulations. Choose a solver and add
its dependencies once the formulation is agreed.

## Build and launch

Run in the project's ROS 2 Jazzy Linux/container environment, from the repository
root. `MPC/` is intentionally at the root; colcon discovers its `package.xml`
recursively. Do not restrict package discovery to `src/`.

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-up-to mpc_controller
source install/setup.bash
ros2 launch mpc_controller mpc.launch.py agent_name:=panther
```

Start the existing simulator/backend separately. Match `agent_name` to that
backend. With the default namespace the interfaces are:

| Direction | Topic | Message |
| --- | --- | --- |
| Subscribe | `/panther/odom` | `nav_msgs/msg/Odometry` |
| Subscribe | `/panther/reference_path` | `nav_msgs/msg/Path` |
| Publish (future implementation) | `/panther/actuator_command` | `cascar_interfaces/msg/ActuatorCommand` |

No reference-path producer is included. Define its coordinate frame and timing
contract before wiring it into the controller. Cascar throttle and steering are
dimensionless efforts in `[-1, 1]`; throttle is not speed in m/s. Add explicit
input conversion, stale-data handling and solver-failure handling before
enabling output. The scaffold does not stop a vehicle controlled by another node.

Suggested implementation order: state/input definitions and model, reference,
costs and constraints, solver, then ROS integration and simulator validation.
