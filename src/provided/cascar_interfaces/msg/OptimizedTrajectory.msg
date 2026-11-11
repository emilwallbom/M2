std_msgs/Header header

# Common optimized sample time Topt. There are len(poses)-1 control intervals.
float64 sample_time

# State trajectory x = [position, yaw from orientation, velocity].
geometry_msgs/Pose[] poses
float64[] velocities

# Input trajectory u = [steering_angle, acceleration].
float64[] steering_angles
float64[] accelerations

float64 objective
