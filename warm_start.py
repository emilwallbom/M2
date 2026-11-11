from dataclasses import dataclass, replace
import csv
import heapq
import itertools
import numpy as np

l = 1 # Läng på bilen
max_steering = np.pi/3 # FYLL I BASERAT PÅ BILEN
N = 9 # Keep this odd so one steering angle is straight ahead.
PLOT_REVERSE = True  # Set to False to show forward paths only.
speed = 1
EDGE_LENGTH = 2
DT = 0.01
GOAL_TOLERANCE = 0.25
GOAL_HEADING_TOLERANCE = np.deg2rad(5)
MAX_EXPANSIONS = 50000

x_max = 10
x_min = -10
y_max = 10
y_min = -10
obstacles = []

@dataclass
class Point:
   x: float
   y: float
   heading: float

@dataclass
class Rectangle:
   x: float
   y: float
   b: float
   h: float


def dist(p1, p2):
   return np.linalg.norm([p1.x-p2.x,p1.y-p2.y])


def dynamics(p, steer, reverse=False):
   """Roll the bicycle model forward for one fixed-length search edge."""
   v = speed * (-1 if reverse else 1)
   p1 = replace(p)
   path = [replace(p1)]
   steps = max(1, int(np.ceil(EDGE_LENGTH / (speed * DT))))
   dt = EDGE_LENGTH / (speed * steps)
   for _ in range(steps):
      p1.x += v * np.cos(p1.heading) * dt
      p1.y += v * np.sin(p1.heading) * dt
      p1.heading = wrap_angle(p1.heading + dt * v * np.tan(steer) / l)
      if not is_point_in_map(p1) or is_point_in_obstacle(p1):
         return None
      path.append(replace(p1))
   return path

def wrap_angle(angle):
   return (angle + np.pi) % (2 * np.pi) - np.pi

def is_goal(p, goal):
   position_ok = dist(p, goal) <= GOAL_TOLERANCE
   heading_error = abs(wrap_angle(p.heading - goal.heading))
   heading_ok = heading_error <= GOAL_HEADING_TOLERANCE
   return position_ok and heading_ok

def state_key(p):
   """Use the generated continuous pose directly as the search state."""
   return (p.x, p.y, wrap_angle(p.heading))

def a_star(start, goal):
   """Search generated continuous poses using Euclidean distance as h."""
   steering_angles = np.linspace(-max_steering, max_steering, N)
   reverse_options = [False, True] if PLOT_REVERSE else [False]
   counter = itertools.count()
   start_key = state_key(start)
   frontier = [(dist(start, goal), next(counter), 0.0, start_key, start)]
   best_cost = {start_key: 0.0}
   parent = {start_key: None}
   edge_paths = {}
   expansions = 0

   while frontier and expansions < MAX_EXPANSIONS:
      _, _, cost, key, pose = heapq.heappop(frontier)
      if cost != best_cost.get(key):
         continue
      if is_goal(pose, goal):
         # Reassemble the sampled motions from start to the reached pose.
         segments = []
         cursor = key
         while parent[cursor] is not None:
            prev = parent[cursor]
            segments.append(edge_paths[cursor][1:])
            cursor = prev
         result = [replace(start)]
         for segment in reversed(segments):
            result.extend(segment)
         return result

      expansions += 1
      for steer in steering_angles:
         for reverse in reverse_options:
            segment = dynamics(pose, steer, reverse=reverse)
            if segment is None:
               continue
            successor = segment[-1]
            successor_key = state_key(successor)
            step_cost = EDGE_LENGTH
            if reverse:
               step_cost *= 1.2
            new_cost = cost + step_cost
            if new_cost >= best_cost.get(successor_key, float("inf")):
               continue
            best_cost[successor_key] = new_cost
            parent[successor_key] = key
            edge_paths[successor_key] = segment
            heuristic = dist(successor, goal)
            heapq.heappush(
               frontier,
               (new_cost + heuristic, next(counter), new_cost, successor_key, successor),
            )
   return None

def save_path_csv(path, filename="planned_path.csv"):
   """Save every sampled pose; heading is stored in radians."""
   with open(filename, "w", newline="") as csv_file:
      writer = csv.writer(csv_file)
      writer.writerow(["x", "y", "heading_rad"])
      writer.writerows((point.x, point.y, point.heading) for point in path)
   print(f"Saved plan to {filename}")

def is_point_in_obstacle(p):
   for o in obstacles:
      if p.x <o.x+o.b and p.x>o.x and p.y>o.y and p.y<o.y+o.h:
         return True
   return False

def is_point_in_map(p):
    if p.x <x_max and p.x>x_min and p.y>y_min and p.y<y_max:
        return True
    return False

def plot_point(p):
   return False
   


p_start = Point(0,0,-np.pi/2)
p_goal = Point(9,9,0)

# A varied course of rectangular obstacles, all expressed as
# Rectangle(lower_left_x, lower_left_y, width, height).
obstacles.extend([
   Rectangle(2.0, -1.5, 1.0, 4.0),  # central barrier
   Rectangle(4.5, 2.0, 3.0, 1.0),   # upper horizontal block
   Rectangle(4.5, -4.0, 1.0, 3.0),  # lower vertical block
   Rectangle(-3.5, 3.0, 3.0, 1.0),  # upper-left block
   Rectangle(-5.0, -3.5, 2.5, 2.5), # lower-left block
   Rectangle(7.0, 5.0, 1.5, 2.0),   # near-goal-side block
])



import matplotlib.pyplot as plt

path = a_star(p_start, p_goal)
if path is None:
   print(f"A* found no path within {MAX_EXPANSIONS} expansions.")
else:
   print(f"A* found a path with {len(path)} sampled poses.")
   save_path_csv(path)
   plt.plot([point.x for point in path], [point.y for point in path], label="A* path")
   stride = max(1, len(path) // 20)
   for point in path[::stride]:
      plt.scatter(
         point.x,
         point.y,
         marker=(3, 0, np.degrees(point.heading - np.pi/2)),
         color="tab:blue",
         s=45,
         zorder=3,
      )

for obstacle in obstacles:
   plt.gca().add_patch(
      plt.Rectangle((obstacle.x, obstacle.y), obstacle.b, obstacle.h,
                    color="tab:red", alpha=0.35, label="Obstacle")
   )
plt.scatter([p_start.x, p_goal.x], [p_start.y, p_goal.y],
            color=["green", "black"], zorder=4, label="Start / goal")

plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.legend(fontsize="small")
plt.show()
