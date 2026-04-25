import random
import sys
from typing import List, Tuple
import math
from pathlib import Path
import numpy as np
import cv2

from map_processor import load_and_filter_map, select_start, get_goal_pixels
from navigator import init_sim, execute_waypoint_path

HW3_DIR = Path(__file__).resolve().parents[1]
POINT_CLOUD_DATA = str(HW3_DIR / "semantic_3d_pointcloud" / "point.npy")
COLOR_DATA = str(HW3_DIR / "semantic_3d_pointcloud" / "color0255.npy")

# Sample semantic color and index dictionaries for a few object categories. 
# Check hw0/replica_v1/apartment_0/habitat/info_semantic.json and 
# hw3/color_coding_semantic_segmentation_classes.xlsx for the full list of 
# categories and their corresponding colors and indices.
# Check json to see correct id
SEMANTIC_DICTS = {
    "colors": {
        "rack": [[0, 255, 133]],
        "cooktop": [[7, 255, 224]],
        "sofa": [[10, 0, 255]],
        "cushion": [[255, 9, 92]],
        "stair": [[173, 255, 0]],
    },
    "indices": {
        "rack": 8,
        "cooktop": 280,
        "sofa": 196,
        "cushion": 430,
        "stair": 192,
    },
}

def is_free(point, occupancy_map):
    px, py = point
    h, w = occupancy_map.shape
    if px < 0 or py < 0 or px >= w or py >= h:
        return False
    
    return occupancy_map[py, px] == 0


def get_random_sample(goal, occupancy_map, goal_bias=0.1):
    h, w = occupancy_map.shape
    if random.random() < goal_bias:
        return goal
    
    while True:
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        if is_free((x, y), occupancy_map):
            return (x, y)


def get_adaptive_goal_bias(iter_idx, max_iter, min_bias=0.05, max_bias=0.3):
    # Bonus: Increase goal bias over time to early exploration, late goal-directed sampling.
    if max_iter <= 1:
        return max_bias
    progress = iter_idx / (max_iter - 1)
    return min_bias + (max_bias - min_bias) * progress


def nearest_node(nodes, sample):
    # node in nodes must be free, so don't need to check
    sx, sy = sample
    best_node = None
    min_dist = float("inf")

    for node in nodes:
        nx, ny = node
        dist = math.sqrt((sx - nx) ** 2 + (sy - ny) ** 2)
        if dist < min_dist:
            min_dist = dist
            best_node = node

    return best_node


def steer(from_node, to_node, occupancy_map, step_size=10):
    px_from, py_from = from_node
    px_to, py_to = to_node

    px_dist = px_to - px_from
    py_dist = py_to - py_from
    dist = math.sqrt(px_dist * px_dist + py_dist * py_dist)

    if dist == 0:
        return from_node
    
    max_step = min(step_size, dist)
    for step in range(int(max_step), 0, -1):
        scale = step / dist
        px_new = px_from + px_dist * scale
        py_new = py_from + py_dist * scale

        new_node = (int(round(px_new)), int(round(py_new)))
        if is_free(new_node, occupancy_map):
            return new_node
        
    return from_node


def is_path_collision(from_node, to_node, occupancy_map):
    px_from, py_from = from_node
    px_to, py_to = to_node

    steps = max(abs(px_to - px_from), abs(py_to - py_from))
    if steps == 0:
        return is_free(from_node, occupancy_map)
    
    for i in range(1, steps):
        t = i / steps
        px_test = int(round(px_from + (px_to - px_from) * t))
        py_test = int(round(py_from + (py_to - py_from) * t))

        if not is_free((px_test, py_test), occupancy_map):
            return True
        
    return False


def reconstruct_path(parents, goal):
    path = []
    cur = goal

    while cur is not None:
        path.append(cur)
        cur = parents[cur]

    path.reverse()
    return path


def plan_path(
    start,
    goal,
    occupancy_map,
    iter=5000,
    step_size=10,
    goal_threshold=20,
    min_goal_bias=0.1,
    max_goal_bias=0.35,
):
    if not is_free(start, occupancy_map):
        return None, []
    
    nodes = [start]
    parents = {start: None}
    explored_edges = []
    for i in range(iter):
        goal_bias = get_adaptive_goal_bias(i, iter, min_goal_bias, max_goal_bias)
        sample = get_random_sample(goal, occupancy_map, goal_bias=goal_bias)
        nearest = nearest_node(nodes, sample)
        new_node = steer(nearest, sample, occupancy_map, step_size)

        if is_path_collision(nearest, new_node, occupancy_map):
            continue
        if new_node in parents:
            continue

        nodes.append(new_node)
        parents[new_node] = nearest
        explored_edges.append((nearest, new_node))

        dist_to_goal = math.sqrt((new_node[0] - goal[0]) ** 2 + (new_node[1] - goal[1]) ** 2)
        if dist_to_goal <= goal_threshold:
            parents[goal] = new_node
            return reconstruct_path(parents, new_node), explored_edges
        
    return None, explored_edges
            

def pick_goal(map_img) -> Tuple[str, Tuple[int, int]]:
    prompt = "Enter semantic destination (ex: 'rack', 'cooktop', 'sofa'): "
    goal_prompt = input(prompt).strip().lower()
    if goal_prompt not in SEMANTIC_DICTS["colors"]:
        print(f"Goal '{goal_prompt}' is not valid.")
        sys.exit(1)

    goal_pixels = get_goal_pixels(map_img, SEMANTIC_DICTS["colors"], goal_prompt)
    goal = random.choice(goal_pixels)
    return goal_prompt, goal


def run_in_sim(start_world: Tuple[float, float], world_path: List[Tuple[float, float]], goal_prompt: str):
    start_x, start_z = start_world
    print(f"Spawning Agent at world position: ({start_x:.3f}, {start_z:.3f})")

    sim, agent, _ = init_sim(start_x=start_x, start_z=start_z)
    execute_waypoint_path(world_path, sim, agent, SEMANTIC_DICTS["indices"][goal_prompt])


def visualize_path(map_img, path, explored_edges, start, goal):
    vis_img = (map_img * 255).astype(np.uint8).copy()

    for point_prev, point_cur in explored_edges:
        cv2.line(vis_img, point_prev, point_cur, (0, 0, 0), 1)

    for i in range(1, len(path)):
        point_prev = path[i-1]
        point_cur = path[i]
        cv2.line(vis_img, point_prev, point_cur, (0, 0, 255), 2)

    for point in path:
        cv2.circle(vis_img, point, 1, (255, 0, 255), -1)

    cv2.circle(vis_img, start, 5, (0, 255, 0), -1)
    cv2.circle(vis_img, goal, 5, (255, 255, 0), -1)

    cv2.imwrite("Planned Path.png", vis_img)
    cv2.imshow("Planned Path", vis_img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()


def pixel_to_world(path, origin_world, resolution):
    world_path = []

    for px, py in path:
        x_world = origin_world[0] + px * resolution
        z_world = origin_world[1] + py * resolution
        world_path.append((x_world, z_world))

    return world_path


def main():
    """Entry point."""

    print("=== Step 1: Processing the 3D Map ===")
    # =============== 1-2 ===============
    map_img, occupancy_map, origin_world, resolution = load_and_filter_map(
        POINT_CLOUD_DATA,
        COLOR_DATA,
    )
    print(f"Map size: {map_img.shape[1]} x {map_img.shape[0]} pixels")
    print(f"World origin (x, z): {origin_world}")
    print(f"Map resolution: {resolution:.3f} meters/pixel")

    #cv2.imshow("occupancy_map", (occupancy_map * 255).astype(np.uint8))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    print("=== Step 2: Selecting Agent Start and Goal Positions ===")
    start = select_start(map_img)
    goal_prompt, goal = pick_goal(map_img)
    print(f"Goal pixel selected at coordinates: {goal}")


    print("=== Step 3: Executing Path Planning (RRT) ===")
    # =============== 2 ===============
    # implement RRT path planner in plan_path()
    path, explored_edges = plan_path(start, goal, occupancy_map)
    if not path:
        print("Planner could not find a path.")
        sys.exit(1)


    print("=== Step 4: Visualizing the Planned Path ===")
    # =============== 3 ===============
    # Visualize the planned path over the map
    visualize_path(map_img, path, explored_edges, start, goal)


    print("=== Step 5: Translating Path to Habitat Simulator ===")
    # =============== 4 ===============
    # Convert pixel path to world coordinates
    # world_path is a list of tuples(float, float) representing waypoints in world coordinates
    world_path = pixel_to_world(path, origin_world, resolution)

    print("Pixel path first point:", path[0])
    print("World path first point:", world_path[0])
    print("Pixel path last point:", path[-1])
    print("World path last point:", world_path[-1])


    run_in_sim(world_path[0], world_path, goal_prompt)


if __name__ == "__main__":
    main()
