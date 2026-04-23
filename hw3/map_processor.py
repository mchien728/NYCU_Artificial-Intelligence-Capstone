import cv2
import numpy as np
from typing import List, Tuple

SCALE_FACTOR = 10000.0 / 255.0
CEILING_COLOR = np.array([8, 255, 214])
FLOOR_COLOR = np.array([255, 194, 7])
MAP_RESOLUTION = 0.05
POINT_RADIUS_PX = 1
ROBOT_RADIUS_PX = 1
MIN_COMPONENT_AREA = 20
MIN_OBSTACLE_HEIGHT = 0.10
MAX_OBSTACLE_HEIGHT = 1.50


def load_and_filter_map(point_path: str, color_path: str):
    points = np.load(point_path)
    colors = np.load(color_path)

    # Convert to real-world meters
    coords = points * SCALE_FACTOR

    # =============== 1-1 ===============
    # IMPORTANT: return map_img as float in value range [0, 1] for visualization downstream.
    # NOTE: in habitat sim, x z plane corresponds to world horizontal plane, and y is vertical.

    # To get a good 2d map, filter ceiling/floor, project to 2D
    # Floor is not obstable, the map shouldn't contain it
    ceiling_mask = np.all(np.isclose(colors, CEILING_COLOR), axis=1)
    floor_mask = np.all(np.isclose(colors, FLOOR_COLOR), axis=1)
    valid_mask = ~ceiling_mask & ~floor_mask

    coords_filtered = coords[valid_mask]
    colors_filtered = colors[valid_mask].astype(np.uint8)

    # [:, [0, 2]]: get all rows and get the first and third columns
    xz = coords_filtered[:, [0, 2]]
    x_min, z_min = xz.min(axis=0)
    x_max, z_max = xz.max(axis=0)

    width = int(np.ceil((x_max - x_min) / MAP_RESOLUTION)) + 1
    height = int(np.ceil((z_max - z_min) / MAP_RESOLUTION)) + 1

    x_pixel = np.round((xz[:, 0] - x_min) / MAP_RESOLUTION).astype(np.int32)
    y_pixel = np.round((xz[:, 1] - z_min) / MAP_RESOLUTION).astype(np.int32)

    x_pixel = np.clip(x_pixel, 0, width - 1)
    y_pixel = np.clip(y_pixel, 0, height - 1)

    # remove isolated points, inflate obstacles to get occupancy map, etc.
    map_img = np.ones((height, width, 3), dtype=np.uint8) * 255

    for x, y, color in zip(x_pixel, y_pixel, colors_filtered):
        cv2.circle(map_img, (x, y), POINT_RADIUS_PX, color.tolist(), -1)

    # Build the obstacle map from points that live near the floor plane.
    # Projecting every non-floor 3D point to 2D tends to seal narrow corridors.
    floor_y = float(np.median(coords[floor_mask, 1])) if np.any(floor_mask) else float(np.min(coords[:, 1]))
    obstacle_mask = valid_mask.copy()
    obstacle_mask &= coords[:, 1] >= floor_y + MIN_OBSTACLE_HEIGHT
    obstacle_mask &= coords[:, 1] <= floor_y + MAX_OBSTACLE_HEIGHT

    obstacle_coords = coords[obstacle_mask]
    if obstacle_coords.size > 0:
        obstacle_xz = obstacle_coords[:, [0, 2]]
        obs_x = np.round((obstacle_xz[:, 0] - x_min) / MAP_RESOLUTION).astype(np.int32)
        obs_y = np.round((obstacle_xz[:, 1] - z_min) / MAP_RESOLUTION).astype(np.int32)

        obs_x = np.clip(obs_x, 0, width - 1)
        obs_y = np.clip(obs_y, 0, height - 1)

        obs_mask = np.zeros((height, width), dtype=np.uint8)
        for x, y in zip(obs_x, obs_y):
            cv2.circle(obs_mask, (x, y), POINT_RADIUS_PX, 255, -1)

    # Group the obstacles
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(obs_mask, connectivity=8)
    obs_cleaned = np.zeros_like(obs_mask)
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= MIN_COMPONENT_AREA:
            obs_cleaned[labels == label_idx] = 255

    # reserve safe distance
    inflate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * ROBOT_RADIUS_PX + 1, 2 * ROBOT_RADIUS_PX + 1),
    )
    obs_inflated = cv2.dilate(obs_cleaned, inflate_kernel, iterations=1)

    occupancy_map = (obs_inflated > 0).astype(np.uint8)
    origin_world = (float(x_min), float(z_min))

    return map_img.astype(np.float32) / 255.0, occupancy_map, origin_world, MAP_RESOLUTION


def select_start(map_img: np.ndarray) -> Tuple[int, int]:
    """Display map and return user-clicked start coordinate."""
    start_point = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            start_point.append((x, y))
            print(f"Start selected: ({x}, {y})")

    cv2.namedWindow("Select Start")
    cv2.setMouseCallback("Select Start", mouse_callback)
    print("Click on the map window to select a start location...")

    while True:
        cv2.imshow("Select Start", (map_img * 255).astype(np.uint8))
        key = cv2.waitKey(1) & 0xFF
        if start_point:
            break
        if key == ord("q"):
            raise RuntimeError("No start selected. Exiting.")

    cv2.destroyWindow("Select Start")
    return start_point[0]


def get_goal_pixels(map_img: np.ndarray, semantic_dict: dict, goal_name: str) -> List[Tuple[int, int]]:
    """function to find all pixels corresponding to the goal object based on color matching."""

    if goal_name.lower() not in semantic_dict:
        raise ValueError(f"Unknown semantic object: {goal_name}. Available options: {list(semantic_dict.keys())}")

    goal_colors = semantic_dict[goal_name.lower()]
    goal_pixels: List[Tuple[float, float]] = []

    for gc in goal_colors:
        gc_norm = np.array(gc) / 255.0
        mask_goal = np.all(np.isclose(map_img, gc_norm, atol=10/255.0), axis=2)
        zs, xs = np.where(mask_goal)
        goal_pixels.extend(list(zip(xs, zs)))

    if not goal_pixels:
        raise ValueError(f"No valid pixels found for '{goal_name}'.")

    return goal_pixels
