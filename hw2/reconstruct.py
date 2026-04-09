import os
import re
import glob
import numpy as np
import open3d as o3d
import argparse
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import time

# ---------- Camera Intrinsics (Resolution 512x512, FOV 90) ----------
# These parameters are derived from the Habitat pinhole camera model.
IMG_W, IMG_H = 512, 512
FOV = np.deg2rad(90.0)
FX = (IMG_W / 2.0) / np.tan(FOV / 2.0)
FY = (IMG_H / 2.0) / np.tan(FOV / 2.0)
CX, CY = IMG_W / 2.0, IMG_H / 2.0

# Habitat uses imgs / 10 * 255
DEPTH_SCALE = 1000.0

def depth_image_to_point_cloud(rgb_image, depth_image):
    """
    TASK 1: Geometric Unprojection
    Convert depth pixels (u, v, d) into 3D world points (x, y, z).
    """
    # 1. Convert inputs to numpy arrays
    rgb = np.asarray(rgb_image)
    depth = np.asarray(depth_image).astype(np.float32)

    # 2. Convert depth to meters (Habitat depth is often scaled or normalized)
    depth = (depth * 10.0) / 255.0
    
    # 3. Create a coordinate grid for (u, v) pixels
    u = np.arange(IMG_W)
    v = np.arange(IMG_H)
    uu, vv = np.meshgrid(u, v)
    
    # Implement unprojection logic
    # Depth = 0 is invalid
    z = depth
    valid = (z > 0.1) & (z < 9.9)

    x = (uu - CX) * z / FX
    y = -(vv - CY) * z / FY
    z = -depth

    # shape: (H, W, 3)
    points_3d = np.stack((x, y, z), axis=-1)
    points_3d = points_3d[valid]

    colors_norm = rgb / 255.0
    colors_norm = colors_norm[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)  # N x 3
    pcd.colors = o3d.utility.Vector3dVector(colors_norm)
    return pcd

def preprocess_point_cloud(pcd, voxel_size):
    """"
    Pre-processing: Voxelization and Normal Estimation
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals for pcd_down (required for Point-to-Plane ICP)
    search_radius = voxel_size * 2.0
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=30)
    )
    
    # Compute FPFH features for Global Registration
    radius_feature = voxel_size * 5.0
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, pcd_fpfh

def my_local_icp_algorithm(source_pcd, target_pcd, initial_transform):
    """
    TASK 2: Custom ICP Implementation (BONUS 20%) 
    Implement your own version of Point-to-Plane ICP.
    """
    T_global = initial_transform.copy()
    
    # TODO: Implement the ICP loop:
    # 1. Find nearest neighbors using target_tree.search_knn_vector_3d
    # 2. Build the linear system (AtA)x = Atb
    # 3. Solve for pose update and update T_global
    
def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    """
    TASK 2: Open3D ICP Implementation (REQUIRED)
    """
    # Use o3d.pipelines.registration.registration_icp
    # Estimation method should be TransformationEstimationPointToPlane()
    res_ransac = o3d.pipelines.registration.registration_icp(
        source=source_down,
        target=target_down,
        max_correspondence_distance=threshold,
        init=trans_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=70),
    )
    return res_ransac

def visualize_and_evaluate(reconstructed_pcd, predicted_cam_poses, gt_poses, args):
    """
    TASK 3: Evaluation & Visualization
    """
    # extract camera positions
    pred_pos = np.array([pose[:3, 3] for pose in predicted_cam_poses]) 
    gt_pos = np.array([pose[:3, 3] for pose in gt_poses])
    
    min_len = min(len(pred_pos), len(gt_pos))
    pred_valid = pred_pos[:min_len]
    gt_valid = gt_pos[:min_len]

    # SVD Implementation
    if min_len > 1:
        gt_centroid = np.mean(gt_valid, axis=0)
        pred_centroid = np.mean(pred_valid, axis=0)

        # Convert to mean = 0
        gt_zero = gt_valid - gt_centroid
        pred_zero = pred_valid - pred_centroid

        conv_mat = gt_zero.T @ pred_zero
        U, _, Vt = np.linalg.svd(conv_mat)

        # Kabsch algo.
        R_align = Vt.T @ U.T
        if np.linalg.det(R_align) < 0:
            Vt[-1, :] *= -1
            R_align = Vt.T @ U.T
            
        t_align = pred_centroid - R_align @ gt_centroid
        gt_pos = (R_align @ gt_pos.T).T + t_align

    # 1. Create LineSet for estimated trajectory (Red)
    pred_lines = [[i, i+1] for i in range(len(pred_pos) - 1)]
    pred_lineset = o3d.geometry.LineSet()
    pred_lineset.points = o3d.utility.Vector3dVector(pred_pos)
    pred_lineset.lines  = o3d.utility.Vector2iVector(pred_lines)
    pred_lineset.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in pred_lines])
        
    # 2. Create LineSet for ground truth trajectory (Black)
    gt_lines = [[i, i+1] for i in range(len(gt_pos) - 1)]
    gt_lineset = o3d.geometry.LineSet()
    gt_lineset.points = o3d.utility.Vector3dVector(gt_pos)
    gt_lineset.lines  = o3d.utility.Vector2iVector(gt_lines)
    gt_lineset.colors = o3d.utility.Vector3dVector([[0, 0, 0] for _ in gt_lines])

    # Calculate Mean L2 Distance between predicted_cam_poses and gt_poses
    # L2 = sqrt(dx^2 + dy^2 + dz^2)
    mean_l2_error = np.nan
    
    l2_dis = np.linalg.norm(pred_valid - gt_valid, axis=1)
    mean_l2_error = float(np.mean(l2_dis))
    print(f"Mean L2 distance: {mean_l2_error:.6f} meters")

    # 3. Visualization
    o3d.visualization.draw_geometries([reconstructed_pcd, pred_lineset, gt_lineset], 
                                  window_name=f"Floor {args.floor} Reconstruction")
    
    return mean_l2_error

def remove_ceiling(accumulated_pcd, floor=1):
    points = np.asarray(accumulated_pcd.points)
    colors = np.asarray(accumulated_pcd.colors)

    if len(points) == 0:
        print("Warning: Point cloud is empty.")
        return accumulated_pcd

    ceiling_threshold = 0.6 if floor == 1 else 3.0

    mask = np.isfinite(points).all(axis=1) & (points[:, 1] < ceiling_threshold)
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])
    filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

    return filtered_pcd

def reconstruct(args):
    voxel_size = 0.05
    rgb_dir = os.path.join(args.data_root, "rgb")
    depth_dir = os.path.join(args.data_root, "depth")

    # Sort by file number
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")), 
                       key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")),
                         key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    # Load Ground Truth Poses
    gt_pose_path = os.path.join(args.data_root, "GT_pose.npy")
    gt_poses = []
    if os.path.exists(gt_pose_path):
        gt_data = np.load(gt_pose_path)
        for p in gt_data:
            mat = np.eye(4)
            mat[:3, :3] = R.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()
            mat[:3, 3] = [p[0], p[1], p[2]]
            gt_poses.append(mat)
        gt_poses = np.stack(gt_poses)

    camera_poses = [np.eye(4)]
    accumulated_pcd = o3d.geometry.PointCloud()

    # Reconstruction Loop
    for i in range(1, len(rgb_files)):
        print(f"Processing Frame {i}...")

        rgb_prev  = np.asarray(o3d.io.read_image(rgb_files[i-1]))
        dep_prev  = np.asarray(o3d.io.read_image(depth_files[i-1]))
        rgb_cur   = np.asarray(o3d.io.read_image(rgb_files[i]))
        dep_cur   = np.asarray(o3d.io.read_image(depth_files[i]))
        # Depth loaded as (H, W, 3), take 1 channel
        if dep_prev.ndim == 3:
            dep_prev = dep_prev[:, :, 0]
        if dep_cur.ndim == 3:
            dep_cur = dep_cur[:, :, 0]

        # 1. Convert RGB-D to PointCloud (Task 1)
        pcd_prev = depth_image_to_point_cloud(rgb_prev, dep_prev)
        pcd_cur = depth_image_to_point_cloud(rgb_cur, dep_cur)

        if i == 1 and len(pcd_prev.points) > 0:
            pcd_prev_world = deepcopy(pcd_prev)
            pcd_prev_world.transform(camera_poses[0])
            accumulated_pcd += pcd_prev_world

        # 2. Preprocess (Voxel/FPFH/Normals)
        prev_down, _ = preprocess_point_cloud(pcd_prev, voxel_size)
        cur_down, _ = preprocess_point_cloud(pcd_cur, voxel_size)

        coarse_voxel_size = voxel_size * 2.0
        prev_global_down, prev_global_fpfh = preprocess_point_cloud(pcd_prev, coarse_voxel_size)
        cur_global_down, cur_global_fpfh = preprocess_point_cloud(pcd_cur, coarse_voxel_size)

        # 3. Execute Global Registration (RANSAC)
        distance_threshold = coarse_voxel_size * 1.5
        trans_init = np.eye(4)
        if len(camera_poses) >= 2:
            trans_init = np.linalg.inv(camera_poses[-2]) @ camera_poses[-1]

        if len(cur_global_down.points) < 50 or len(prev_global_down.points) < 50:
            res_ransac = o3d.pipelines.registration.RegistrationResult()
            res_ransac.transformation = np.eye(4)
            res_ransac.fitness = 0.0
        else:    
            ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                cur_global_down, prev_global_down,
                cur_global_fpfh, prev_global_fpfh,
                mutual_filter=False,
                max_correspondence_distance=distance_threshold,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=4,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(40000, 500)
            )
            # Performance of RANSAC is too bad
            if ransac_result.fitness > 0.15 and np.linalg.norm(ransac_result.transformation[:3, 3]) < 1.0:
                trans_init = ransac_result.transformation

        # 4. Execute Local Registration (ICP - Task 2)
        icp_threshold = voxel_size * 1.5
        if args.version == 'open3d':
            res_icp = local_icp_algorithm(cur_down, prev_down, trans_init, icp_threshold)
        else:
            # not implement
            res_icp = my_local_icp_algorithm(cur_down, prev_down, trans_init)

        # 5. Update camera_poses and accumulate points
        T_icp = res_icp.transformation
        T_world = camera_poses[-1] @ T_icp
        camera_poses.append(T_world)
        
        pcd_cur_world = deepcopy(pcd_cur)
        pcd_cur_world.transform(T_world)
        accumulated_pcd += pcd_cur_world
        '''
        if i % 10 == 0: 
            o3d.visualization.draw_geometries([accumulated_pcd], window_name="Frame 0 check")
            '''

    # Post-processing: remove the ceiling
    accumulated_pcd = remove_ceiling(accumulated_pcd)
    
    return accumulated_pcd, camera_poses, gt_poses

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='open3d', help='open3d or my_icp')
    args = parser.parse_args()

    # Set data root based on floor
    args.data_root = f"data_collection/first_floor/" if args.floor == 1 else f"data_collection/second_floor/"

    start_time = time.time()
    result_pcd, pred_poses, gt_poses = reconstruct(args)
    
    print(f"Total execution time: {time.time() - start_time:.2f}s")
    visualize_and_evaluate(result_pcd, pred_poses, gt_poses, args)

