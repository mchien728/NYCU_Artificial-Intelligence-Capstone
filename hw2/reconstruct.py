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
    TASK 1: Geometric Unprojection [cite: 12, 25-27]
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
    """
    Pre-processing: Voxelization and Normal Estimation [cite: 17, 29]
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals for pcd_down (required for Point-to-Plane ICP)
    search_radius = voxel_size * 2.0
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=30)
    )
    
    # Compute FPFH features for Global Registration
    # radius_feature = voxel_size * 5.0
    radius_feature = voxel_size * 5.0
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, pcd_fpfh

def cross_product_matrix(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def my_local_icp_algorithm(source_pcd, target_pcd, initial_transform):
    """
    TASK 2: Custom ICP Implementation (BONUS 20%) 
    Implement your own version of Point-to-Plane ICP.
    """
    # Implement the ICP loop:
    # 1. Find nearest neighbors using target_tree.search_knn_vector_3d
    # 2. Build the linear system (AtA)x = Atb
    # 3. Solve for pose update and update T_global

    T_global = initial_transform.copy()

    target_tree = o3d.geometry.KDTreeFlann(target_pcd)

    source_points = np.asarray(source_pcd.points)
    target_points = np.asarray(target_pcd.points)

    target_normals = np.asarray(target_pcd.normals)

    iterations = 20
    for _ in range(iterations):
        A = []
        b = []

        for i in range(len(source_points)):
            p_s = source_points[i]
            
            # Find the nearest neighbor
            _, idx, _ = target_tree.search_knn_vector_3d(p_s, 1)
            p_t = target_points[idx[0]]
            n_t = target_normals[idx[0]]
            error = np.dot((p_s - p_t), n_t)

            J_rotate = np.cross(p_s, n_t)
            J_transit = n_t
            J = np.hstack((J_rotate, J_transit))

            A.append(J)
            b.append(-error)

        A = np.array(A)
        b = np.array(b)
        # Solve Ax = b and get x
        delta_x = np.linalg.lstsq(A, b, rcond=None)[0]
        delta_theta = delta_x[:3]
        delta_t = delta_x[3:]

        R = np.eye(3) + cross_product_matrix(delta_theta)

        T_update = np.eye(4)
        T_update[:3, :3] = R
        T_update[:3, 3] = delta_t

        T_global = T_update @ T_global

        ones = np.ones((source_points.shape[0], 1))
        homo = np.hstack((source_points, ones))

        source_points = (T_update @ homo.T).T[:, :3]

        if np.linalg.norm(delta_x) < 1e-6:
            break

    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = T_global
    return result

def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    """
    TASK 2: Open3D ICP Implementation (REQUIRED)
    """
    # Use o3d.pipelines.registration.registration_icp
    # Estimation method should be TransformationEstimationPointToPlane()
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source=source_down, 
        target=target_down,
        max_correspondence_distance=threshold, 
        init=trans_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return reg_p2l

def visualize_and_evaluate(reconstructed_pcd, predicted_cam_poses, gt_poses, args):
    """
    TASK 3: Evaluation & Visualization
    """
    pred_points = np.array([pose[:3, 3] for pose in predicted_cam_poses])
    gt_points = np.array([pose[:3, 3] for pose in gt_poses])

    lines = [[i, i+1] for i in range(len(pred_points)-1)]

    # 1. Create LineSet for estimated trajectory (Red)
    pred_lineset = o3d.geometry.LineSet()
    pred_lineset.points = o3d.utility.Vector3dVector(pred_points)
    pred_lineset.lines = o3d.utility.Vector2iVector(lines)
    pred_lineset.colors = o3d.utility.Vector3dVector([1, 0, 0] for _ in lines)

    # 2. Create LineSet for ground truth trajectory (Black)
    gt_lineset = o3d.geometry.LineSet()
    gt_lineset.points = o3d.utility.Vector3dVector(gt_points)
    gt_lineset.lines = o3d.utility.Vector2iVector(lines)
    gt_lineset.colors = o3d.utility.Vector3dVector([0, 0, 0] for _ in lines)

    # Calculate Mean L2 Distance between predicted_cam_poses and gt_poses
    # L2 = sqrt(dx^2 + dy^2 + dz^2)
    errors = np.linalg.norm(pred_points - gt_points, axis=1)
    mean_l2_error = np.mean(errors)
    
    print(f"Mean L2 distance: {mean_l2_error:.6f} meters")
    
    # 3. Visualization
    o3d.visualization.draw_geometries([reconstructed_pcd], 
                                      window_name=f"Floor {args.floor} Reconstruction")
    return mean_l2_error

def reconstruct(args):
    voxel_size = 0.1
    rgb_dir = os.path.join(args.data_root, "rgb")
    depth_dir = os.path.join(args.data_root, "depth")

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

    # The first frame
    rgb = np.asarray(o3d.io.read_image(rgb_files[0]))
    depth = np.asarray(o3d.io.read_image(depth_files[0]))
    prev_pcd = depth_image_to_point_cloud(rgb, depth)

    accumulated_pcd += prev_pcd
    prev_down, prev_fpfh = preprocess_point_cloud(prev_pcd, voxel_size)

    recent_pcds = []
    # Reconstruction Loop
    for i in range(1, len(rgb_files)):
        print(f"Processing Frame {i}...")
        # 1. Convert RGB-D to PointCloud (Task 1)
        rgb = np.asarray(o3d.io.read_image(rgb_files[i]))
        depth = np.asarray(o3d.io.read_image(depth_files[i]))
        cur_pcd = depth_image_to_point_cloud(rgb, depth)

        # 2. Preprocess (Voxel/FPFH/Normals)
        cur_down, cur_fpfh = preprocess_point_cloud(cur_pcd, voxel_size)

        # 3. Execute Global Registration (RANSAC)
        distance_threshold = voxel_size * 1.5
        res_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            cur_down, prev_down,
            cur_fpfh, prev_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 500)
        )

        T_ran = res_ransac.transformation.copy()
        angle = np.linalg.norm(R.from_matrix(T_ran[:3,:3]).as_rotvec())
        angle_deg = np.degrees(angle)
        print(f"Frame {i} RANSAC rotation (deg): {np.degrees(angle):.4f}")

        # 4. Execute Local Registration (ICP - Task 2)
        local_map = o3d.geometry.PointCloud()
        for p in recent_pcds:
            local_map += p

        if len(accumulated_pcd.points) < 500:
            print("Warm-up phase, use RANSAC only")
            T_trans = np.linalg.inv(res_ransac.transformation)
        else:
            # prepare local_map
            local_map = accumulated_pcd.voxel_down_sample(voxel_size)
            local_map.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30)
            )

            if args.version == "my_icp":
                res_icp = my_local_icp_algorithm(cur_down, local_map, res_ransac.transformation)
            else:
                res_icp = local_icp_algorithm(
                    cur_down,
                    local_map,
                    res_ransac.transformation,
                    threshold=voxel_size * 0.4
                )

        T_icp = res_icp.transformation.copy()
        angle = np.linalg.norm(R.from_matrix(T_icp[:3,:3]).as_rotvec())
        angle_deg = np.degrees(angle)
        print(f"Frame {i} ICP rotation (deg): {np.degrees(angle):.4f}")

        print(f"Frame {i} RANSAC fitness: {res_ransac.fitness:.4f}, ICP fitness: {res_icp.fitness:.4f}")
        print(f"Frame {i} RANSAC inlier: {res_ransac.inlier_rmse:.4f}, ICP inlier: {res_icp.inlier_rmse:.4f}")
        
        if res_ransac.fitness < 0.5 or res_ransac.inlier_rmse > 0.05:
            print(f"Frame {i} RANSAC too poor, fallback to previous pose")
            # fallback: use previous pose without updating
            T_trans = np.eye(4)
        else:
            # ICP refinement
            T_icp = res_icp.transformation.copy()
            angle_deg = np.degrees(np.linalg.norm(R.from_matrix(T_icp[:3,:3]).as_rotvec()))
            if angle_deg > 10 or res_icp.fitness < 0.65:
                print(f"Frame {i} ICP failed, fallback to RANSAC")
                T_trans = np.linalg.inv(res_ransac.transformation)
            else:
                T_trans = np.linalg.inv(T_icp)

        # 5. Update camera_poses and accumulate points
        new_pose = camera_poses[-1] @ T_trans
        camera_poses.append(new_pose)

        R_global = np.array(new_pose[:3, :3])
        angle_global = np.degrees(
            np.linalg.norm(R.from_matrix(R_global).as_rotvec())
        )
        print(f"Frame {i} GLOBAL rotation (deg): {angle_global:.4f}")

        if res_icp.fitness > 0.7:
            temp_pcd = deepcopy(cur_pcd)
            temp_pcd.transform(new_pose)

            recent_pcds.append(temp_pcd)
            if len(recent_pcds) > 5:
                recent_pcds.pop(0)

            accumulated_pcd += temp_pcd

            if i % 10 == 0:
                accumulated_pcd = accumulated_pcd.voxel_down_sample(voxel_size=0.03)

        prev_down, prev_fpfh = cur_down, cur_fpfh

    accumulated_pcd = accumulated_pcd.voxel_down_sample(voxel_size=0.02)
    accumulated_pcd, _ = accumulated_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Post-processing: remove the ceiling
    points = np.asarray(accumulated_pcd.points)
    mask = points[:, 1] < np.percentile(points[:, 1], 95)
    accumulated_pcd = accumulated_pcd.select_by_index(np.where(mask)[0])
    
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
    
    print(f"Total execution time: {time.time() - start_time:.2f}s") # 
    visualize_and_evaluate(result_pcd, pred_poses, gt_poses, args)