# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import trimesh
import numpy as np
import matplotlib
from scipy.spatial.transform import Rotation
import copy
import cv2
import os
import requests


def visualize_camera_trajectories(trajectories, colors=None, scene_scale=None, show_up_axis: bool = True):
    """
    Creates a trimesh.Scene visualizing multiple camera trajectories.
    
    Args:
        trajectories (list): List of numpy arrays, each of shape [V, 4, 4] representing 
                           camera extrinsics (world to camera transformation)
        colors (np.ndarray, optional): Array of shape (len(trajectories), 3) with RGB colors.
                                     If None, uses matplotlib colormap
        scene_scale (float, optional): Scale factor for camera models. If None, auto-calculated
    
    Returns:
        trimesh.Scene: Scene containing camera models and trajectory lines
    """
    if not isinstance(trajectories, list) or len(trajectories) == 0:
        raise ValueError("trajectories must be a non-empty list")
    
    # Validate trajectory shapes
    for i, traj in enumerate(trajectories):
        if not isinstance(traj, np.ndarray) or traj.ndim != 3 or traj.shape[1:] != (4, 4):
            raise ValueError(f"Trajectory {i} must be numpy array of shape [V, 4, 4]")
            
    # This has no effect on visualization, but it aligns the global coordinate to the vggt scene.
    trajectories = [np.matmul(traj, np.array([
                    [-1, 0., 0., 0.],
                    [0.,  -1., 0., 0.],
                    [0.,  0., 1., 0.],
                    [0.,  0., 0., 1.],
                    ], dtype=traj.dtype)) for traj in trajectories]
    
    # Calculate scene scale if not provided
    if scene_scale is None:
        all_positions = []
        for traj in trajectories:
            # Extract camera positions (inverse of world-to-camera gives camera-to-world)
            camera_to_world = np.linalg.inv(traj)
            positions = camera_to_world[:, :3, 3]  # Extract translation part
            all_positions.append(positions)
        
        all_positions = np.concatenate(all_positions, axis=0)
        if len(all_positions) > 0:
            # Calculate bounding box diagonal
            min_pos = np.min(all_positions, axis=0)
            max_pos = np.max(all_positions, axis=0)
            scene_scale = np.linalg.norm(max_pos - min_pos)
        else:
            scene_scale = 1.0
    
    # Generate colors if not provided
    if colors is None:
        # Use a more vibrant color palette
        vibrant_colors = [
            (255, 0, 0),      # Bright Red
            (0, 255, 0),      # Bright Green  
            (0, 0, 255),      # Bright Blue
            (255, 255, 0),    # Bright Yellow
            (255, 0, 255),    # Bright Magenta
            (0, 255, 255),    # Bright Cyan
            (255, 128, 0),    # Bright Orange
            (128, 0, 255),    # Bright Purple
            (0, 255, 128),    # Bright Lime
            (255, 0, 128),    # Bright Pink
            (128, 255, 0),    # Bright Chartreuse
            (0, 128, 255),    # Bright Sky Blue
        ]
        
        colors = []
        for i in range(len(trajectories)):
            color_idx = i % len(vibrant_colors)
            colors.append(vibrant_colors[color_idx])
    else:
        if not isinstance(colors, np.ndarray) or colors.shape != (len(trajectories), 3):
            raise ValueError("colors must be numpy array of shape (len(trajectories), 3)")
        colors = [tuple(color) for color in colors]
    
    # Initialize scene
    scene = trimesh.Scene()
    
    # Add camera models for each trajectory
    for traj_idx, trajectory in enumerate(trajectories):
        trajectory_color = colors[traj_idx]
        
        # Convert world-to-camera to camera-to-world for visualization
        camera_to_world = np.linalg.inv(trajectory)
        
        # Add camera model for each frame
        for frame_idx in range(len(trajectory)):
            transform = camera_to_world[frame_idx]
            integrate_camera_into_scene(scene, transform, trajectory_color, scene_scale, show_up_axis=show_up_axis)
    
    # Add trajectory lines connecting camera origins
    for traj_idx, trajectory in enumerate(trajectories):
        trajectory_color = colors[traj_idx]
        
        # Get camera positions in world coordinates
        camera_to_world = np.linalg.inv(trajectory)
        positions = camera_to_world[:, :3, 3]
        
        # Create line segments connecting consecutive positions
        if len(positions) > 1:
            for i in range(len(positions) - 1):
                start_pos = positions[i]
                end_pos = positions[i + 1]
                
                # Create a thin cylinder as a line segment
                line_length = np.linalg.norm(end_pos - start_pos)
                if line_length > 1e-6:  # Only create line if points are not identical
                    # Create a thin cylinder oriented along the line
                    line_radius = scene_scale * 0.002  # Thin line
                    line_mesh = trimesh.creation.cylinder(radius=line_radius, height=line_length)
                    
                    # Calculate rotation to align cylinder with line direction
                    direction = (end_pos - start_pos) / line_length
                    z_axis = np.array([0, 0, 1])
                    
                    # Find rotation to align z-axis with direction
                    if np.allclose(direction, z_axis):
                        rotation_matrix = np.eye(3)
                    elif np.allclose(direction, -z_axis):
                        rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    else:
                        # Use cross product to find rotation axis
                        rotation_axis = np.cross(z_axis, direction)
                        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                        
                        # Calculate rotation angle
                        cos_angle = np.dot(z_axis, direction)
                        angle = np.arccos(np.clip(cos_angle, -1, 1))
                        
                        # Create rotation matrix using Rodrigues' formula
                        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                                     [rotation_axis[2], 0, -rotation_axis[0]],
                                     [-rotation_axis[1], rotation_axis[0], 0]])
                        rotation_matrix = (np.eye(3) + 
                                         np.sin(angle) * K + 
                                         (1 - np.cos(angle)) * (K @ K))
                    
                    # Create transformation matrix
                    transform = np.eye(4)
                    transform[:3, :3] = rotation_matrix
                    transform[:3, 3] = start_pos + (end_pos - start_pos) / 2
                    
                    # Apply transformation to mesh
                    line_mesh.apply_transform(transform)
                    
                    # Set color
                    line_mesh.visual.face_colors = np.array([*trajectory_color, 255])
                    
                    scene.add_geometry(line_mesh)
    
    return scene


def predictions_to_glb(
    predictions,
    conf_thres=50.0,
    filter_by_frames="all",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    target_dir=None,
    prediction_mode="Predicted Pointmap",
    show_up_axis: bool = True,
    scene_scale: float | None = None,
) -> trimesh.Scene:
    """
    Converts VGGT predictions to a 3D scene represented as a GLB file.

    Args:
        predictions (dict): Dictionary containing model predictions with keys:
            - world_points: 3D point coordinates (S, H, W, 3)
            - world_points_conf: Confidence scores (S, H, W)
            - images: Input images (S, H, W, 3)
            - extrinsic: Camera extrinsic matrices (S, 3, 4)
        conf_thres (float): Percentage of low-confidence points to filter out (default: 50.0)
        filter_by_frames (str): Frame filter specification (default: "all")
        mask_black_bg (bool): Mask out black background pixels (default: False)
        mask_white_bg (bool): Mask out white background pixels (default: False)
        show_cam (bool): Include camera visualization (default: True)
        mask_sky (bool): Apply sky segmentation mask (default: False)
        target_dir (str): Output directory for intermediate files (default: None)
        prediction_mode (str): Prediction mode selector (default: "Predicted Pointmap")

    Returns:
        trimesh.Scene: Processed 3D scene containing point cloud and cameras

    Raises:
        ValueError: If input predictions structure is invalid
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    if conf_thres is None:
        conf_thres = 10.0

    print("Building GLB scene")
    selected_frame_idx = None
    if filter_by_frames != "all" and filter_by_frames != "All":
        try:
            # Extract the index part before the colon
            selected_frame_idx = int(filter_by_frames.split(":")[0])
        except (ValueError, IndexError):
            pass

    if "Pointmap" in prediction_mode:
        print("Using Pointmap Branch")
        if "world_points" in predictions:
            pred_world_points = predictions["world_points"]  # No batch dimension to remove
            pred_world_points_conf = predictions.get("world_points_conf", np.ones_like(pred_world_points[..., 0]))
        else:
            print("Warning: world_points not found in predictions, falling back to depth-based points")
            pred_world_points = predictions["world_points_from_depth"]
            pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))
    else:
        print("Using Depthmap and Camera Branch")
        pred_world_points = predictions["world_points_from_depth"]
        pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))

    # Get images from predictions
    images = predictions["images"]
    # Use extrinsic matrices instead of pred_extrinsic_list
    camera_matrices = predictions["extrinsic"]

    if mask_sky:
        if target_dir is not None:
            import onnxruntime

            skyseg_session = None
            target_dir_images = target_dir + "/images"
            image_list = sorted(os.listdir(target_dir_images))
            sky_mask_list = []

            # Get the shape of pred_world_points_conf to match
            S, H, W = (
                pred_world_points_conf.shape
                if hasattr(pred_world_points_conf, "shape")
                else (len(images), images.shape[1], images.shape[2])
            )

            # Download skyseg.onnx if it doesn't exist
            if not os.path.exists("skyseg.onnx"):
                print("Downloading skyseg.onnx...")
                download_file_from_url(
                    "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx"
                )

            for i, image_name in enumerate(image_list):
                image_filepath = os.path.join(target_dir_images, image_name)
                mask_filepath = os.path.join(target_dir, "sky_masks", image_name)

                # Check if mask already exists
                if os.path.exists(mask_filepath):
                    # Load existing mask
                    sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
                else:
                    # Generate new mask
                    if skyseg_session is None:
                        skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
                    sky_mask = segment_sky(image_filepath, skyseg_session, mask_filepath)

                # Resize mask to match H×W if needed
                if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
                    sky_mask = cv2.resize(sky_mask, (W, H))

                sky_mask_list.append(sky_mask)

            # Convert list to numpy array with shape S×H×W
            sky_mask_array = np.array(sky_mask_list)

            # Apply sky mask to confidence scores
            sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
            pred_world_points_conf = pred_world_points_conf * sky_mask_binary

    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        camera_matrices = camera_matrices[selected_frame_idx][None]

    vertices_3d = pred_world_points.reshape(-1, 3)
    # Handle different image formats - check if images need transposing
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    conf = pred_world_points_conf.reshape(-1)
    # Convert percentage threshold to actual confidence value
    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        conf_threshold = np.percentile(conf, conf_thres)

    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    if mask_black_bg:
        black_bg_mask = colors_rgb.sum(axis=1) >= 16
        conf_mask = conf_mask & black_bg_mask

    if mask_white_bg:
        # Filter out white background pixels (RGB values close to white)
        # Consider pixels white if all RGB values are above 240
        white_bg_mask = ~((colors_rgb[:, 0] > 240) & (colors_rgb[:, 1] > 240) & (colors_rgb[:, 2] > 240))
        conf_mask = conf_mask & white_bg_mask

    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    if scene_scale is None:
        if vertices_3d is None or np.asarray(vertices_3d).size == 0:
            vertices_3d = np.array([[1, 0, 0]])
            colors_rgb = np.array([[255, 255, 255]])
            scene_scale = 1
        else:
            # Calculate the 5th and 95th percentiles along each axis
            lower_percentile = np.percentile(vertices_3d, 5, axis=0)
            upper_percentile = np.percentile(vertices_3d, 95, axis=0)

            # Calculate the diagonal length of the percentile bounding box
            scene_scale = np.linalg.norm(upper_percentile - lower_percentile)
    else:
        scene_scale = scene_scale

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")

    # Initialize a 3D scene
    scene_3d = trimesh.Scene()

    # Add point cloud data to the scene
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)

    scene_3d.add_geometry(point_cloud_data)

    # Prepare 4x4 matrices for camera extrinsics
    num_cameras = len(camera_matrices)
    extrinsics_matrices = np.zeros((num_cameras, 4, 4))
    extrinsics_matrices[:, :3, :4] = camera_matrices
    extrinsics_matrices[:, 3, 3] = 1

    if show_cam:
        # Add camera models to the scene
        for i in range(num_cameras):
            world_to_camera = extrinsics_matrices[i]
            camera_to_world = np.linalg.inv(world_to_camera)
            rgba_color = colormap(i / num_cameras)
            current_color = tuple(int(255 * x) for x in rgba_color[:3])

            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale, show_up_axis=show_up_axis)

    # Align scene to the observation of the first camera
    scene_3d = apply_scene_alignment(scene_3d, extrinsics_matrices)

    print("GLB Scene built")
    return scene_3d


def integrate_camera_into_scene(
    scene: trimesh.Scene,
    transform: np.ndarray,
    face_colors: tuple,
    scene_scale: float,
    show_up_axis: bool = True,
):
    """
    Integrates a fake camera mesh into the 3D scene.

    Args:
        scene (trimesh.Scene): The 3D scene to add the camera model.
        transform (np.ndarray): Transformation matrix for camera positioning.
        face_colors (tuple): Color of the camera face.
        scene_scale (float): Scale of the scene.
    """

    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1

    # Create cone shape for camera
    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    opengl_transform = get_opengl_conversion_matrix()
    # Combine transformations
    complete_transform = transform @ opengl_transform @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    # Generate mesh for the camera
    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    vertices_combined = np.concatenate(
        [
            camera_cone_shape.vertices,
            0.95 * camera_cone_shape.vertices,
            transform_points(slight_rotation, camera_cone_shape.vertices),
        ]
    )
    vertices_transformed = transform_points(complete_transform, vertices_combined)

    mesh_faces = compute_camera_faces(camera_cone_shape)

    # Add the camera mesh to the scene
    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)

    if show_up_axis:
        R_wc = transform[:3, :3]
        pos = transform[:3, 3]
        direction = -R_wc[:, 1]
        norm = np.linalg.norm(direction)
        if norm > 1e-9:
            direction = direction / norm
            axis_len = scene_scale * 0.06
            axis_radius = scene_scale * 0.002
            axis_mesh = trimesh.creation.cylinder(radius=axis_radius, height=axis_len)

            z_axis = np.array([0.0, 0.0, 1.0])
            if np.allclose(direction, z_axis):
                rotation_matrix = np.eye(3)
            elif np.allclose(direction, -z_axis):
                rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            else:
                rotation_axis = np.cross(z_axis, direction)
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                cos_angle = np.dot(z_axis, direction)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                              [rotation_axis[2], 0, -rotation_axis[0]],
                              [-rotation_axis[1], rotation_axis[0], 0]])
                rotation_matrix = (np.eye(3) +
                                   np.sin(angle) * K +
                                   (1 - np.cos(angle)) * (K @ K))

            transform_axis = np.eye(4)
            transform_axis[:3, :3] = rotation_matrix
            transform_axis[:3, 3] = pos + direction * (axis_len / 2.0)
            axis_mesh.apply_transform(transform_axis)
            axis_mesh.visual.face_colors = face_colors
            scene.add_geometry(axis_mesh)



def apply_scene_alignment(scene_3d: trimesh.Scene, extrinsics_matrices: np.ndarray) -> trimesh.Scene:
    """
    Aligns the 3D scene based on the extrinsics of the first camera.

    Args:
        scene_3d (trimesh.Scene): The 3D scene to be aligned.
        extrinsics_matrices (np.ndarray): Camera extrinsic matrices.

    Returns:
        trimesh.Scene: Aligned 3D scene.
    """
    # Set transformations for scene alignment
    opengl_conversion_matrix = get_opengl_conversion_matrix()

    # Rotation matrix for alignment (180 degrees around the y-axis)
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()

    # Apply transformation
    initial_transformation = np.linalg.inv(extrinsics_matrices[0]) @ opengl_conversion_matrix @ align_rotation
    scene_3d.apply_transform(initial_transformation)
    return scene_3d


def get_opengl_conversion_matrix() -> np.ndarray:
    """
    Constructs and returns the OpenGL conversion matrix.

    Returns:
        numpy.ndarray: A 4x4 OpenGL conversion matrix.
    """
    # Create an identity matrix
    matrix = np.identity(4)

    # Flip the y and z axes
    matrix[1, 1] = -1
    matrix[2, 2] = -1

    return matrix


def transform_points(transformation: np.ndarray, points: np.ndarray, dim: int = None) -> np.ndarray:
    """
    Applies a 4x4 transformation to a set of points.

    Args:
        transformation (np.ndarray): Transformation matrix.
        points (np.ndarray): Points to be transformed.
        dim (int, optional): Dimension for reshaping the result.

    Returns:
        np.ndarray: Transformed points.
    """
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]

    # Apply transformation
    transformation = transformation.swapaxes(-1, -2)  # Transpose the transformation matrix
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]

    # Reshape the result
    result = points[..., :dim].reshape(*initial_shape, dim)
    return result


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    """
    Computes the faces for the camera mesh.

    Args:
        cone_shape (trimesh.Trimesh): The shape of the camera cone.

    Returns:
        np.ndarray: Array of faces for the camera mesh.
    """
    # Create pseudo cameras
    faces_list = []
    num_vertices_cone = len(cone_shape.vertices)

    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices_cone
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone

        faces_list.extend(
            [
                (v1, v2, v2_offset),
                (v1, v1_offset, v3),
                (v3_offset, v2, v3),
                (v1, v2, v2_offset_2),
                (v1, v1_offset_2, v3),
                (v3_offset_2, v2, v3),
            ]
        )

    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)


def segment_sky(image_path, onnx_session, mask_filename=None):
    """
    Segments sky from an image using an ONNX model.
    Thanks for the great model provided by https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing

    Args:
        image_path: Path to input image
        onnx_session: ONNX runtime session with loaded model
        mask_filename: Path to save the output mask

    Returns:
        np.ndarray: Binary mask where 255 indicates non-sky regions
    """

    assert mask_filename is not None
    image = cv2.imread(image_path)

    result_map = run_skyseg(onnx_session, [320, 320], image)
    # resize the result_map to the original image size
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    # Fix: Invert the mask so that 255 = non-sky, 0 = sky
    # The model outputs low values for sky, high values for non-sky
    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original < 32] = 255  # Use threshold of 32

    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    cv2.imwrite(mask_filename, output_mask)
    return output_mask


def run_skyseg(onnx_session, input_size, image):
    """
    Runs sky segmentation inference using ONNX model.

    Args:
        onnx_session: ONNX runtime session
        input_size: Target size for model input (width, height)
        image: Input image in BGR format

    Returns:
        np.ndarray: Segmentation mask
    """

    # Pre process:Resize, BGR->RGB, Transpose, PyTorch standardization, float32 cast
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    # Post process
    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")

    return onnx_result


def download_file_from_url(url, filename):
    """Downloads a file from a Hugging Face model repo, handling redirects."""
    try:
        # Get the redirect URL
        response = requests.get(url, allow_redirects=False)
        response.raise_for_status()  # Raise HTTPError for bad requests (4xx or 5xx)

        if response.status_code == 302:  # Expecting a redirect
            redirect_url = response.headers["Location"]
            response = requests.get(redirect_url, stream=True)
            response.raise_for_status()
        else:
            print(f"Unexpected status code: {response.status_code}")
            return

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")