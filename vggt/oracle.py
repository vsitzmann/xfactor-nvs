import os
from pathlib import Path
from collections import defaultdict
import math
from copy import deepcopy

import numpy as np
import trimesh
from trimesh import Scene

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .utils.pose_enc import pose_encoding_to_extri_intri
from .utils.load_fn import resize_images_tensor
from .visual_utils import predictions_to_glb, visualize_camera_trajectories
from .utils.geometry import unproject_depth_map_to_point_map
from .utils import torch_scatter
from .metrics.camera_metric import get_camera_metrics
from .metrics.vggt_loss import camera_loss, depth_loss
from .models.vggt import VGGT
from .sft_heads import VggtCameraHead, VggtDptHead

# ====================================================================================
# Typing
# ====================================================================================
from typing import *
from numbers import Number, Integral as Integer, Real

class ArrayLikeMeta(type):
    def __instancecheck__(self, instance):
        return hasattr(instance, 'shape')

class ArrayLike(metaclass=ArrayLikeMeta):
    """Any object with a .shape attribute"""

PathLike = Union[os.PathLike, str]
VggtPreds = Dict[str, Tensor | List[Tensor]]

try:
    import wandb
    WandbObject3D = wandb.Object3D
except ImportError:
    WandbObject3D = Any

# ====================================================================================
# Main Module
# ====================================================================================
class VggtOracle(torch.nn.Module):
    HW: Final[int] = 518

    def __init__(self, checkpoint_path: str | Path = "from_hub", *, img_size=518):
        super().__init__()
        self.HW = img_size

        print(f"Initializing VGGT...", flush=True)
        self.vggt = VGGT(
            img_size=img_size,
        )

        print("Loading VGGT Weights...", flush=True)
        if checkpoint_path == "from_hub":
            self.vggt.load_state_dict(
                torch.hub.load_state_dict_from_url("https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt")
            )      
        else:
            self.vggt.load_state_dict(torch.load(checkpoint_path))

        self.vggt.eval()
        for param in self.vggt.parameters():
            param.requires_grad = False


    def forward(
        self, images: Tensor | np.ndarray, 
        *,
        preds: str | Iterable[str] = ["depth", "pose", "color"],
        batch_chunk_size: int | None = None,
        frames_chunk_size: int | None = None,
        preprocess_mode: Literal["crop", "pad"] | None = "crop",
        resize_preds: bool = True,
        ref_frame_id: int = 0,
        use_middle_frame_for_inference: bool = True,
    ) -> VggtPreds:
        assert isinstance(ref_frame_id, int), f"{ref_frame_id}"

        H,W = images.shape[-3:-1] if isinstance(images, np.ndarray) else images.shape[-2:]
        if preprocess_mode is None:
            assert (H,W) == (self.HW, self.HW), f"Preprocess mode must be set to 'crop' or 'pad: {tuple(images.shape)}"


        images = self.preprocess_input(images, mode=preprocess_mode)
        B, S, _, _, _ = images.shape
        assert images.shape[-3:] == (3, self.HW, self.HW), f"images.shape: {tuple(images.shape)}"

        if use_middle_frame_for_inference:
            original_ref_frame_id = ref_frame_id
            ref_frame_id = S//2

        assert isinstance(ref_frame_id, int), f"{ref_frame_id}"
        if ref_frame_id < 0:
            ref_frame_id = S + ref_frame_id
        assert 0 <= ref_frame_id < S, f"{ref_frame_id} < {S}"
        im1, im2, im3 = images[:, :ref_frame_id], images[:, ref_frame_id:ref_frame_id+1], images[:, ref_frame_id+1:]
        images = torch.cat([im2, im1, im3], dim=1)

        if batch_chunk_size is None:
            batch_chunk_size = B
        num_chunks = math.ceil(B / batch_chunk_size)

        outputs = defaultdict(list)
        for image_chunk in torch.chunk(images, num_chunks):
            output = self._forward_once(image_chunk, preds=preds, frames_chunk_size=frames_chunk_size)
            for k, v in output.items():
                outputs[k].append(v)

        for k, v in outputs.items():
            if k == "intermediates":
                outputs[k]: List[Tensor] = sum(v, [])
            else:
                outputs[k]: Tensor = torch.cat(v, dim=0)

        if (H != self.HW) or (W != self.HW):
            if preprocess_mode == "crop":
                target_size = min(H,W)
            elif preprocess_mode == "pad":
                target_size = max(H,W)
            else:
                raise ValueError(f"Invalid preprocess mode: {preprocess_mode}")
            outputs = self.resize_preds(outputs, size=target_size, original_size=self.HW) if resize_preds else outputs


        for k, v in outputs.items():
            if isinstance(v, Tensor):
                v2, v1, v3 = v[:, :1], v[:, 1:ref_frame_id+1], v[:, ref_frame_id+1:]
                assert [v1.shape[1], v2.shape[1], v3.shape[1]] == [ref_frame_id, 1, S - ref_frame_id - 1], f"{[v1.shape[1], v2.shape[1], v3.shape[1]]} == {[ref_frame_id, 1, S - ref_frame_id - 1]}"
                outputs[k] = torch.cat([v1, v2, v3], dim=1)
            else:
                # assert isinstance(v, list)
                # assert all(isinstance(v_, Tensor) for v_ in v)
                raise NotImplementedError

        if use_middle_frame_for_inference:
            ref_frame_id = original_ref_frame_id

            if "pmap" in outputs:
                raise NotImplementedError
            if "T_cw" in outputs:
                T_cw = outputs["T_cw"]
                T_cw = torch.matmul(T_cw, torch.linalg.inv(T_cw[:, ref_frame_id:ref_frame_id+1]))
                outputs["T_cw"] = T_cw

        return outputs


    def compute_TPS(
        self,
        target_images: Tensor | np.ndarray,
        transfer_images: Tensor | np.ndarray,
        **kwargs,
    ) -> dict[str, float | Tensor]:
        target_T_cw = self.forward(target_images, preds=["pose", "color"], **kwargs)["T_cw"]
        transfer_T_cw = self.forward(transfer_images, preds=["pose", "color"], **kwargs)["T_cw"]

        return self.get_camera_metrics(target_T_cw=target_T_cw, pred_T_cw=transfer_T_cw)[0]
        
        


    # =============================================================================================
    # Metrics
    # =============================================================================================

    @classmethod
    def get_camera_metrics(
        cls, *,
        target_T_cw: Tensor,
        pred_T_cw: Tensor,
        baseline_T_cw: Tensor | None = None,  # Used to compute relative metrics.
        clip_scale: float | None = 50.0,
    ) -> tuple[
        dict[str, float | Tensor], 
        dict[str, float | Tensor], 
        dict[str, float | Tensor],
    ]:
        B, V, _, _ = target_T_cw.shape
        assert target_T_cw.shape == pred_T_cw.shape == (B,V,4,4), f"{tuple(target_T_cw.shape)} == {tuple(pred_T_cw.shape)} == (B, V, 4, 4)"
        if baseline_T_cw is not None:
            assert baseline_T_cw.shape == (B,V,4,4), f"{tuple(baseline_T_cw.shape)} == ({B}, {V}, 4, 4)"

        # ---------------------------------------------------------------------------------------------
        # Compute absolute metrics.
        # ---------------------------------------------------------------------------------------------
        low_metrics_pred, high_metrics_pred = cls._get_raw_camera_metrics(
            target_T_cw = target_T_cw,
            pred_T_cw = pred_T_cw,
        )
        low_metric_names = list(low_metrics_pred.keys())
        high_metric_names = list(high_metrics_pred.keys())


        # ---------------------------------------------------------------------------------------------
        # Compute relative metrics.
        # <low/high>_metrics_{abs/baseline/rel}: Dict[str, Tensor]  ||  Tensor shape: [B,]
        # ---------------------------------------------------------------------------------------------
        clip_scale = math.log(clip_scale)
        
        if baseline_T_cw is not None:
            low_metrics_baseline, high_metrics_baseline = cls._get_raw_camera_metrics(
                target_T_cw = target_T_cw,
                pred_T_cw = baseline_T_cw,
            )

            low_metrics_rel = {}
            for k in low_metric_names:
                metric = torch.log(low_metrics_baseline[k] / low_metrics_pred[k]).clamp(min=-clip_scale, max=clip_scale)
                low_metrics_rel[k] = metric.detach().cpu()


            high_metrics_rel = {}
            for k in high_metric_names:
                metric = torch.log(high_metrics_pred[k] / high_metrics_baseline[k]).clamp(min=-clip_scale, max=clip_scale)
                high_metrics_rel[k] = metric.detach().cpu()

        else:
            low_metrics_baseline = {}
            high_metrics_baseline = {}
            low_metrics_rel = {}
            high_metrics_rel = {}

        metrics_pred = {**low_metrics_pred, **high_metrics_pred}
        metrics_baseline = {**low_metrics_baseline, **high_metrics_baseline}
        metrics_rel = {**low_metrics_rel, **high_metrics_rel}    # Always higher the better


        return metrics_pred, metrics_baseline, metrics_rel


    # =============================================================================================
    # Visualization
    # =============================================================================================
    @classmethod
    def visualize_trajectory_comparison(
        cls,
        preds: List[VggtPreds],
        scene_ids: List[int],
        rescale: bool = False,
        show_up_axis: bool = True,
        scene_scale: float | None = None,
    ) -> List[trimesh.Scene]:
        ref_T_cw = {'T_cw': preds[0]["T_cw"].detach().cpu()[scene_ids]}
        # if rescale:
        #     ref_T_cw = cls.rescale_preds(ref_T_cw, scale=None)
        trajectories = []
        for pred in preds[1:]:
            pred = {'T_cw': pred["T_cw"].detach().cpu()[scene_ids]}
            if rescale:
                pred = cls.rescale_preds(pred, scale=ref_T_cw)
            trajectories.append(pred)
        ref_T_cw = ref_T_cw['T_cw'].numpy()
        assert len(ref_T_cw.shape) == 4, f"{ref_T_cw.shape}" # [B, V, 4, 4]
        trajectories = [traj['T_cw'].numpy() for traj in trajectories]
        trajectories = [ref_T_cw] + trajectories
        scenes = []
        for traj in zip(*trajectories):
            scene = visualize_camera_trajectories(list(traj), show_up_axis=show_up_axis, scene_scale=scene_scale)
            scenes.append(scene)
        return scenes


    @classmethod
    def visualize_scenes(
        cls,
        preds=None, 
        *,
        scene_ids = None,
        depth_map=None, depth_conf=None, T_cw=None, K=None, color=None, 
        conf_thres=10.0,
        exclude_camera=False,
        show_up_axis=False,
        scene_scale: float | None = None,
    ) -> List[Scene] | Scene:
        # ---------------------------------------------------------------------------------------------
        # Get predictions from preds.
        # ---------------------------------------------------------------------------------------------
        if preds is not None:
            depth_map = preds.get("depth_map", None) if depth_map is None else depth_map
            depth_conf = preds.get("depth_conf", None) if depth_conf is None else depth_conf
            T_cw = preds.get("T_cw", None) if T_cw is None else T_cw
            K = preds.get("K", None) if K is None else K
            color = preds.get("color", None) if color is None else color

        assert depth_map is not None, "depth_map is required"
        assert depth_conf is not None, "depth_conf is required"
        assert T_cw is not None, "T_cw is required"
        assert K is not None, "K is required"
        assert color is not None, "color is required"
        n_scenes = len(T_cw)
        assert n_scenes == len(depth_map) == len(depth_conf) == len(T_cw) == len(K) == len(color), f"{n_scenes} == {len(depth_map)} == {len(depth_conf)} == {len(T_cw)} == {len(K)} == {len(color)}"


        # ---------------------------------------------------------------------------------------------
        # Select scenes to visualize.
        # ---------------------------------------------------------------------------------------------
        squeeze_output = False
        if scene_ids is None:
            scene_ids = list(range(n_scenes))
        elif isinstance(scene_ids, Integer):
            squeeze_output = True
            scene_ids = [scene_ids]
        assert all(isinstance(i, Integer) for i in scene_ids), f"{scene_ids}"
        assert all((0 <= i < n_scenes) for i in scene_ids), f"{scene_ids}"

        def to_numpy(x, msg):
            if isinstance(x, Tensor):
                x = x.detach().cpu().numpy()
            else:
                assert isinstance(x, np.ndarray), f"{msg}: {type(x)}"
            return x

        depth_map = [to_numpy(depth_map[i], f"depth_map[{i}]") for i in scene_ids]
        depth_conf = [to_numpy(depth_conf[i], f"depth_conf[{i}]") for i in scene_ids]
        T_cw = [to_numpy(T_cw[i], f"T_cw[{i}]") for i in scene_ids]
        K = [to_numpy(K[i], f"K[{i}]") for i in scene_ids]
        color = [to_numpy(color[i], f"color[{i}]") for i in scene_ids]

        if isinstance(conf_thres, Real):
            conf_thres = [conf_thres for _ in range(n_scenes)]
        assert len(conf_thres) == n_scenes, f"{len(conf_thres)} != {n_scenes}"

        # ---------------------------------------------------------------------------------------------
        # Visualize scenes.
        # ---------------------------------------------------------------------------------------------
        scenes = []
        for i in range(len(depth_map)):
            scene_viz = cls._visualize_scene(
                depth_map=depth_map[i], 
                depth_conf=depth_conf[i], 
                T_cw=T_cw[i], 
                K=K[i], 
                color=color[i], 
                conf_thres=conf_thres[i],
                show_up_axis=show_up_axis,
                scene_scale=scene_scale,
            )
            scenes.append(scene_viz)

        if exclude_camera:
            for scene in scenes:
                for key in list(scene.geometry.keys()):
                    if isinstance(scene.geometry[key], trimesh.Trimesh):
                        scene.delete_geometry(key)
        
        if squeeze_output:
            return scenes[0]
        else:
            return scenes


    @classmethod
    def scene_to_pcd(
        cls,
        scene_viz: trimesh.Scene,
        *,
        mesh_sample_num: int | None = 2000,
        voxel_filter_kwargs: dict[str, Any] | None = None,
    ):
        import trimesh

        point_clouds: List[trimesh.PointCloud] = []
        for geometry in scene_viz.geometry.values():
            if isinstance(geometry, trimesh.Trimesh) and mesh_sample_num is not None:
                try:
                    points, face_indices = geometry.sample(mesh_sample_num, return_index=True)
                    colors = geometry.visual.face_colors[face_indices]
                    point_clouds.append(trimesh.PointCloud(points, colors=colors))
                except IndexError:
                    pass
            elif isinstance(geometry, trimesh.PointCloud):
                point_clouds.append(geometry)
            else:
                raise ValueError(f"unknown geometry type: {type(geometry)}")
        point_clouds: trimesh.PointCloud = sum(point_clouds)
        if isinstance(point_clouds, int):  # this happends when there is no point cloud in the scene. In this case, we use a dummy point cloud.
            vertices = np.array([[0., 0., 0.] for _ in range(100)], dtype=np.float32)
            colors = np.array([[1., 1., 1.] for _ in range(100)], dtype=np.float32)
        else:
            vertices = np.asarray(point_clouds.vertices, dtype=np.float32)
            colors = np.asarray(point_clouds.colors, dtype=np.float32) / 255.0

            if voxel_filter_kwargs is not None:
                vertices, colors = cls.voxel_filter(vertices, colors, **deepcopy(voxel_filter_kwargs))
                colors = np.clip(colors, 0., 1.)

        if colors.shape[-1] == 4:
            colors, alpha = colors[...,:3], colors[...,3:]
        elif colors.shape[-1] == 3:
            alpha = np.ones_like(colors[...,:1])
        else:
            raise ValueError(f"unknown colors shape: {colors.shape}")

        return vertices, colors, alpha


    @staticmethod
    def pcd_to_wandb(vertices: np.ndarray, colors: np.ndarray) -> WandbObject3D:
        """
        wandb renderer convention (left-handed):
        camera_right: -y
        camera_bottom: -z
        camera_forward: -x
        """
        if colors.dtype != np.uint8:
            colors = colors * 255.0

        camera_change = np.array([[ 0.,  0.,  -1.,  0.],
                                  [-1.,  0.,  0.,   0.],
                                  [ 0., -1.,  0.,   0.],
                                  [ 0.,  0.,  0.,   1.]])
        vertices = np.einsum("ij,...j", camera_change[:3, :3], vertices) + camera_change[:3, 3]

        import wandb
        return wandb.Object3D(np.concatenate([vertices, colors], axis=-1).astype(np.float16))


    @classmethod
    def scene_to_wandb(
        cls, scene_viz: trimesh.Scene,
        *,
        mesh_sample_num: int | None = 2000,
        voxel_filter_kwargs: Mapping[str, Any] | None = None,
    ) -> WandbObject3D:
        """
        Usage:

        wandb.log(
            {
                "train_scene_1": oracle.scene_to_wandb(scene_1),
                "train_scene_swapped_1": oracle.scene_to_wandb(scene_swapped_1),
                "train_scene_2": oracle.scene_to_wandb(scene_2),
                "train_scene_swapped_2": oracle.scene_to_wandb(scene_swapped_2),
            },
            step=total_steps
        )
        """
        import wandb

        assert isinstance(scene_viz, trimesh.Scene), f"{type(scene_viz)}"
        vertices, colors, alpha = cls.scene_to_pcd(scene_viz, mesh_sample_num=mesh_sample_num, voxel_filter_kwargs=voxel_filter_kwargs)
        return cls.pcd_to_wandb(vertices, colors)


    @classmethod
    def wandb_log_scene(
        cls, 
        scene: WandbObject3D | Mapping[str, Any] | Sequence[WandbObject3D], 
        key: str | None = None, 
        step: int | None = None, commit: bool | None = None,
    ):
        import wandb
        if isinstance(scene, Mapping):
            data = {}
            for k, v in scene.items():
                data[f"{key}_{k}" if key is not None else k] = v
        else:
            data = {key if key is not None else "scene": scene}

        wandb.log(data, step=step, commit=commit)


    # =============================================================================================
    # Supervised Fine-tuning heads
    # =============================================================================================

    @classmethod
    def get_camera_head(
        cls,
        dim_in: int,
        dim_embed: int,
        trunk_depth: int = 4,
        **kwargs,
    ) -> VggtCameraHead:
        kwargs = deepcopy(kwargs)
        pose_encoding_type = kwargs.pop("pose_encoding_type", "absT_quaR_FoV")
        num_heads = kwargs.pop("num_heads", 16)
        mlp_ratio = kwargs.pop("mlp_ratio", 4)
        init_values = kwargs.pop("init_values", 0.01)
        trans_act = kwargs.pop("trans_act", "linear")
        quat_act = kwargs.pop("quat_act", "linear")
        fl_act = kwargs.pop("fl_act", "relu")
        if kwargs:
            raise ValueError(f"Unknown kwargs: {kwargs.keys()}")
        return VggtCameraHead(
            dim_in=dim_in,
            dim_embed=dim_embed,
            trunk_depth=trunk_depth,
            pose_encoding_type=pose_encoding_type,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            init_values=init_values,
            trans_act=trans_act,
            quat_act=quat_act,
            fl_act=fl_act,
        )


    @classmethod
    def get_depth_head(
        cls,
        dim_in: int,
        dim_features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [1, 3, 5, 7],
        **kwargs,
    ) -> VggtDptHead:
        """
        Pseudo-code:
            DPT_FORWARD(multi_scale_tokens):
                # Select tokens from specific transformer layers
                multi_scale_features = SELECT_LAYERS(multi_scale_tokens, [4,11,17,23])
                
                # Project each scale to target channel dimensions
                for each feature in multi_scale_features:
                    feature = CONV2D_1x1(feature, dim_in → out_channels[i])
                
                # Adjust spatial resolutions to create pyramid
                for each feature in multi_scale_features:
                    if i == 0: feature = CONV_TRANSPOSE_4x(feature)      # finest
                    if i == 1: feature = CONV_TRANSPOSE_2x(feature)      # medium-fine
                    if i == 2: feature = IDENTITY(feature)               # medium-coarse
                    if i == 3: feature = CONV2D_3x3_STRIDE2(feature)     # coarsest
                
                # Process each pyramid level to common feature dimension
                for each feature in multi_scale_features:
                    feature = CONV2D_3x3(feature, out_channels[i] → dim_features)
                
                # Fuse pyramid levels from coarse to fine with upsampling
                fused = multi_scale_features[coarsest]
                for each finer_level in multi_scale_features[coarse_to_fine]:
                    fused = UPSAMPLE_2x_AND_ADD(fused, finer_level)
                
                # Generate final output
                output = CONV2D_3x3(fused, dim_features → output_dim)
                
                return output
        """
        kwargs = deepcopy(kwargs)
        pos_embed = kwargs.pop("pos_embed", True)
        down_ratio = kwargs.pop("down_ratio", 1)
        if kwargs:
            raise ValueError(f"Unknown kwargs: {kwargs.keys()}")

        return VggtDptHead(
            dim_in=dim_in,
            dim_features=dim_features,
            out_channels=out_channels,
            intermediate_layer_idx=intermediate_layer_idx,
            pose_embed=pos_embed,
            down_ratio=down_ratio,
            output_dim=2,
            activation="exp",
            conf_activation="expp1",
            feature_only=False,
        )


    @staticmethod
    def camera_loss(
        pred_pose_enc_list: List[Tensor], 
        gt_T_cw: Tensor,
        gt_K: Tensor,
        image_size_hw: Tuple[int, int] | None = None,
        point_masks: Tensor | None = None,
        **kwargs,
    ):
        B, V, _, _ = gt_T_cw.shape
        assert gt_T_cw.shape == (B, V, 4, 4), f"{tuple(gt_T_cw.shape)} == (B, V, 4, 4)"
        assert gt_K.shape == (B, V, 3, 3), f"{tuple(gt_K.shape)} == ({B}, {V}, 3, 3)"
        

        if point_masks is None and image_size_hw is None:
            raise ValueError("Either point_masks or image_size_hw must be provided")

        if image_size_hw is None:
            H, W = point_masks.shape[-2:]
        else:
            H, W = image_size_hw

        if point_masks is None and image_size_hw is not None:
            point_masks = torch.ones((B, V, H, W), dtype=torch.bool, device=gt_T_cw.device).expand(B, V, H, W)
        assert point_masks.shape == (B, V, H, W), f"{tuple(point_masks.shape)} == ({B}, {V}, {H}, {W})"

        batch = dict(
            extrinsics=gt_T_cw,
            intrinsics=gt_K,
            images=torch.zeros((B, V, 3, H, W), dtype=gt_T_cw.dtype, device=gt_T_cw.device),
            point_masks=point_masks,
        )

        kwargs = deepcopy(kwargs)
        loss_type=kwargs.pop("loss_type", "l1")
        gamma=kwargs.pop("gamma", 0.6)
        pose_encoding_type=kwargs.pop("pose_encoding_type", "absT_quaR_FoV")
        weight_T = kwargs.pop("weight_T", 1.0)
        weight_R = kwargs.pop("weight_R", 1.0)
        weight_fl = kwargs.pop("weight_fl", 0.5)
        frame_num = kwargs.pop("frame_num", -100)

        if kwargs:
            raise ValueError(f"Unknown kwargs: {kwargs.keys()}")

        return camera_loss(
            pred_pose_enc_list=pred_pose_enc_list, 
            batch=batch, 
            loss_type=loss_type, 
            gamma=gamma, 
            pose_encoding_type=pose_encoding_type, 
            weight_T=weight_T, 
            weight_R=weight_R, 
            weight_fl=weight_fl, 
            frame_num=frame_num,
        )


    # =============================================================================================
    # Utilities
    # =============================================================================================
    def preprocess_input(
        self,
        images: np.ndarray | Tensor, 
        *,
        mode: Literal["crop", "pad"] = "crop",
    ) -> Tensor:
        if isinstance(images, np.ndarray):
            assert images.shape[-1] == 3 and images.ndim > 3, f"{tuple(images.shape)}"
            images = torch.asarray(images).moveaxis(-1, -3)
        assert isinstance(images, torch.Tensor), f"{type(images)}"
        assert images.shape[-3] == 3 and images.ndim > 3, f"{tuple(images.shape)}"
        images = images.to(device=self.device)
        if images.dtype == torch.uint8:
            images = images / 255.0

        images = self._resize_images(images.to(dtype=torch.float32), mode=mode)

        if images.ndim == 4:
            images = images.unsqueeze(0)
        images = images.clamp(min=0.0, max=1.0)

        return images # [B, V, 3, H, W]


    @classmethod
    def resize_preds(
        cls, 
        preds: VggtPreds, 
        size: int | Tuple[int, int], 
        *, 
        original_size: int | Tuple[int, int] | None = None,
    ) -> VggtPreds:
        if isinstance(size, Integer):
            size = (size, size)
        if isinstance(original_size, Integer):
            original_size = (original_size, original_size)
        
        if original_size is None:
            original_size = cls._infer_HW_from_preds(preds)
        else:
            assert original_size == cls._infer_HW_from_preds(preds), f"{original_size} != {cls._infer_HW_from_preds(preds)}"

        # ---------------------------------------------------------------------------------------------
        # Resize inputs whose original size can be inferred.
        # ---------------------------------------------------------------------------------------------
        preds_resized = {}
        for k, v in preds.items():
            if k in ["depth_map", "point_map", "color"]:
                assert isinstance(v, Tensor), f"'{k}': {type(v)}"
                *batch_size, H, W, C = v.shape
                preds_resized[k] = F.interpolate(v.movedim(-1, -3).flatten(end_dim=-4), size=size, mode='bicubic', align_corners=False).movedim(-3, -1).reshape(tuple(batch_size) + size + (C,))

            elif k in ["depth_conf", "point_conf"]:
                assert isinstance(v, Tensor), f"'{k}': {type(v)}"
                *batch_size, H, W = v.shape
                preds_resized[k] = F.interpolate(v.flatten(end_dim=-3).unsqueeze(-3), size=size, mode='bicubic', align_corners=False).squeeze(-3).reshape(tuple(batch_size) + size)

            elif k == "K":
                assert isinstance(v, Tensor), f"'{k}': {type(v)}"
                try:
                    *batch_size, A, B = v.shape
                except ValueError:
                    raise ValueError(f"'{k}': {tuple(v.shape)}")
                assert A == B == 3, f"'{k}': {tuple(v.shape)}"

                v = v * torch.tensor([
                    [size[-1]/original_size[-1],         0.,                                         size[-1]/original_size[-1]],
                    [0.,                                      size[-2]/original_size[-2],            size[-2]/original_size[-2]],
                    [0.,                                      0.,                                         1.                             ]
                ], device=v.device, dtype=v.dtype)
                preds_resized[k] = v
            else:
                preds_resized[k] = v

        return preds_resized


    @classmethod
    def rescale_preds(
        cls,
        preds: VggtPreds,
        scale: Real | Tensor | Mapping | None,
        allow_inversion: bool = False,
    ) -> VggtPreds:
        if scale is None:
            T_cw = preds.get("T_cw", None)
            assert isinstance(T_cw, Tensor), f"{type(T_cw)}"
            denom = torch.einsum('...ij,...ij->...', T_cw[..., :3, 3], T_cw[..., :3, 3])
            scale: Tensor = 1 / torch.clamp(denom, min=torch.finfo(denom.dtype).eps)

        if isinstance(scale, Mapping):
            T_cw = preds.get("T_cw", None)
            T_cw_reference = scale.get("T_cw", None)
            assert isinstance(T_cw, Tensor), f"{type(T_cw)}"
            assert isinstance(T_cw_reference, Tensor), f"{type(T_cw_reference)}"

            num = torch.einsum('...ij,...ij->...', T_cw_reference[..., :3, 3], T_cw[..., :3, 3])
            if not allow_inversion:
                num = num.abs()
            denom = torch.einsum('...ij,...ij->...', T_cw[..., :3, 3], T_cw[..., :3, 3])
            scale: Tensor = num / torch.clamp(denom, min=torch.finfo(denom.dtype).eps)
            

        if isinstance(scale, Real):
            scale = torch.tensor(scale)
        else:
            assert isinstance(scale, Tensor), f"{type(scale)}"

        rescaled_preds = {}
        for k, v in preds.items():
            if k in ["depth_map", "point_map"]:
                rescaled_preds[k] = v * scale[..., None, None, None, None].to(v.device, v.dtype) # expand [V, H, W, C]
            elif k in ["T_cw"]:
                T_cw = torch.zeros_like(v)
                T_cw[..., -1, -1] = 1.0
                T_cw[..., :3, :3] = v[..., :3, :3]
                T_cw[..., :3, 3] = v[..., :3, 3] * scale[..., None, None].to(v.device, v.dtype)  # expand [V, 3]
                rescaled_preds[k] = T_cw
            else:
                rescaled_preds[k] = v.clone()
        return rescaled_preds


    # =============================================================================================
    # Private Methods
    # =============================================================================================
    @property
    def device(self):
        return next(self.vggt.parameters()).device

    @property
    def dtype(self):
        return next(self.vggt.parameters()).dtype


    def _forward_once(
        self, images: Tensor, 
        *, 
        preds: str | Iterable[str] = ["depth", "pose", "color"],
        frames_chunk_size: int | None = None,
    ) -> VggtPreds:
        if frames_chunk_size is None:
            frames_chunk_size = images.shape[1]

        for p in preds:
            if p not in ["depth", "pmap", "pose", "intermediates", "color"]:
                raise ValueError(f"Invalid prediction head: {p}")

        output = {}
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images)   # L x [B//chunk_size, T, nTokens, emb_dim * 2]
        aggregated_tokens_list: List[Tensor] = [tok.contiguous() for tok in aggregated_tokens_list]

        if "intermediates" in preds:
            output["intermediates"]: List[Tensor] = aggregated_tokens_list

        if "pose" in preds:
            pose_enc = self.vggt.camera_head(aggregated_tokens_list)[-1]
            T_cw, K = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            T_cw = torch.cat([T_cw, torch.zeros_like(T_cw[...,:1,:])], dim=-2)
            T_cw[..., -1, -1] = 1.0
            output["T_cw"]: Tensor = T_cw
            output["K"]: Tensor = K
        
        if "depth" in preds:
            depth_map, depth_conf = self.vggt.depth_head(aggregated_tokens_list, images, patch_start_idx, frames_chunk_size=frames_chunk_size)
            output["depth_map"]: Tensor = depth_map
            output["depth_conf"]: Tensor = depth_conf

        if "pmap" in preds:
            point_map, point_conf = self.vggt.point_head(aggregated_tokens_list, images, patch_start_idx, frames_chunk_size=frames_chunk_size)
            output["point_map"]: Tensor = point_map
            output["point_conf"]: Tensor = point_conf

        if "color" in preds:
            output["color"]: Tensor = images.movedim(-3, -1)

        return output


    @staticmethod
    def _infer_HW_from_preds(preds: VggtPreds, strict: bool = True) -> Tuple[int, int] | None:
        shapes = {}
        HWs = {}
        for k, v in preds.items():
            if k in ["depth_map", "point_map", "color"]:
                try:
                    H, W, C = v.shape[-3:]
                except ValueError:
                    raise ValueError(f"'{k}': {tuple(v.shape)}")
                assert C == 1 or C == 3, f"'{k}': {tuple(v.shape)}"
                HWs[k] = (H, W)
                shapes[k] = v.shape

            elif k in ["depth_conf", "point_conf"]:
                try:
                    H, W = v.shape[-2:]
                except ValueError:
                    raise ValueError(f"'{k}': {tuple(v.shape)}")
                assert v.shape[-1] != 1, f"'{k}': {tuple(v.shape)}"   # No unsqueeze for confidence map
                HWs[k] = (H, W)
                shapes[k] = v.shape

            else:
                pass
        HW = set(HWs.values())

        if len(HW) == 0:
            if strict:
                raise ValueError(f"HW cannot be inferred from {set(preds.keys())}")
            else:
                return None

        if len(HW) > 1:
            raise ValueError(f"HW is not consistent in {shapes}")
        HW = HW.pop()

        return HW


    @staticmethod
    def _visualize_scene(
        depth_map, depth_conf, T_cw, K, color, 
        conf_thres=10.0,
        show_up_axis=False,
        scene_scale: float | None = None,
    ) -> Scene:  
        with torch.no_grad():
            assert isinstance(depth_map, np.ndarray), f"{type(depth_map)}"
            assert isinstance(depth_conf, np.ndarray), f"{type(depth_conf)}"
            assert isinstance(color, np.ndarray), f"{type(color)}"
            assert isinstance(T_cw, np.ndarray), f"{type(T_cw)}"
            assert isinstance(K, np.ndarray), f"{type(K)}"


            S, H, W, C = depth_map.shape
            assert C == 1, f"{tuple(depth_map.shape)}"
            assert depth_conf.shape == (S, H, W), f"depth_conf.shape == (S, H, W) | {tuple(depth_conf.shape)} == {(S, H, W)}"
            assert color.shape == (S, H, W, 3), f"color.shape == (S, H, W, 3) | {tuple(color.shape)} == {(S, H, W, 3)}"
            assert T_cw.shape == (S, 4, 4) or T_cw.shape == (S, 3, 4), f"{T_cw.shape}"
            assert K.shape == (S, 3, 3), f"{K.shape}"


            viz_point_map = unproject_depth_map_to_point_map(depth_map, T_cw, K)
            viz_point_conf = depth_conf
            viz = {
                "world_points": viz_point_map,
                "world_points_conf": viz_point_conf,
                "images": color,
                "extrinsic": T_cw[..., :3, :],
            }

            scene_viz = predictions_to_glb(
                viz,
                conf_thres=conf_thres,
                show_up_axis=show_up_axis,
                scene_scale=scene_scale,
            )

        return scene_viz


    @staticmethod
    def _resize_images(
        images: Tensor, 
        *,
        mode: Literal["crop", "pad"] = "crop"
    ) -> Tensor:
        images = resize_images_tensor(images, mode=mode)
        return images # [B, V, 3, H, W]


    @staticmethod
    def _get_raw_camera_metrics(*, target_T_cw: Tensor, pred_T_cw: Tensor) -> tuple[dict[str, float], dict[str, float]]:
        with torch.no_grad():
            lower_the_better, higher_the_better = get_camera_metrics(target_T_cw=target_T_cw, pred_T_cw=pred_T_cw)
        return lower_the_better, higher_the_better


    @staticmethod
    def _replace_inf_with_mean(metric: Tensor, replace_even_if_all_nan: bool = False) -> Tensor:
        if metric.ndim != 1:
            raise NotImplementedError(f"{tuple(metric.shape)} != (B,)")

        isinf = metric == float('inf')
        if not replace_even_if_all_nan:
            if metric.isnan().all():
                return metric

        if not isinf.all():
            metric = torch.where(isinf, metric[~isinf].nanmean(), metric)
        return metric


    @staticmethod
    def _squash_log_tanh(
        metric: Tensor, 
        scale: float | None = 4.0,  # log(50) = 3.91 -> rel_metric saturates beyond [1/50, 50]
    ) -> Tensor:
        if scale is None:
            return torch.log(metric)
        return scale * torch.tanh(torch.log(metric) / scale)


    @staticmethod
    def voxel_filter(points: np.ndarray, 
                     features: np.ndarray, 
                     voxel_size: float, 
                     *,
                     coord_reduction: str = "average",
                     min_pts: int = 1,
                     device = "cpu",
                     ) -> tuple[np.ndarray, np.ndarray]:
        points = torch.from_numpy(points).to(device)
        features = torch.from_numpy(features).to(device)

        assert points.device == features.device, f"{points.device} != {features.device}"
        device = points.device
        mins = points.min(dim=-2).values

        vox_idx = torch.div((points - mins), voxel_size, rounding_mode='trunc').type(torch.long)
        shape = vox_idx.max(dim=-2).values + 1
        raveled_idx = torch.tensor(np.ravel_multi_index(vox_idx.T.cpu().numpy(), shape.cpu().numpy()), device = device, dtype=vox_idx.dtype)

        n_pts_per_vox = torch_scatter.scatter(torch.ones_like(raveled_idx, device=device), raveled_idx, dim_size=shape[0]*shape[1]*shape[2])
        nonzero_vox = (n_pts_per_vox >= min_pts).nonzero()
        n_pts_per_vox = n_pts_per_vox[nonzero_vox].squeeze(-1)

        feature_vox = torch_scatter.scatter(features, raveled_idx.unsqueeze(-1), dim=-2, dim_size=shape[0]*shape[1]*shape[2])[nonzero_vox].squeeze(-2)
        feature_vox /= n_pts_per_vox.unsqueeze(-1)

        if coord_reduction == "center":
            coord_vox = np.stack(np.unravel_index(nonzero_vox.cpu().numpy().reshape(-1), shape.cpu().numpy()), axis=-1)
            coord_vox = torch.tensor(coord_vox, device = device, dtype=vox_idx.dtype)
            coord_vox = coord_vox * voxel_size + mins + (voxel_size/2)
        elif coord_reduction == "average":
            coord_vox = torch_scatter.scatter(points, raveled_idx.unsqueeze(-1), dim=-2, dim_size=shape[0]*shape[1]*shape[2])[nonzero_vox].squeeze(-2)
            coord_vox /= n_pts_per_vox.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown coordinate reduction method: {coord_reduction}")

        return coord_vox.detach().cpu().numpy(), feature_vox.detach().cpu().numpy()
