import torch
from torch import Tensor
from .vggt_loss import camera_to_rel_deg

def scale_procrustes(target: Tensor, pred: Tensor) -> tuple[Tensor, Tensor]:
    """
    Rescales pred to target.
    Not actually doing any procrustes alignment, but the name is kept for legacy reasons.
    """
    num = torch.einsum('...ij,...ij->...', target, pred)
    denom = torch.einsum('...ij,...ij->...', pred, pred)
    # num = torch.trace(torch.matmul(pred.transpose(-2, -1), target), dim1=-2, dim2=-1)
    # denom = torch.trace(torch.matmul(pred.transpose(-2, -1), pred), dim1=-2, dim2=-1)

    s = num / torch.clamp(denom, min=1.0e-8)

    s = s.detach()

    return target, s[..., None, None] * pred 


def procrustes_loss(target: Tensor, pred: Tensor) -> Tensor:
    target_scaled, pred_scaled = scale_procrustes(target, pred)

    lossT = torch.mean(torch.abs(target_scaled - pred_scaled), dim=(-2, -1))

    return lossT 


def calculate_normalized_histogram(errors, max_threshold=30):
    assert errors.ndim == 1, f"errors.ndim: {errors.ndim}"

    # Define histogram bins
    bins = torch.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram = torch.histc(errors, bins=max_threshold + 1, min=0, max=max_threshold) # shape: (max_threshold + 1,)

    # Normalize the histogram
    num_pairs = float(len(errors))
    normalized_histogram = histogram / num_pairs

    return normalized_histogram


@torch.no_grad()
def get_camera_metrics(*, target_T_cw: Tensor, pred_T_cw: Tensor) -> dict[str, Tensor]:
    target_T_cw = torch.clone(target_T_cw).double()
    pred_T_cw = torch.clone(pred_T_cw).double()
    B, V, _, _ = target_T_cw.shape


    # ---------------------------------------------------------------
    # Lower the better metrics
    # ---------------------------------------------------------------
    lower_the_better = {}
    traj_err_scale_invariant = procrustes_loss(target=torch.linalg.inv(target_T_cw)[..., :3, 3].flatten(end_dim=-3), pred=torch.linalg.inv(pred_T_cw)[..., :3, 3].flatten(end_dim=-3))
    traj_err_unscaled = torch.mean(torch.abs(torch.linalg.inv(pred_T_cw)[..., :3, 3] - torch.linalg.inv(target_T_cw)[..., :3, 3]), dim=(-2, -1))
    lower_the_better["traj_err_scale_invariant"] = traj_err_scale_invariant
    lower_the_better["traj_err_unscaled"] = traj_err_unscaled


    # ---------------------------------------------------------------
    # Higher the better metrics
    # ---------------------------------------------------------------
    higher_the_better = {}
    rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(pred_cameras=pred_T_cw, gt_cameras=target_T_cw)
    rel_rangle_deg = rel_rangle_deg.reshape(B,V*(V-1)//2)
    rel_tangle_deg = rel_tangle_deg.reshape(B,V*(V-1)//2)

    if rel_rangle_deg.numel() == 0 and rel_tangle_deg.numel() == 0:
        rel_rangle_deg = torch.FloatTensor([0]).to(target_T_cw.device).to(target_T_cw.dtype)
        rel_tangle_deg = torch.FloatTensor([0]).to(target_T_cw.device).to(target_T_cw.dtype)

    thresholds = [3, 5, 10, 15, 20, 30]
    for threshold in thresholds:
        higher_the_better[f"Rac_{threshold}"] = (rel_rangle_deg < threshold).double().mean(dim=-1)
        higher_the_better[f"Tac_{threshold}"] = (rel_tangle_deg < threshold).double().mean(dim=-1)

    rot_hist_normalized = []
    trans_hist_normalized = []
    se3_hist_normalized = []
    for b in range(B):
        rot_hist_normalized.append(calculate_normalized_histogram(rel_rangle_deg[b], max_threshold=30))
        trans_hist_normalized.append(calculate_normalized_histogram(rel_tangle_deg[b], max_threshold=30))
        se3_hist_normalized.append(calculate_normalized_histogram(torch.stack([rel_rangle_deg[b], rel_tangle_deg[b]], dim=-1).max(dim=-1).values, max_threshold=30))

    rot_hist_normalized = torch.stack(rot_hist_normalized, dim=0)      # normalized to [0, 1]. Shape: (B, 31)
    trans_hist_normalized = torch.stack(trans_hist_normalized, dim=0)  # normalized to [0, 1]. Shape: (B, 31)
    se3_hist_normalized = torch.stack(se3_hist_normalized, dim=0)      # normalized to [0, 1]. Shape: (B, 31)

    auc_thresholds = [3, 5, 10, 15, 20, 30]
    for auc_threshold in auc_thresholds:
        higher_the_better[f"Rot_Auc_{auc_threshold}"] = torch.cumsum(rot_hist_normalized[:, :auc_threshold], dim=-1).mean(dim=-1)
        higher_the_better[f"Trans_Auc_{auc_threshold}"] = torch.cumsum(trans_hist_normalized[:, :auc_threshold], dim=-1).mean(dim=-1)
        higher_the_better[f"SE3_Auc_{auc_threshold}"] = torch.cumsum(se3_hist_normalized[:, :auc_threshold], dim=-1).mean(dim=-1)

    return lower_the_better, higher_the_better