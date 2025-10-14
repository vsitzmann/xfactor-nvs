# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import Literal

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as TF



def resize_images_tensor(
    image_tensors: Tensor, mode: Literal["crop", "pad"] = "crop", 
    target_size: int = 518, patch_size: int = 14,
) -> Tensor:
    """
    Preprocesses a batch of image tensors for model input using vectorized operations.
    This function is the tensor-based equivalent of `load_and_preprocess_images`.
    It assumes all tensors in the batch have the same height and width.

    Args:
        image_tensors (torch.Tensor): A tensor of images with shape (..., C, H, W).
                                      Can handle arbitrary leading dimensions.
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                              - "crop" (default): Sets width to 518px and center crops
                                                height if needed.
                              - "pad": Preserves all pixels by making the largest dimension 518px
                                       and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (..., C, H_out, W_out),
                      where H_out and W_out are the new dimensions after processing.

    Raises:
        ValueError: If the input tensor has fewer than 3 dimensions or if the mode is invalid.
    """
    assert isinstance(image_tensors, Tensor), f"{type(image_tensors)}"
    # Validate input tensor dimensions
    if image_tensors.dim() < 3:
        raise ValueError("Input tensor must have at least 3 dimensions (C, H, W)")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    # --- Vectorized Implementation ---
    
    # Store original shape and get input dimensions
    original_shape = image_tensors.shape
    *batch_dims, C, H, W = original_shape

    # --- 1. Calculate new dimensions (as scalars) ---
    # Because the input is a tensor, all images have the same H and W.
    if mode == "pad":
        if W >= H:
            new_width = target_size
            new_height = round(H * (new_width / W) / patch_size) * patch_size
        else:
            new_height = target_size
            new_width = round(W * (new_height / H) / patch_size) * patch_size
    else:  # mode == "crop"
        scale = target_size / min(H, W)
        new_height = int(math.ceil((H * scale) / patch_size) * patch_size)
        new_width = int(math.ceil((W * scale) / patch_size) * patch_size)
    
    # Ensure dimensions are integers for interpolation
    new_height, new_width = int(new_height), int(new_width)

    # --- 2. Resize the entire batch ---
    flat_tensors = image_tensors.reshape(-1, C, H, W)
    resized_tensors = F.interpolate(
        flat_tensors,
        size=(new_height, new_width),
        mode="bicubic",
        align_corners=False,
    )

    # --- 3. Apply crop or pad to the entire batch ---
    processed_tensors = resized_tensors
    if mode == "crop":
        _, _, current_h, current_w = resized_tensors.shape
        start_y = max(0, (current_h - target_size) // 2)
        start_x = max(0, (current_w - target_size) // 2)
        processed_tensors = resized_tensors[:, :, start_y : start_y + target_size, start_x : start_x + target_size]

    elif mode == "pad":
        # Get current H/W after resize
        _, _, current_h, current_w = resized_tensors.shape
        h_padding = target_size - current_h
        w_padding = target_size - current_w
        
        if h_padding > 0 or w_padding > 0:
             pad_top = h_padding // 2
             pad_bottom = h_padding - pad_top
             pad_left = w_padding // 2
             pad_right = w_padding - pad_left
             processed_tensors = F.pad(
                resized_tensors, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
             )

    # --- 4. Restore original batch dimensions ---
    # Get the final C, H, W from the processed tensor
    _, final_c, final_h, final_w = processed_tensors.shape
    # Combine original batch dims with new C, H, W
    final_shape = batch_dims + [final_c, final_h, final_w]
    output_tensor = processed_tensors.reshape(final_shape)

    return output_tensor