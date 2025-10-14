# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Iterable, Optional

from hydra.utils import instantiate
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset, Sampler

from .worker_fn import get_worker_init_fn

def build_dynamic_dataloader(
    dataset: dict,
    common_config: dict,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
    drop_last: bool = True,
    collate_fn: Optional[Callable] = None,
    worker_init_fn: Optional[Callable] = None,
    persistent_workers: bool = False,
    seed: int = 42,
    epoch: int = 0,
    max_img_per_gpu: int = 48,
):
    """
    Constructs a dynamic data loader with the specified configuration.

    This data loader dynamically adjusts batch size, aspect ratio, and the number of images
    during training.

    Args:
        dataset (dict): Configuration dictionary for the dataset to be instantiated.
        common_config (dict): Shared configuration parameters.
        num_workers (int): Number of subprocesses for data loading.
        shuffle (bool): Whether to shuffle the dataset.
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory.
        drop_last (bool): If True, drop the last incomplete batch.
        collate_fn (Optional[Callable]): Function to merge samples into a batch.
        worker_init_fn (Optional[Callable]): Function to initialize workers.
        persistent_workers (bool): If True, workers will not be shut down after a dataset has been consumed.
        seed (int): Random seed for reproducibility.
        epoch (int): Current epoch number.
        max_img_per_gpu (int): Maximum number of images to fit in GPU memory.

    Returns:
        DataLoader: A PyTorch DataLoader configured with dynamic batch sampling.
    """
    print("Building dynamic dataloader with seed:", seed)
    
    # Instantiate the dataset
    dataset_obj = instantiate(dataset, common_config=common_config, _recursive_=False)

    # Extract aspect ratio and image number ranges from the configuration
    aspect_ratio_range = common_config.aspects  # e.g., [0.5, 1.0]
    image_num_range = common_config.img_nums    # e.g., [2, 24]
    
    # Validate the aspect ratio and image number ranges
    if len(aspect_ratio_range) != 2 or aspect_ratio_range[0] >= aspect_ratio_range[1]:
        raise ValueError(f"aspect_ratio_range must be [min, max] with min < max, got {aspect_ratio_range}")
    if len(image_num_range) != 2 or image_num_range[0] < 1 or image_num_range[0] >= image_num_range[1]:
        raise ValueError(f"image_num_range must be [min, max] with 1 <= min < max, got {image_num_range}")
    
    # Create samplers
    sampler = DynamicDistributedSampler(dataset_obj, seed=seed, shuffle=shuffle)
    batch_sampler = DynamicBatchSampler(
        sampler, 
        aspect_ratio_range, 
        image_num_range, 
        seed=seed,
        max_img_per_gpu=max_img_per_gpu
    )
    
    # Set the epoch for the sampler
    sampler.set_epoch(epoch)
    if hasattr(dataset_obj, "epoch"):
        dataset_obj.epoch = epoch
    if hasattr(dataset_obj, "set_epoch"):
        dataset_obj.set_epoch(epoch)

    # Create and return the dataloader
    return DataLoader(
        dataset_obj,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        worker_init_fn=get_worker_init_fn(
            seed=seed,
            num_workers=num_workers,
            epoch=epoch,
            worker_init_fn=worker_init_fn,
        ),
    )



class DynamicBatchSampler(Sampler):
    """
    A custom batch sampler that dynamically adjusts batch size, aspect ratio, and image number
    for each sample. Batches within a sample share the same aspect ratio and image number.
    """
    def __init__(self,
                 sampler,
                 aspect_ratio_range,
                 image_num_range,
                 epoch=0,
                 seed=42,
                 max_img_per_gpu=48):
        """
        Initializes the dynamic batch sampler.

        Args:
            sampler: Instance of DynamicDistributedSampler.
            aspect_ratio_range: List containing [min_aspect_ratio, max_aspect_ratio].
            image_num_range: List containing [min_images, max_images] per sample.
            epoch: Current epoch number.
            seed: Random seed for reproducibility.
            max_img_per_gpu: Maximum number of images to fit in GPU memory.
        """
        self.sampler = sampler
        self.aspect_ratio_range = aspect_ratio_range
        self.image_num_range = image_num_range
        self.rng = random.Random()
        
        # Uniformly sample from the range of possible image numbers
        # For any image number, the weight is 1.0 (uniform sampling). You can set any different weights here.
        self.image_num_weights = {num_images: 1.0 for num_images in range(image_num_range[0], image_num_range[1]+1)}

        # Possible image numbers, e.g., [2, 3, 4, ..., 24]
        self.possible_nums = np.array([n for n in self.image_num_weights.keys()
                                       if self.image_num_range[0] <= n <= self.image_num_range[1]])
        
        # Normalize weights for sampling
        weights = [self.image_num_weights[n] for n in self.possible_nums]
        self.normalized_weights = np.array(weights) / sum(weights)

        # Maximum image number per GPU
        self.max_img_per_gpu = max_img_per_gpu

        # Set the epoch for the sampler
        self.set_epoch(epoch + seed)

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler, affecting the random sequence.

        Args:
            epoch: The epoch number.
        """
        self.sampler.set_epoch(epoch)
        self.epoch = epoch
        self.rng.seed(epoch * 100)

    def __iter__(self):
        """
        Yields batches of samples with synchronized dynamic parameters.

        Returns:
            Iterator yielding batches of indices with associated parameters.
        """
        sampler_iterator = iter(self.sampler)

        while True:
            try:
                # Sample random image number and aspect ratio
                random_image_num = int(np.random.choice(self.possible_nums, p=self.normalized_weights))
                random_aspect_ratio = round(self.rng.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1]), 2)

                # Update sampler parameters
                self.sampler.update_parameters(
                    aspect_ratio=random_aspect_ratio,
                    image_num=random_image_num
                )

                # Calculate batch size based on max images per GPU and current image number
                batch_size = self.max_img_per_gpu / random_image_num
                batch_size = np.floor(batch_size).astype(int)
                batch_size = max(1, batch_size)  # Ensure batch size is at least 1

                # Collect samples for the current batch
                current_batch = []
                for _ in range(batch_size):
                    try:
                        item = next(sampler_iterator)  # item is (idx, aspect_ratio, image_num)
                        current_batch.append(item)
                    except StopIteration:
                        break  # No more samples

                if not current_batch:
                    break  # No more data to yield

                yield current_batch

            except StopIteration:
                break  # End of sampler's iterator

    def __len__(self):
        # Return a large dummy length
        return 1000000


class DynamicDistributedSampler(DistributedSampler):
    """
    Extends PyTorch's DistributedSampler to include dynamic aspect_ratio and image_num
    parameters, which can be passed into the dataset's __getitem__ method.
    """
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )
        self.aspect_ratio = None
        self.image_num = None

    def __iter__(self):
        """
        Yields a sequence of (index, image_num, aspect_ratio).
        Relies on the parent class's logic for shuffling/distributing
        the indices across replicas, then attaches extra parameters.
        """
        indices_iter = super().__iter__()

        for idx in indices_iter:
            yield (idx, self.image_num, self.aspect_ratio,)

    def update_parameters(self, aspect_ratio, image_num):
        """
        Updates dynamic parameters for each new epoch or iteration.

        Args:
            aspect_ratio: The aspect ratio to set.
            image_num: The number of images to set.
        """
        self.aspect_ratio = aspect_ratio
        self.image_num = image_num
