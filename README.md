# XFactor: True Self-Supervised Novel View Synthesis is Transferable 
Official code release for XFactor, enabling transferable self-supervised novel view synthesis.


<p align="center">  
    <a href="https://www.mitchel.computer/">Thomas W. Mitchel</a>,
    <a href="https://sites.google.com/view/hyunwooryu">Hyunwoo Ryu</a>,
    <a href="https://www.vincentsitzmann.com/">Vincent Sitzmann</a>
</p>


</div>


<div align="center">
    <a href="https://www.mitchel.computer/xfactor/"><strong>Project Page</strong></a> |
    <a href="http://arxiv.org/abs/2510.13063"><strong>Paper</strong></a> 
</div>

<br>

This code base is implemented in JAX. An official Pytorch implementation is forthcoming. 
## Installation
We reccommend creating a dedicated environment before installing the requirements.
```
python3 -m venv XFactor
source XFactor/bin/activate
pip install -r requirements.txt
```

## Data 
Our models are trained on a combined video dataset consisting of all of RE10K, DL3DV, MVImgNet, and CO3Dv2. We use `grain` for dataloading. Our dataloader is contained in the `data/video_dataset.py` file. Our dataloader is designed to stream video frames: Each dataset is stored as a list of filepaths, where each entry corresponds to the location of a video. Our dataloader makes some assumptions about the file structure for each dataset based on our internal copies of these datasets. You may need to modify the syntax to align with your copies of these datasets. 

The train/test splits we use in the paper are slightly different than what will be produced by the dataloader here due to modifications we made after the fact to ensure consistency across systems.  The JSON files containing our train/test splits for each dataset as used in the paper can be downloaded [here](https://www.dropbox.com/scl/fi/q6qnmd7b7816suprbv8d7/paper_splits.zip?rlkey=y40aqidrw09tbkpxu0n02rh3v&st=o6u5s90x&dl=0).  Note that the sequence IDs are the dictionary keys.  

## Training
Training scripts and configs for stereo-monocular and multi-view XFactor are provided in the `train` and `train/configs` directories. The maximum baseline (maximal distance between the start and end frames in a sequence) for each dataset is given by the `max_win_len_end` keys in the `DATASETS` dictionary.  

The directory containing each dataset must be specified by setting the `RE10K_PATH`, `DL3DV_PATH`, `MVIMGNET_PATH`, and `CO3DV2_PATH` variables in each config file. 

The stereo-monocular model (with batch size 256) was trained on two NVIDIA H200 GPUs. It can also be trained on at least four NVIDIA A100 GPUs. The multi-view model was trained on eight H200 GPUs. If the models are too big to fit on your compute, we reccomend setting the `BATCH_SIZE` variable in the config files to a smaller divisor of `TRUE_BATCH_SIZE`, e.g. `BATCH_SIZE = 64`. In this regime, the training scripts will automatically perform gradient accumulation to simulate a batch size of 256. 

The default setting for the multi-view config trains with five context views, i.e. `MAX_VIEWS = 5`. This can be changed as desired. 

## Checkpoints 
We provide pre-trained checkpoints for both stereo-monocular and multi-view XFactor. These checkpoints are slightly different than the ones use in the paper for evaluation, as they are trained on the data split produced by the dataloader here (see above). 

| Model | 
| ----- | 
| [Stereo-Monocular XFactor](https://www.dropbox.com/scl/fi/4cuw2esx0ofjgpqj4cbiz/stereo_monocular.zip?rlkey=ixdaaqzae9heh0i2tc4wxvz24&st=qanaz3d1&dl=0) | 
| [Multi-View XFactor](https://www.dropbox.com/scl/fi/3qkm7knlkznfajg4w75i5/multiview.zip?rlkey=lsk7islgffgaa2dxwshfvpxqc&st=zg6cexiv&dl=0) | 

Download and unzip these files into a desired directory. Then, the paramaters can be loaded as follows
```
from clu import checkpoint

params_sm = checkpoint.load_state_dict("/your/directory/stereo_monocular/checkpoints-0")["params"]
params_mv = checkpoint.load_state_dict("/your/directory/multiview/checkpoints-0")["params"]
```

Note that the `Render` module in multi-view XFactor is trained with five context views so inference will be most effective with five context views.

## Usage 

```
import jax
import jax.numpy as jnp
import flax 
from clu import checkpoint
from nn import models

PoseEnc = models.PoseEnc(features=1024,
                         num_heads=16,
                         patch_size=16,
                         num_layers=8,
                         pose_dim=256)

Render = models.Render(features=1024,
                         num_heads=16,
                         patch_size=16,
                         num_layers=8)


## Stereo-monocular model
params_sm = checkpoint.load_state_dict(stereo_monocular_checkpoint_dir)["params"]

# Target pairs from which to extract target pose
# B x 2 x 256 x 256 x 3, jnp.float32 array normalized to [-1, 1]
target_images = ...

# Context images used to render new view w/ target pose 
# B x 1 x 256 x 256 x 3, jnp.float32 array normalized to [-1, 1]
context_images = ...

# Compute target pose latent
# B x 2 x 256, P[:, 0, :] is zeros.
ZT = PoseEnc.apply({"params":params_sm["PoseEnc"]}, target_images)

# Render using context images and target poses
# B x 1 x 256 x 256 x 3, jnp.float32 array normalized to [-1, 1]
out = Render.apply({"params":params_sm["Render"]}, context_images, ZT)


## Multi-view model
params_mv = checkpoint.load_state_dict(multiview_checkpoint_dir)["params"]

# Target pair from which to extract target pose
# B x 2 x 256 x 256 x 3, jnp.float32 array normalized to [-1, 1]
target_images = ...

# Context sequence
# B x 5 x 256 x 256 x 3, jnp.float32 array normalized to [-1, 1]
# This sequence is ordered such that the "middlest" frame in the sequence is the reference image and has index [:, 0, ...]
# E.g. if you load frames [0, 1, 2, 3, 4] as context images, the input context sequence should be ordered [2, 0, 1, 3, 4]
# See proximity_permutation in train/train_mv.py for handling irregular spacing
context_sequence = ...

# Compute target pose latent
# B x 2 x 256, P[:, 0, :] is zeros
ZT = PoseEnc.apply({"params":params_mv["PoseEnc"]}, target_images)

# Compute poses of context sequence (requires a bit of reshape + tile jiu jitsu if you want to vectorize it)

# Form context pairs
# B*5 x 2 x 256 x 256 x 3
context_pairs = jnp.stack((jnp.tile(context_sequence[:, 0, None, ...], (1, 5, 1, 1, 1)), context_sequence[:, 1:, ...]), axis=2)
context_pairs = jnp.concatenate( jnp.split(context_pairs, 5, axis=1), axis=0)[:, 0, ...]

# Compute poses
# P: B x 5 x 256
# Reference pose is at index [:, 0, :] and is always zeros
ZC = PoseEnc.apply({"params":params["PoseEnc"]}, context_pairs)
ZC = jnp.stack(jnp.split(ZC, 5, axis=0), axis=1)
ZC = jnp.concatenate((ZC[:, 0, ...], ZC[:, 1:, 1, ...]), axis=1)

# Render target pose in context sequence
# B x 1 x 256 x 256 x 3, jnp.float32 array normalized to [-1, 1]
out = Render.apply({"params":params_mv["Render"]}, context_sequence, jnp.concatenate((ZC, ZT[:, 1, None, ...]), axis=1))
```

## True Pose Similarity (TPS) Metric 
We also provide code to compute our proposed True Pose Similarity (TPS) metric which quantifies the transferability of an NVS model. Here, our implementation uses VGGT as the oracle. Please see the `TPS_example.ipynb` notebook for example usage.

 
