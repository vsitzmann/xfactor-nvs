"""
Train.


Usage:
  train [options]
  train (-h | --help)
  train --version


Options:
  -h --help               Show this screen.
  --version               Show version.
  -i, --in=<input_dir>    Input directory  [default: {root_path}/weights/stereo_monocular/]. 
  -o, --out=<output_dir>  Output directory [default: {root_path}/weights/].
  -c, --config=<config>   Config directory [default: config_mv].
  -n, --name=<name>       Experiment name  
"""


import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

## Flags that help when paralellizing over > 2 GPUS 

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"

import sys
import functools
import importlib.util
import ml_collections
import numpy as np
import matplotlib
import matplotlib.cm as cm 
import wandb
import scipy as sp
from skimage.transform import resize
from sklearn.decomposition import PCA
import grain
import multiprocessing as mp


#jax.config.update("jax_enable_x64", True)


from docopt import docopt


from typing import Any, Callable, Dict, Sequence, Tuple, Union
from clu import checkpoint, metric_writers, metrics, parameter_overview, periodic_actions
from tqdm import tqdm
from icecream import ic

from os.path import dirname, abspath
ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
sys.path.append( ROOT_PATH )


from data import video_dataset as vd 
                 
# Import all jax stuff after multiprocessing
import flax
import flax.jax_utils as flax_utils
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax 
from lpips_j.lpips import LPIPS
from nn import losses, nutils, models


PI = np.pi 


# Constants
PMAP_AXIS = "batch" 


clr_norm   = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0, clip=True)
clr_mapper_D = cm.ScalarMappable(norm=clr_norm, cmap="inferno")


######################


@flax.struct.dataclass
class TrainState:
  step        : int
  opt_state   : Any
  params      : Any
  lpips_params: Any 
  key         : Any
  multi_steps : Any 
  train_metrics: Any
  eval_metrics: Any

def init_metrics(metric_names):
  metrics = {}
  for k in metric_names:
    metrics[k] = jnp.array(0.0)
  return metrics
  
train_metrics_names = ["train_loss", 
                       "train_l1_loss", 
                       "train_lpips_loss",
                       "mean_grads", 
                       "max_grads",
                       "mean_updates",
                       "max_updates",
                       "count"]

eval_metrics_names = ["eval_loss", 
                     "eval_l1_loss", 
                     "eval_lpips_loss",
                     "count"]


def _parse_losses(loss_dict, num_multi_steps=1):
    loss_schedule = {}

    for k in loss_dict.keys():
        weight = loss_dict[k]["value"]
        warmup = loss_dict[k]["warmup_steps"]
        s = loss_dict[k]["s"]

        if weight <= 1.0e-18:
            # No need for any schedule — always 0
            w_fn = optax.constant_schedule(0.0)

        elif warmup is None:
            # Always use the full weight
            w_fn = optax.constant_schedule(weight)

        else:
            if s is None or s == "linear":
                w_fn = lambda t, w=weight, wu=warmup: optax.linear_schedule(
                    init_value=0.0,
                    end_value=w,
                    transition_steps=wu
                )(t // num_multi_steps)

            elif s == "jump":
                # Step-function jump from 0.0 → weight after warmup steps
                w_fn = lambda t, w=weight, wu=warmup: optax.join_schedules(
                    [optax.constant_schedule(0.0), optax.constant_schedule(w)],
                    [wu]
                )(t // num_multi_steps)

            else:
                raise ValueError(f"Unknown schedule type: {s}")

        loss_schedule[k] = w_fn

    return loss_schedule



def create_train_state( cfg: Any, data_shape: Tuple, in_dir: str = None ) -> Tuple[nn.Module, nn.Module, Any, Any, Any, Any, Any, TrainState]:


  # Random key 
  seed = 0 #np.random.randint(low=0, high=1e8, size=(1, ))[0]
  
  key                              = jax.random.PRNGKey( seed )
  key, pose_key, render_key, train_key = jax.random.split( key, 4 )


  
  PoseEnc = models.PoseEnc(features=cfg.FEATURES,
                        num_heads=cfg.NUM_HEADS,
                        patch_size=cfg.PATCH_SIZE,
                        num_layers=cfg.POSE_LAYERS,
                        pose_dim=cfg.POSE_DIM)
  
  Render = models.Render(features=cfg.FEATURES,
                         num_heads=cfg.NUM_HEADS,
                         patch_size=cfg.PATCH_SIZE,
                         num_layers=cfg.RENDER_LAYERS)
  

  if cfg.MIXED:
    dtype = jnp.bfloat16
  else:
    dtype = jnp.float32


  # Dummy tensors for init 
  im_encode = jnp.ones( (data_shape[0], 2, cfg.INPUT_SIZE[0], cfg.INPUT_SIZE[1], 3), dtype=dtype)
  
  im_decode  = jnp.ones( (data_shape[0], cfg.MAX_VIEWS, cfg.INPUT_SIZE[0], cfg.INPUT_SIZE[1], 3), dtype=dtype)
  
  pose_input = jnp.ones((data_shape[0], cfg.MAX_VIEWS+1, 2, cfg.POSE_DIM), dtype=dtype)

  pmask_input = jnp.ones((data_shape[0], (cfg.INPUT_SIZE[0] // cfg.PATCH_SIZE) * (cfg.INPUT_SIZE[1] // cfg.PATCH_SIZE)), dtype=jnp.int32)
  
  # Initialize
  pose_params   = PoseEnc.init( pose_key, im_encode, pmask_input)["params"]
  render_params    = Render.init( render_key, im_decode, pose_input, pmask_input)["params"]

  # Load stereo-monocular weights
  if in_dir is not None:
    pmodel_dict = checkpoint.load_state_dict(in_dir)
    pose_params = pmodel_dict['params']['PoseEnc']
    dec_params = pmodel_dict['params']['Render']
    
  params        = {"PoseEnc": pose_params, "Render": render_params} 

  
  
  # Set up optimizer 
  schedule = optax.warmup_cosine_decay_schedule( init_value   = cfg.INIT_LR,
                                                 peak_value   = cfg.LR,
                                                 warmup_steps = cfg.WARMUP_STEPS,
                                                 decay_steps  = cfg.NUM_TRAIN_STEPS,
                                                 end_value    = cfg.END_LR )


  batch_size  = cfg.BATCH_SIZE
  multi_steps = cfg.TRUE_BATCH_SIZE // batch_size
  
  weight_decay_mask = models.mask_exclude_special_params(params)

  optim = optax.chain(
  optax.clip_by_global_norm(1.0),
  optax.adamw(learning_rate=schedule, b1=cfg.ADAM_B1, b2=cfg.ADAM_B2, weight_decay=cfg.WEIGHT_DECAY, mask=weight_decay_mask),
  )
  optimizer = optax.MultiSteps( optim, multi_steps )
  
  # LPIPS
  lpips = LPIPS()
  lpips_params = lpips.init(pose_key, jnp.zeros((data_shape[0], cfg.INPUT_SIZE[0], cfg.INPUT_SIZE[1], 3)), jnp.zeros((data_shape[0], cfg.INPUT_SIZE[0], cfg.INPUT_SIZE[1], 3)))


  loss_schedule = _parse_losses(cfg.L_WEIGHTS, multi_steps)

  # Data augmentations
  if cfg.AUG["pose"]:
    
    pose_aug_map = lambda key, x: nutils.jitter_and_blur(key, 
                                                        x, 
                                                        p_apply_jit=cfg.AUG["jitter"]["p_apply"],
                                                        p_apply_blur=cfg.AUG["blur"]["p_apply"],
                                                        d_bright=cfg.AUG["jitter"]["d_bright"], 
                                                        d_cont=cfg.AUG["jitter"]["d_cont"], 
                                                        d_sat=cfg.AUG["jitter"]["d_sat"], 
                                                        d_hue=cfg.AUG["jitter"]["d_hue"], 
                                                        min_sigma=cfg.AUG["blur"]["min_sigma"], 
                                                        max_sigma=cfg.AUG["blur"]["max_sigma"], 
                                                        kernel_size=cfg.AUG["blur"]["kernel_size"],
                                                        time_invar=True)
  else:
    pose_aug_map = None 


  
  state       = optimizer.init( params ) 
  train_state = TrainState( step=0, opt_state=state, params=params, lpips_params=lpips_params, key=key, multi_steps=multi_steps,
                           train_metrics=init_metrics(train_metrics_names), eval_metrics=init_metrics(eval_metrics_names))


  return PoseEnc, Render, optimizer, loss_schedule, lpips, pose_aug_map, train_key, train_state



def proximity_permutation(x):
    """
    Args:
        x: jnp.ndarray of shape (B, N), each row is a set of integer indices

    Returns:
        jnp.ndarray of shape (B, N), where each row is a permutation of indices (0..N-1)
        such that x[b, perm[b]] reorders each row of x according to distance to its medoid.
    """
    def process_row(row):
        # Compute max distance to endpoints for each element
        endpoint_dists = jnp.maximum(jnp.abs(row - row[0]), jnp.abs(row - row[-1]))
        # Sort indices ascending by distance to endpoints
        perm = jnp.argsort(endpoint_dists)
        return perm

    return jax.vmap(process_row)(x)

def permute(key, A):
  #import jax 
  
  def _perm(key, a):
    return jax.random.permutation(key, a)

  p_key = jax.random.split(key, A.shape[0])
  
  return jax.vmap(_perm, in_axes=(0, 0), out_axes=0)(p_key, A)


def set_diff(A, B, size):

  def _diff(a, b):
    return jnp.setdiff1d(a, b, size=size)

  return jax.vmap(_diff, in_axes=(0, 0), out_axes=0)(A, B)

def select_indices(key, t_ind):
    
  ind = proximity_permutation(t_ind)
  ind = ind.at[:, 1:].set(permute(key, ind[:, 1:]))
  
  
  return ind 
  

def train_step(x: Any,  t_ind: Any, PoseEnc: nn.Module, Render: nn.Module, state: TrainState,
               optimizer: Any, lpips: Any, w_dict: Any, patch_size: int, 
               pose_aug_map: Any=None, train: bool = True,  mixed=True):


  if mixed:
    dtype = jnp.bfloat16
  else:
    dtype = jnp.float32 
    
  key    = state.key 
  s0 = state.step
  step   = state.step+1
  lpips_params = state.lpips_params 
  
  key, sel_key, pmask_key, aug_key, self_key = jax.random.split(key, 5)
  
  B, num_frames, H, W  = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
  max_views = num_frames - 1 
  
  sample_ind = select_indices(sel_key, t_ind)[:, :(max_views+1)]

  x = x[jnp.tile(jnp.arange(B)[:, None], (1, num_frames)), sample_ind, ...]

  # Small chance that we decode reference frame
  s0_key, s1_key = jax.random.split(self_key, 2)
  
  self_mask = jax.random.uniform(s0_key, shape=(B, )) <= 0.02
           
  xT = x[:, -1, ...]
  xTS = permute(s1_key, x[:, :-1, ...])[:, 0, ...]
  
  xT = (1 - self_mask)[:, None, None, None] * xT + self_mask[:, None, None, None] * xTS 

  x = x.at[:, -1, ...].set(xT)

  # Stack for PoseEnc input
  # B*(V-1) x 2 x H x W x 3
  x_enc = jnp.stack( (jnp.tile(x[:, 0, None, ...], (1, max_views, 1, 1, 1)), x[:, 1:, ...]), axis=2)
  x_enc = jnp.concatenate( jnp.split(x_enc, max_views, axis=1), axis=0)[:, 0, ...]
  
  # B x 2 x H x W x 3
  if pose_aug_map is not None and train:
    x_enc = pose_aug_map(aug_key, x_enc)
  
  # Compute patch masks 
  num_patches = (H // patch_size) * (W // patch_size)
  
  pmask, qc = nutils.qpatch_masks(pmask_key, H // patch_size, W // patch_size, B)
  
  if train == False:
    pmask = jnp.zeros_like(pmask)
    qc = jnp.zeros_like(qc)

  pmask_enc = jnp.tile(pmask[:, None, ...], (1, max_views, 1))
  pmask_enc = jnp.concatenate(jnp.split(pmask_enc, max_views, axis=1), axis=0)[:, 0, ...]


  q0 = 1 * (qc == 0)
  q1 = 1 * (qc == 4)
  
  def loss_fn(params):
    
    # Compute relative poses 

    # P: B x 2 x 2 (this is aug dim) x Cp
    P = PoseEnc.apply({"params":params["PoseEnc"]}, x_enc, pmask_enc)

    # Unpack
    # B x V-1 x 2 x 2 x Cp
    P = jnp.stack(jnp.split(P, max_views, axis=0), axis=1)

    # B x V x 2 x Cp
    P = jnp.concatenate((P[:, 0, ...], P[:, 1:, 1, ...]), axis=1)
    
    # Predictions for each pair
    # B x V x 2 x Cp
    P0, P1 = P[:, :, 0, ...], P[:, :, 1, ...]

    # Copy over if we did not mask
    P0, P1 = (1 - q1)[:, None, None] * P0 + q1[:, None, None] * P1, (1 - q0)[:, None, None] * P1 + q0[:, None, None] * P0

    # Swap poses (transferability)
    P = jnp.stack((P1, P0), axis=2)
    

    # B x H x W x 3   
    xtp_S = Render.apply({"params":params["Render"]}, x[:, :-1, ...].astype(dtype), P, pmask)

    xtp_S = xtp_S.astype(jnp.float32)

    # If you're doing eval, also do a transfer for visualization
    if not train:
      PT = jnp.roll(P, 1, axis=0)
      PT = PT.at[:, -1, ...].set(P[:, -1, ...])
    
      xtp_T = Render.apply({"params":params["Render"]}, jnp.roll(x, 1, axis=0)[:, :-1, ...].astype(dtype), PT, jnp.roll(pmask, 1, axis=0))
  
      xtp_T = xtp_T.astype(jnp.float32)
    
    # Reconstruction loss 
    l10 = losses.l1_loss(x[:, -1, ...], xtp_S)


    lP0 = jnp.reshape(lpips.apply(lpips_params, x[:, -1, ...], xtp_S), (B, -1))
    lP0 = jnp.mean(lP0)

   
    w_l1, w_lpips = w_dict["w_l1"], w_dict["w_lpips"]

   
    loss = w_l1(s0) * l10 + w_lpips(s0) * lP0

    losses_all = (l10, lP0)

    # Just for visualization
    x_viz = x
          
    if not train:
      x_out = jnp.concatenate((x_viz, jnp.roll(x_viz, 1, axis=0)), axis=1)
      xp_out = jnp.concatenate((x_viz[:, :-1, ...], xtp_S[:, None, ...], jnp.roll(x_viz, 1, axis=0)[:, :-1, ...], xtp_T[:, None, ...]), axis=1)
    else:
      x_out = x_viz
      xp_out = jnp.concatenate((x_viz[:, :-1, ...], xtp_S[:, None, ...]), axis=1)

    xp = jnp.stack((x_out, xp_out), axis=1)

    return loss, (losses_all, xp)


  if train: 
    # Compute gradient
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)


    (loss, (losses_all, x_out)), grad = grad_fn(state.params)


    (loss_l1, loss_lpips) = losses_all 


    grad = jax.lax.pmean(grad, axis_name=PMAP_AXIS)
    grad = jax.tree_util.tree_map(jnp.conj, grad)


    updates, opt_state  = optimizer.update(grad, state.opt_state, state.params) 
    new_params              = optax.apply_updates(state.params, updates) 



    # Compute gradient statistics


    gravel, _ = jax.flatten_util.ravel_pytree(grad)
    uravel, _ = jax.flatten_util.ravel_pytree(updates)
    gravel = jnp.abs(gravel)
    uravel = jnp.abs(uravel)
    g_max = jnp.max(gravel)
    g_mean = jnp.mean(gravel)
    u_max = jnp.max(uravel)
    u_mean = jnp.mean(uravel)
  

    ## Update metrics 
    metrics = state.train_metrics

    
    metrics["train_loss"] += jax.lax.pmean(loss, axis_name=PMAP_AXIS)
    metrics["train_l1_loss"] += jax.lax.pmean(loss_l1, axis_name=PMAP_AXIS)
    metrics["train_lpips_loss"] += jax.lax.pmean(loss_lpips, axis_name=PMAP_AXIS)
    metrics["mean_grads"] += jax.lax.pmean(g_mean, axis_name=PMAP_AXIS)
    metrics["max_grads"] += jax.lax.pmean(g_max, axis_name=PMAP_AXIS)
    metrics["mean_updates"] += jax.lax.pmean(u_mean, axis_name=PMAP_AXIS)
    metrics["max_updates"] += jax.lax.pmean(u_max, axis_name=PMAP_AXIS)
    
    
    metrics["count"] += 1.0

    new_state           = state.replace( step=step,
                                         opt_state=opt_state,
                                         params=new_params,
                                         key=key,
                                         train_metrics=metrics)

  else:
    (loss, (losses_all, x_out)) = loss_fn(state.params) 


    (loss_l1, loss_lpips) = losses_all 


    metrics = state.eval_metrics

    metrics["eval_loss"] += jax.lax.pmean(loss, axis_name=PMAP_AXIS)
    metrics["eval_l1_loss"] += jax.lax.pmean(loss_l1, axis_name=PMAP_AXIS)
    metrics["eval_lpips_loss"] += jax.lax.pmean(loss_lpips, axis_name=PMAP_AXIS)
    metrics["count"] += 1.0


    new_state = state.replace(eval_metrics=metrics, key=key)   
    
  return new_state, x_out


def tile_array( a ):
  C, m, n = a.shape


  h = int( np.ceil(np.sqrt(C)) )
  w = int( np.ceil(C/h) )


  out                       = np.zeros( (h, w, m, n), dtype=a.dtype )
  out.reshape(-1, m, n)[:C] = a
  out                       = out.swapaxes( 1, 2 ).reshape(-1, w * n)


  return out  




def viz_results( cfg, x_out, mode="train", std_factor=2.5, image_dir=None):
  
  x_out = np.asarray(x_out)
  
  batch, x_out = x_out[:, 0, ...], x_out[:, 1, ...]

  B, num_samples, H, W = batch.shape[0], batch.shape[1], batch.shape[-3], batch.shape[-2]
  

  batch = jax.image.resize(batch, (B, num_samples, cfg.INPUT_SIZE[0], cfg.INPUT_SIZE[1], batch.shape[-1]), method="nearest")
      
  x_out = jax.image.resize(x_out, (B, num_samples, cfg.INPUT_SIZE[0], cfg.INPUT_SIZE[1], batch.shape[-1]), method="nearest")


  batch = 0.5 * (np.asarray(batch) + 1.0)
  x_out = 0.5 * (np.asarray(x_out) + 1.0)

  x_out = np.concatenate((batch, x_out), axis=-3)
   
  imB = np.concatenate(np.split(x_out, x_out.shape[1], axis=1), axis=-2)[:, 0, ...]
  

  imB = (255.0 * np.clip(imB, a_min=0, a_max=1)).astype(np.uint8)
  
  imBase_wandb   = []
  imB = imB[:100, ...]
  
  print("Converting images to wandb", flush=True)
  for l in tqdm(range(imB.shape[0])):
    imBase_wandb.append( wandb.Image(imB[l, ...], caption="{}_input_{}".format(mode, l)) )



  wandb_ims = {
               "{}_reconstructions".format(mode): imBase_wandb
                }
  
  return wandb_ims 




       
def train_and_evaluate( cfg: Any, in_dir: str, output_dir: str ):

  if not os.path.exists(output_dir):
    os.makedirs(dir_path)
  


  '''
  ========== Setup W&B =============
  '''
  project_name = cfg.PROJECT_NAME

  exp_name     =  "xfactor_multiview_{}".format(cfg.EXP_ID)

  cfg.WORK_DIR = output_dir 

  image_dir = os.path.join(output_dir, "images")
  
  module_to_dict = lambda module: {k: getattr(module, k) for k in dir(module) if not k.startswith('_')}
  config         = module_to_dict( cfg )
 
  run = wandb.init( config=config, project=project_name, name=exp_name, id=cfg.EXP_ID, resume="allow")
  
  '''
  ==================================
  '''


  ic(jax.default_backend())
  ic(jax.local_device_count())


  ## Set up meta-parameters 
  true_batch_size = cfg.TRUE_BATCH_SIZE
  batch_size      = cfg.BATCH_SIZE 
  batch_dim     = jax.local_device_count()
  
  assert true_batch_size % batch_dim == 0 
  assert batch_size % batch_dim == 0 
    
  multi_steps  = true_batch_size // batch_size 
  
  train_fn        = train_step
    
  viz_fn        = viz_results 
 
  input_size = cfg.INPUT_SIZE
  eval_every = cfg.EVAL_EVERY

  mixed = cfg.MIXED 
  

  #Create models
  print( "Initializing models..." )
  PoseEnc, Render, optimizer, w_dict, lpips, pose_aug_map, train_key, state = create_train_state( cfg, (4, *input_size, 3), in_dir)
  print( "Done..." )

  
  # Create checkpoints
  checkpoint_dir = os.path.join( output_dir, "checkpoints" )
  ckpt           = checkpoint.MultihostCheckpoint( checkpoint_dir, max_to_keep=2 )
  state          = ckpt.restore_or_initialize( state )
  
  initial_step = int(state.step) + 1

  #global_step = [state.step]
  step_tracker = vd.StepTracker(initial_step)

  ## Get dataset 
  print( "Getting dataset..." )

  # Set up samplers:
  samplers = {}

  for ds in cfg.DATASETS.keys():
    samplers[ds] = vd.make_dynamic_window_sampler(
                    train_step_fn=step_tracker.get_step,
                    num_samples=cfg.NUM_SAMPLES,
                    start_max=cfg.DATASETS[ds]["max_win_len_begin"],
                    end_max=  cfg.DATASETS[ds]["max_win_len_end"],
                    total_iters=cfg.FRAME_WARMUP_ITERS,
                    delay=cfg.FRAME_WARMUP_DELAY,
                    min_win_len=cfg.DATASETS[ds]["min_win_len"],
                    upweight = cfg.DATASETS[ds]["upweight"],
                    inclusive=False
                    )

  #train_dataset, test_dataset = vd.dynamic_amalgamated_vid_dataset(cfg, samplers)
  train_dataset, test_dataset = vd.dynamic_vid_amalgam_dataset(cfg, samplers)

  train_iter = iter(train_dataset)
  test_iter = iter(test_dataset)


  print( "Done..." )
  
  # Distribute
  state = flax_utils.replicate( state )


  print( "Distributing..." )

  p_train_step = jax.pmap( functools.partial(train_fn,
                                             PoseEnc      = PoseEnc,
                                             Render       = Render,
                                             optimizer    = optimizer,
                                             w_dict       = w_dict,
                                             lpips        = lpips,
                                             train        = True,
                                             patch_size   = cfg.PATCH_SIZE,
                                             pose_aug_map = pose_aug_map,
                                             mixed        = mixed),
                           axis_name=PMAP_AXIS )
    
  p_eval_step = jax.pmap( functools.partial(train_fn,
                                             PoseEnc      = PoseEnc,
                                             Render       = Render,
                                             optimizer    = optimizer,
                                             w_dict       = w_dict,
                                             lpips        = lpips,
                                             train        = False,
                                             patch_size   = cfg.PATCH_SIZE,
                                             pose_aug_map = pose_aug_map,
                                             mixed        = mixed),
                          axis_name=PMAP_AXIS )
 



  if cfg.STOP_AFTER is None:
    stop_at = cfg.NUM_TRAIN_STEPS + 1
  else:
    stop_at = cfg.STOP_AFTER + 1 
    
  print( "Beginning training..." )




  for step in tqdm( range(initial_step, stop_at) ):
    is_last_step = step == stop_at - 1
    step_tracker.set_step(step)

    try:
      ex = next( train_iter )
    except:
      train_iter = iter( train_dataset )
      ex        = next( train_iter )

    
    batch = ex["image"]
    t_ind = ex["frame_ind"]
    
    batch  = jnp.asarray(batch).astype(jnp.float32)
    t_ind = jnp.asarray(t_ind).astype(jnp.int32)
    
    batch = (2.0 * (batch / 255.0) - 1.0).astype(jnp.float32)

    batch_in = jnp.reshape(batch, (batch_dim, -1) + batch[0, ...].shape)
    t_ind_in = jnp.reshape(t_ind, (batch_dim, -1) + t_ind[0, ...].shape)
    
    state, x_out = p_train_step( x=batch_in, t_ind=t_ind_in, state=state )

    
    if step % cfg.LOG_LOSS_EVERY == 0 or is_last_step:
      
      state = flax_utils.unreplicate(state)
      metrics = state.train_metrics
      count = float(metrics["count"])
      del metrics["count"]

      for k in metrics.keys():
        metrics[k] = float(metrics[k]) / count
    
      run.log(data=metrics, step=step)


      state = state.replace(train_metrics=init_metrics(train_metrics_names))
      state = flax_utils.replicate(state)

    
    if step % cfg.CHECKPOINT_EVERY == 0 or is_last_step:
      ckpt.save( flax_utils.unreplicate(state) )
      
    '''
    ===========================================
    ============== Eval Loop ==================
    ===========================================
    '''
    
    if step % cfg.EVAL_EVERY == 0 or is_last_step:
      eval_metrics = None 
      
      for j in range( cfg.NUM_EVAL_STEPS ):
     
        try:
          ex = next( test_iter )
        except:
          test_iter = iter( test_dataset )
          ex        = next( test_iter )
      
        batch = ex["image"]    
        t_ind = ex["frame_ind"]

        batch  = jnp.asarray(batch).astype(jnp.float32)
        t_ind = jnp.asarray(t_ind).astype(jnp.int32)

        batch = (2.0 * (batch / 255.0) - 1.0).astype(jnp.float32)

        batch_in = jnp.reshape(batch, (batch_dim, -1) + batch[0, ...].shape)
        t_ind_in = jnp.reshape(t_ind, (batch_dim, -1) + t_ind[0, ...].shape)

        state, x_out = p_eval_step( x=batch_in, t_ind=t_ind_in, state=state )
    
        if j==(cfg.NUM_EVAL_STEPS - 1): 
    
          x_out = np.asarray(x_out)
          
          x_out = np.reshape(x_out, (-1, ) + x_out[0, 0, ...].shape)
      
      
      state = flax_utils.unreplicate(state)
      metrics = state.eval_metrics
      count = float(metrics["count"])
      del metrics["count"]

      for k in metrics.keys():
        metrics[k] = float(metrics[k]) / count
    
      run.log(data=metrics, step=step)


      state = state.replace(eval_metrics=init_metrics(eval_metrics_names))
      state = flax_utils.replicate(state)
      
      wandb_ims = viz_fn( cfg, x_out, mode="eval" )

      run.log(wandb_ims, step=step)
    
   



  
'''
#######################################################
###################### Main ###########################
#######################################################
'''


if __name__ == '__main__':

  # Ensure spawn start method so worker processes don't inherit parent's CUDA state.
  # Use a try/except because set_start_method can only be called once per interpreter.
  try:
    mp.set_start_method("forkserver", force=False)  # don't force unless you must
  except RuntimeError:
    # start method already set earlier; check it for debugging if needed
    current = mp.get_start_method(allow_none=True)
    print(f"multiprocessing start method already set to: {current}", flush=True)

  arguments = docopt( __doc__, version='Train 1.0' )

  in_dir = arguments['--in']
  in_dir = in_dir.format( root_path=ROOT_PATH )

  
  out_dir = arguments['--out']
  out_dir = out_dir.format( root_path=ROOT_PATH )


  config = arguments['--config']
  path   = dirname( abspath(__file__) )

  name = arguments['--name']


  spec                       = importlib.util.spec_from_file_location( "config", f"{path}/configs/{config}.py" )
  cfg                        = importlib.util.module_from_spec(spec)
  sys.modules["module.name"] = cfg
  spec.loader.exec_module( cfg )


  if not os.path.exists( out_dir ):
    os.makedirs( out_dir )


  if name is None:
    print("Name is none!", flush=True)

    files = os.listdir( out_dir )
    count = 0
    
    for f in files:
  
  
      if os.path.isdir( os.path.join(out_dir, f) ) and f.isnumeric():
        count += 1
  
  
  
    exp_name = str( count )

  else:
    exp_name = name 
    
  print( "==========================" )
  print( f"Experiment # {exp_name}" )
  print( "==========================" )
  
  exp_dir = os.path.join( out_dir, exp_name )
  if not os.path.exists(exp_dir):
    os.mkdir( exp_dir )
  ic( exp_dir )
  cfg.WORK_DIR = exp_dir
  cfg.EXP_ID = exp_name


  
  train_and_evaluate( cfg, in_dir, exp_dir )
