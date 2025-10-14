from functools import partial
from typing import Tuple, Any, Optional
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.typing import Array
import flax
from flax import linen as nn
import jax.experimental.pallas.ops.gpu.attention as jepattn
from flax.linen.dtypes import promote_dtype
from icecream import ic 
import sys 

import nn.nutils as nutils 


NORM_QK = True 

EPS = 1.0e-5

PI = np.pi

BIG_INT = 4096



def get_1d_sincos_rope(embed_dim, t_ind):

  assert embed_dim % 2 == 0
  
  omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
  omega /= embed_dim / 2.
  omega = 1. / 10000**omega  # (D/4,)
  omega = jnp.repeat(omega, 2)

  omega = omega[None, :] * t_ind[:, None]

  sincos = jnp.stack((jnp.cos(omega), jnp.sin(omega)), axis=-1)

  return sincos



def get_2d_sincos_rope(embed_dim, grid_size):

  # Returns: (H*W, embed_dim, 2)
  
  grid_h = jnp.arange(grid_size[0], dtype=jnp.float32)
  grid_w = jnp.arange(grid_size[1], dtype=jnp.float32)
  
  I, J = jnp.meshgrid(grid_h, grid_w, indexing='ij')
  
  I, J = jnp.reshape(I, (-1, )), jnp.reshape(J, (-1, ))

  assert embed_dim % 4 == 0
  
  omega = jnp.arange(embed_dim // 4, dtype=jnp.float32)
  omega /= embed_dim / 4.
  omega = 1. / 10000**omega  # (D/4,)
  omega = jnp.repeat(omega, 2)

  # H*W x embed_dim
  omega = jnp.concatenate(( omega[None, :] * I[:, None], omega[None, :] * J[:, None]), axis=-1)

  sincos = jnp.stack((jnp.cos(omega), jnp.sin(omega)), axis=-1)

  return sincos


def window_partition(x, window_size, final_reshape=True):

    # x: (B, H, W, C)
    # Returns (B, num_windows, window_size * window_size * C) if final_reshape
    # or (B, num_windows, window_size, window_size, C)
  
    batch, height, width, channels = x.shape
    x = jnp.reshape(
        x,
        (
            batch,
            height // window_size,
            window_size,
            width // window_size,
            window_size,
            channels,
        ),
    )
    windows = jnp.reshape(
        jnp.transpose(x, (0, 1, 3, 2, 4, 5)), (batch, -1, window_size, window_size, channels)
    )

    if final_reshape:
      windows = jnp.reshape(windows, (windows.shape[0], windows.shape[1], -1))
   
    return windows
 

def window_reverse(windows, window_size, height, width):

    # windows: (B, num_windows, window_size*window_size*C) or (B, num_windows, window_size, window_size, C)
    # Returns: (B, H, W, C)
  
    batch = windows.shape[0]
   
    windows = jnp.reshape(windows, (windows.shape[0], windows.shape[1], window_size, window_size, -1))
    #batch = int(windows.shape[0] / (height * width / window_size / window_size))
    x = jnp.reshape(
        windows,
        (
            batch,
            height // window_size,
            width // window_size,
            window_size,
            window_size,
            -1,
        ),
    )
    x = jnp.reshape(jnp.transpose(x, (0, 1, 3, 2, 4, 5)), (batch, height, width, -1))
    return x




class MLP2(nn.Module):
  features: int 
  out_dim: int 
  use_bias: bool = True
  small_init: bool = False 
  
  
  @nn.compact
  def __call__(self, x):

    # Input: (B, T, C)
    # Output: (B, T, out_dim)
    
    dtype = x.dtype 

    if self.small_init:
      k_init = nn.initializers.normal(stddev=1e-6)
    else:
      k_init = nn.initializers.xavier_uniform()
      
    x0 = nn.Dense(features=4*self.features,
                 kernel_init=k_init,
                 bias_init=nn.initializers.normal(stddev=1e-6),
                 use_bias=self.use_bias,
                 dtype=dtype)(x)
    
    x0 = jax.nn.gelu(x0)
    
    x0 = nn.Dense(features=self.out_dim,
                 kernel_init=k_init,
                 bias_init=nn.initializers.normal(stddev=1e-6),
                 use_bias=self.use_bias,
                 dtype=dtype)(x0)
    
    return x0

class MLP3(nn.Module):
  features: int 
  out_dim: int 
  use_bias: bool = True 
  small_init: bool = False 
  
  @nn.compact
  def __call__(self, x):

    # Input: (B, T, C)
    # Output: (B, T, out_dim)
    
    dtype = x.dtype 

    if self.small_init:
      k_init = nn.initializers.normal(stddev=1e-6)
    else:
      k_init = nn.initializers.xavier_uniform()
    
    x0 = nn.Dense(features=4*self.features,
                 kernel_init=k_init,
                 bias_init=nn.initializers.normal(stddev=1e-6),
                 dtype=dtype)(x)
    
    x0 = jax.nn.gelu(x0)
    
    x0 = nn.Dense(features=self.features,
                 kernel_init=k_init,
                 bias_init=nn.initializers.normal(stddev=1e-6),
                 dtype=dtype)(x0)
  
    x0 = jax.nn.gelu(x0)
  
    x0 = nn.Dense(features=self.out_dim,
             kernel_init=k_init,
             bias_init=nn.initializers.normal(stddev=1e-6),
             dtype=dtype)(x0) 
    
    
    return x0


def rotate_two(x):
  # x: (..., D) where D must be even
  # Returns: same shape as x, (..., D)
  # Applies a rotation transform to 2D subvectors in the last dimensio
  
  return jnp.reshape(jnp.concatenate( (-1.0 * x[..., 1::2, None], x[..., ::2, None]), axis=-1), x.shape)



class RMSNorm(nn.Module):

  @nn.compact
  def __call__(self, x):

    # Input: (B, T, C) or (B, T, H, D)
    # Output: same shape
    
    dtype = x.dtype 
    
    alpha = self.param("alpha", nn.initializers.ones, shape=(x.shape[-2], x.shape[-1]), dtype=jnp.float32)

    alpha = jnp.broadcast_to(alpha, x.shape).astype(dtype)

    mag = jnp.sqrt(jnp.mean(x.astype(jnp.float32)**2, axis=-1, keepdims=True) + EPS).astype(dtype)

    return alpha * x / mag 
        

class LayerScale(nn.Module):
  init_vals: float = 0.01 
  @nn.compact
  def __call__(self, x):

    # Input: (B, T, C)
    # Output: same shape
    
    alpha = self.param("alpha", nn.initializers.constant(self.init_vals), shape=(x.shape[-1], ), dtype=jnp.float32)

    alpha = jnp.broadcast_to(alpha, x.shape).astype(x.dtype)

    return alpha * x 


class AdaLNZ(nn.Module):
  small_init = True
  @nn.compact
  def __call__(self, x, P):
    # x: B x N x C 
    # P: B x N x Cp

    dtype = x.dtype 
    
    #mods = TinyMLP(features=x.shape[-1], out_dim=3*x.shape[-1], small_init=True)(P)

    if self.small_init:
      kernel_init = nn.initializers.normal(stddev=1e-6)
    else:
      kernel_init = nn.initializers.xavier_uniform()
      
    mods = nn.Dense(features=3*x.shape[-1],
           kernel_init=nn.initializers.normal(stddev=1e-6),
           bias_init=nn.initializers.normal(stddev=1e-6),
           dtype=dtype)(P)
    
    scale, shift, gate = jnp.split(mods , 3, axis=-1)

    x = nn.LayerNorm(use_bias=False, use_scale=False, dtype=dtype)(x)

    x = (1 + scale) * x + shift

    return x, gate 
                   

# Multi-view VIT Layer
# Fused per-image and global attention
class MVLayer(nn.Module):
  features: int 
  num_heads: int 
  normalize_qk: bool = True
  block_size: int = 64 # Has to divide spatial dimension
  use_bias: bool = True

  def setup(self):
    blocks = jepattn.BlockSizes(block_q=self.block_size,
                                block_k=self.block_size,
                                block_q_dkv=self.block_size,
                                block_kv_dkv = self.block_size,
                                block_q_dq = self.block_size,
                                block_kv_dq = self.block_size)
    self.blocks = blocks 
      
  @nn.compact
  def __call__(self, x, patch_ids=None, sincos_rope=None):

    # Input: x (B, V, T, C), patch_ids (B, T),  sincos_rope: (T, C // NUM_HEADS, 2)
    # Output: (B, T, C)
    
    _init = nn.initializers.xavier_uniform()

    B, V, T, C = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

    assert T % self.block_size == 0

    if patch_ids is None:
      patch_ids = jnp.zeros((x.shape[0], x.shape[2]), dtype=jnp.int32)

    if patch_ids.ndim == 2: 
      patch_ids = jnp.tile(patch_ids[:, None, :], (1, V, 1))
    
    dtype = x.dtype 
    
    in_features = x.shape[-1]

    head_dim = in_features // self.num_heads 

    q_dim = head_dim * self.num_heads

    x_in = nn.LayerNorm(use_bias=False, dtype=dtype)(x)
  
    qkv_all = nn.Dense(features=(6 * q_dim),                   
                       kernel_init=_init,
                       use_bias=False, 
                       dtype=dtype)(x_in)
    
    q, k, v, q_self, k_self, v_self = jnp.split(qkv_all, (q_dim, 2*q_dim, 3*q_dim, 4*q_dim, 5*q_dim), axis=-1)

    q = jnp.reshape(q, (B, V, T, self.num_heads, -1))
    k = jnp.reshape(k, (B, V, T , self.num_heads, -1))
    v = jnp.reshape(v, (B, V, T, self.num_heads, -1))    
    
    q_self = jnp.reshape(q_self, (B, V, T, self.num_heads, -1))
    k_self = jnp.reshape(k_self, (B, V, T, self.num_heads, -1))
    v_self = jnp.reshape(v_self, (B, V, T, self.num_heads, -1))

    
    # Normalization
    if self.normalize_qk:
      
      q = RMSNorm()(q)
      k = RMSNorm()(k)
      q_self = RMSNorm()(q_self)
      k_self = RMSNorm()(k_self)

    # RoPE
    if sincos_rope is not None: 

      q = q * sincos_rope[None, None, :, None, :, 0] + rotate_two(q) * sincos_rope[None, None, :, None, :, 1]
      k = k * sincos_rope[None, None, :, None, :, 0] + rotate_two(k) * sincos_rope[None, None, :, None, :, 1]

      q_self = q_self * sincos_rope[None, None, :, None, :, 0] + rotate_two(q_self) * sincos_rope[None, None, :, None, :, 1]
      k_self = k_self * sincos_rope[None, None, :, None, :, 0] + rotate_two(k_self) * sincos_rope[None, None, :, None, :, 1]

    ## Global reshape
    # B x V x T x num_heads x Ch -> B x V * T x num_heads x Ch
    q = jnp.concatenate(jnp.split(q, V, axis=1), axis=2)[:, 0, ...]
    k = jnp.concatenate(jnp.split(k, V, axis=1), axis=2)[:, 0, ...]
    v = jnp.concatenate(jnp.split(v, V, axis=1), axis=2)[:, 0, ...]

    # B x V * T 
    global_patch_ids = jnp.concatenate(jnp.split(patch_ids, V, axis=1), axis=-1)[:, 0, :]

    ## Self reshape
    # B X V x T x num_heads x Ch -> B * V x T x num_heads x Ch  
    q_self = jnp.concatenate(jnp.split(q_self, V, axis=1), axis=0)[:, 0, ...]
    k_self = jnp.concatenate(jnp.split(k_self, V, axis=1), axis=0)[:, 0, ...]
    v_self = jnp.concatenate(jnp.split(v_self, V, axis=1), axis=0)[:, 0, ...]

    # B x V * T 
    self_patch_ids = jnp.concatenate(jnp.split(patch_ids, V, axis=1), axis=0)[:, 0, :]

    
    ## Global (multi-view) attention
    x_global = jepattn.mha(q=q.astype(jnp.bfloat16),
                           k=k.astype(jnp.bfloat16),
                           v=v.astype(jnp.bfloat16),
                           segment_ids=global_patch_ids,
                           block_sizes=self.blocks, # Has to divide spatial dimension
                           sm_scale=1.0/(q.shape[-1] ** 0.5)).astype(dtype)
    
    x_global = jnp.reshape(x_global, (x_global.shape[0], x_global.shape[1], -1))

    # B x V * T x num_heads x Cout-> B X V x T x num_heads x Cout
    x_global = jnp.stack(jnp.split(x_global, V, axis=1), axis=1)
    
    # Self-attention
    x_self = jepattn.mha(q=q_self.astype(jnp.bfloat16),
           k=k_self.astype(jnp.bfloat16),
           v=v_self.astype(jnp.bfloat16),
           segment_ids=self_patch_ids,
           block_sizes=self.blocks, # Has to divide spatial dimension
           sm_scale=1.0/(q_self.shape[-1] ** 0.5)).astype(dtype)

    x_self = jnp.reshape(x_self, (x_self.shape[0], x_self.shape[1], -1))

    # B * V x T x Cout -> B X V x T x num_heads x Cout
    x_self = jnp.stack(jnp.split(x_self, V, axis=0), axis=1)
    
    x0 = jnp.concatenate((x_global, x_self), axis=-1)

    # B x V x T x C
    x0 = nn.Dense(features=self.features,                        
                     kernel_init=_init,
                     bias_init=nn.initializers.normal(stddev=1e-6),
                     use_bias=self.use_bias,
                     dtype=dtype)(x0)


    x0 = LayerScale()(x0)
    
    x0 = x0 + x 

    x1 = nn.LayerNorm(dtype=dtype, use_bias=self.use_bias)(x0)

    x1 = jax.nn.gelu(x1)
    x1 = MLP2(features=self.features, out_dim=self.features, use_bias=self.use_bias)(x1)

    x1 = LayerScale()(x1)
    
    x2 = x1 + x0

    return x2




'''
===========================================================================================
============================================ Core =========================================
===========================================================================================
'''


class PoseHead(nn.Module):
  pose_dim: int 

  @nn.compact
  def __call__(self, x, x0):

    dtype = x.dtype 
    
    proj = MLP3(features=x.shape[-1], out_dim=self.pose_dim)

    P = jnp.concatenate((x, x0), axis=-1)
    P0 = jnp.concatenate((x0, x0), axis=-1)
    
    P = proj(P) - proj(P0)
    
    return P 



# Multi-Image Pose Encoder
class PoseEnc(nn.Module):

  features: int 
  num_heads: int
  num_layers: int
  patch_size: int 
  pose_dim: int
  checkpoint: bool = True
  block_size: int = 64
  use_bias: bool = True
  
  def setup(self):
   
    layers = []

    for l in range(self.num_layers):

      if not self.checkpoint:
        layers.append(MVLayer(features=self.features, 
                              num_heads=self.num_heads,
                              block_size=self.block_size,
                              use_bias=self.use_bias))
      else:
        layers.append(nn.checkpoint(MVLayer)(features=self.features, 
                                             num_heads=self.num_heads,
                                             block_size=self.block_size,
                                             use_bias=self.use_bias))

    
    self.layers = layers 

  @nn.compact
  def __call__(self, x, pmask=None, return_probe=False):

    # V is the number of input images of the scene
    # x: B x V x H X W x C
    # pmask: B x num_pix_tokens (num_pix_tokens = H*W//patch_size**2) (binary patch mask)

    B, V, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    Hp, Wp = H // self.patch_size, W // self.patch_size
    dtype = x.dtype     

    
    ## Setup mask
    if pmask is not None:
      return_single = False
      pmask = pmask.astype(jnp.int32)
    else:
      return_single = True 
      pmask = jnp.zeros((B, Hp * Wp), dtype=jnp.int32)
      
    if jnp.ndim(pmask) == 2:
      pmask = jnp.tile(pmask[:, None, :], (1, V, 1))


    ## Patchify
    x = jnp.reshape(x, (-1, H, W, x.shape[-1]))
    x = window_partition(x, self.patch_size)
   
    x = jnp.reshape(x, (x.shape[0], -1, x.shape[-1]))

    # B x V x T x C 
    x = jnp.reshape(x, (B, V, x.shape[-2], x.shape[-1]))
    
    # num_im_tokens = T = H * W // (patch_size ** 2)
    num_im_tokens = x.shape[2]

    
    x = nn.Dense(features=self.features,                   
                       kernel_init=nn.initializers.xavier_uniform(),
                       bias_init=nn.initializers.normal(stddev=1e-6),
                       use_bias=self.use_bias,
                       dtype=dtype)(x)
      
    
    ## Global pose tokens
    # Here I'm initializing a global token for the poses, copying it twice for each set of patches, and appending 
    # the copies to the image tokens for each view

    # Intializing block_size//2 tokens is a necessary quirk so that we can play nice with JAX's Flash Attention.
    # Just do 16 or something if you're porting this to PyTorch or whatever

    global_tokens = self.block_size // 2 
    
    global_embed = self.param("global_embed", nn.initializers.normal(stddev=0.02), (global_tokens, x.shape[-1]), jnp.float32)[None, None, ...]

    global_embed = jnp.tile(global_embed, (B, V, 1, 1)).astype(dtype)

    # Update pmask to assign each copy of the tokens to black and white patches during attention
    # B x V x (T + block_size) 
    pmask = jnp.concatenate((pmask, 
                             jnp.zeros( (B, V, global_tokens), dtype=jnp.int32), 
                             jnp.ones( (B, V, global_tokens), dtype=jnp.int32)), axis=-1)
    

    # B x V x (T + block_size) x C 
    z = jnp.concatenate((x, global_embed, global_embed), axis=2)
    
    ## Set up ROPE embeddings
    assert self.features % self.num_heads == 0 
    head_feat = self.features // self.num_heads 

    # T x C_heads x 2 
    sincos_rope = get_2d_sincos_rope(head_feat, (H // self.patch_size, W // self.patch_size))

    # Attach identity features to global tokens
    id_block = jnp.concatenate( (jnp.ones((2*global_tokens, head_feat, 1), dtype=jnp.float32),
                                 jnp.zeros((2*global_tokens, head_feat, 1), dtype=jnp.float32)), axis=-1)

    # (T + block_size) x C x 2
    sincos_rope = jnp.concatenate((sincos_rope, id_block), axis=0).astype(dtype)

    ## Pass through network 
    # z: B x V x (T + block_size) x C
    # pmask: B x V x (T + block_size)
    # sincos_rope: (T + block_size) x C x 2
    for L in self.layers:      
      z = L(z, pmask, sincos_rope)

    ## Extract pose tokens 
    # B x V x 2 x block_size // 2 x C
    z = jnp.stack(jnp.split(z[:, :, num_im_tokens:, :], 2, axis=2), axis=2)

    # Just take first global token as pose token for each set of patches
    # B x V x 2 x C 
    z = z[..., 0, :]
    z = nn.LayerNorm(dtype=dtype, use_bias=self.use_bias)(z)
    z = jax.nn.gelu(z)


    ## Pose head
    # First frame is assumed to be the reference frame relative to which all other poses are expressed

    z0 = jnp.tile(z[:, 0, None, ...], (1, V, 1, 1))

    
    # B x V x 2 x Cp
    P = PoseHead(pose_dim=self.pose_dim)(z, z0)


    # Probe input, if desired
    # B x V x 2 x 2*C
    P0 = jnp.concatenate((z0, z), axis=-1)
    
    # If we didn't pass a mask (which assumes we're working over the full image)
    # just return the pose latent, not two copies
    if return_single:
      P = P[..., 0, :]
      P0 = P0[..., 0, :]

    # Latent pose vector of reference frame is always zero vector
    if return_probe:
      return P, P0
    else:
      return P

# Multi-Image Render
class Render(nn.Module):

  features: int 
  num_heads: int
  num_layers: int
  patch_size: int 
  checkpoint: bool = True 
  block_size: int = 64
  use_bias: bool = True

  
  def setup(self):
   
    layers = []
    for l in range(self.num_layers):

      if not self.checkpoint:
        layers.append(MVLayer(features=self.features, 
                                    num_heads=self.num_heads, 
                                    block_size=self.block_size,
                                    use_bias=self.use_bias))
      else:
        layers.append(nn.checkpoint(MVLayer)(features=self.features,
                                                   num_heads=self.num_heads, 
                                                   block_size=self.block_size,
                                                   use_bias=self.use_bias))
          
    self.layers = layers

  @nn.compact
  def __call__(self, x, P, pmask=None):
    
    # x: B x V-1, H X W x C
    # P: B x V x 2 x Cp or B x V x Cp (P[:, -1, ...] is the pose for the image that will be reconstructed)
    # pmask: B x T (T = num_pix_tokens=H*W // patch_size**2) (mask for each view)
    # smask: B x V (which views to include in attention, binary, 1 to include, 0 to exclude)

  
    B, V, H, W = x.shape[0], P.shape[1], x.shape[2], x.shape[3]
    Hp, Wp = H // self.patch_size, W // self.patch_size
    dtype = x.dtype     

    ## Setup mask
    if pmask is not None:
      pmask = pmask.astype(jnp.int32)
    else:
      pmask = jnp.zeros((B, Hp * Wp), dtype=jnp.int32)
      
    if jnp.ndim(pmask) == 2:
      pmask = jnp.tile(pmask[:, None, :], (1, V, 1))

    
    ## Patchify
    x = jnp.reshape(x, (-1, H, W, x.shape[-1]))
    
    x = window_partition(x, self.patch_size)
    x = jnp.reshape(x, (x.shape[0], -1, x.shape[-1]))
    
    # x: B x V-1 x T = num_im_tokens x C
    x = jnp.reshape(x, (B, V-1, x.shape[-2], x.shape[-1]))

    num_im_tokens = x.shape[-2]

    # Broadcast pose tokens over image tokens 
    # P0, P1: B x V x Cp

    if P.ndim == 3:
      P0, P1 = P[:, :, None, :], P[:, :, None, :]
    else:
      P0, P1 = P[:, :, 0, None, :], P[:, :, 1, None, :]
      
    # P x V x T x Cp         
    P = (1 - pmask)[..., None] * P0 + pmask[..., None] * P1

    # Concatenate context + reference images with context + reference poses
    x = jnp.concatenate((x, P[:, :-1, ...]), axis=-1)

    # Target 
    y = P[:, -1, ...]

    # Lift both context and target patchifications
    
    # B x V-1 x T x C
    x = MLP2(features=self.features, out_dim=self.features, use_bias=self.use_bias)(x)

    # B x T x C
    y = MLP2(features=self.features, out_dim=self.features, use_bias=self.use_bias)(y)

    # Combine
    # B x V x T x C 
    z = jnp.concatenate((x, y[:, None, ...]), axis=1)

                                       
    # Set up RoPE Embeddings
    assert self.features % self.num_heads == 0 
    head_feat = self.features // self.num_heads 

    # T x C x 2
    sincos_rope = get_2d_sincos_rope(head_feat, (H // self.patch_size, W // self.patch_size)).astype(dtype)
  

    # z: B x V x T x C 
    # pmask: B x V x T
    # sincos_rope: V x T x 2 
    for L in self.layers:
      z = L(z, pmask, sincos_rope)


    ## Pixel head
    
    # z: B x num_pix_tokens x C (get tokens for target image)
    z = z[:, -1, ...]
    
    z = nn.LayerNorm(dtype=dtype, use_bias=self.use_bias)(z)
    
    z = jax.nn.gelu(z)
    
    z = MLP3(features=z.shape[-1], out_dim=self.patch_size*self.patch_size*3, use_bias=self.use_bias)(z)

    z = 2.0 * (jax.nn.sigmoid(z) - 0.5)

    # z: B x H x W x 3 
    z = window_reverse(z, self.patch_size, H, W)

    return z 
    

      
'''
==================================================================================
================================== Probe =========================================
==================================================================================
'''
  
class pose_probe(nn.Module):
  features: int = 64

  @nn.compact
  def __call__(self, P):

    P = MLP3(features=self.features, out_dim=12, small_init=False)(P)
    
    t, R = P[..., :3], P[..., 3:]

    R = nutils.orthogonal_projection_kernel(jnp.reshape(R, R[..., 0].shape + (3, 3)))
    
    return R, t 


'''
==================================================================================
========================= Optimizer Mask =========================================
==================================================================================
'''

def mask_exclude_special_params(params):
    """
    Returns a boolean mask matching `params`, where leaves are False if:
      - They are under LayerNorm_*, RMSNorm_*, or LayerScale_* modules
      - Their name is 'bias'
      - Their name contains '_embed'
    """
    flat_params = flax.traverse_util.flatten_dict(params, keep_empty_nodes=True)

    def is_excluded(path):
        # Exclude if any module in the path is LayerNorm_*, RMSNorm_*, or LayerScale_*
        if any(
            isinstance(k, str) and (
                k.startswith("LayerNorm_") or
                k.startswith("RMSNorm_") or
                k.startswith("LayerScale_")
            )
            for k in path
        ):
            return True

        # Exclude if leaf name is 'bias' or contains '_embed'
        leaf_name = path[-1]
        if isinstance(leaf_name, str) and ("_embed" in leaf_name or leaf_name == "bias"):
            return True

        return False

    flat_mask = {
        path: not is_excluded(path)
        for path in flat_params
    }

    return flax.traverse_util.unflatten_dict(flat_mask)
  
