from functools import partial
from typing import Tuple, Any
import jax 
import jax.numpy as jnp
from jax._src.typing import Array 
from jax._src.config import config 
import numpy as np



def safe_inverse(x):
  return x / (x**2 + 1.0e-8)
  
def l1_loss(x, x_tilde):
  return jnp.mean( jnp.abs(x - x_tilde))
  #return jnp.sum( jnp.abs(x - x_tilde) ) / x.shape[0]

def l2_loss(x, x_tilde):
  return jnp.mean( (x - x_tilde)**2 )


def scale_procrustes(A, B):

  num = jnp.trace(jnp.matmul(jnp.swapaxes(B, -2, -1), A), axis1=-2, axis2=-1)
  denom = jnp.trace(jnp.matmul(jnp.swapaxes(B, -2, -1), B), axis1=-2, axis2=-1)
  
  s = num / jnp.clip(denom, a_min=1.0e-8)
  
  #s = jax.lax.stop_gradient(s)
  
  return s[..., None, None] * B 
  

def camera_loss(R0, t0, Rp, tp):

  loss_t = jnp.mean( jnp.abs(t0 - tp) )
  
  loss_R = jnp.mean( jnp.abs( R0 - Rp) )

  return loss_R, loss_t 

  

def camera_seq_loss(R0, t0, Rp, tp):

  tp = scale_procrustes(t0, tp)

  loss_t = jnp.mean(jnp.abs(t0 - tp))
  loss_R = jnp.mean(jnp.abs(R0 - Rp))

  return loss_R, loss_t

