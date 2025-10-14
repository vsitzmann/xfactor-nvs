from functools import partial
from typing import Tuple, Any, Optional
import jax 
import jax.numpy as jnp
from jax import lax
from jax import custom_jvp, custom_vjp, vjp
from jax._src.typing import Array 
from jax._src.config import config 
from jax._src.numpy import ufuncs
import numpy as np
import scipy as sp
import dm_pix as pix 


EPS = 1.0e-8
PI = np.pi 


def _T(x: Array) -> Array: return jnp.swapaxes(x, -1, -2)
def _H(x: Array) -> Array: return ufuncs.conj(_T(x))

def safe_inverse(x):
  return x / (x**2 + EPS) 

def _extract_diagonal(s: Array) -> Array:
  """Extract the diagonal from a batched matrix"""
  i = jax.lax.iota('int32', min(s.shape[-2], s.shape[-1]))
  return s[..., i, i]
  
@custom_jvp
def safe_svd(A):

  U, S, VT = jnp.linalg.svd(A, full_matrices=False)
  
  return U, S, VT
  
@safe_svd.defjvp
def safe_svd_jvp(primals, tangents):
  
  A, = primals 
  dA, = tangents 
  
  U, s, Vt = jnp.linalg.svd(A, full_matrices=False)
  
  Ut, V = _H(U), _H(Vt)
  s_dim = s[..., None, :]
  dS = Ut @ dA @ V
  ds = ufuncs.real(jnp.diagonal(dS, 0, -2, -1))



  s_diffs = (s_dim + _T(s_dim)) * (s_dim - _T(s_dim))
  s_diffs_zeros = jnp.eye(s.shape[-1], dtype=s.dtype)  # jnp.ones((), dtype=A.dtype) * (s_diffs == 0.)  # is 1. where s_diffs is 0. and is 0. everywhere else
  s_diffs_zeros = lax.expand_dims(s_diffs_zeros, range(s_diffs.ndim - 2))
  #F = 1 / (s_diffs + s_diffs_zeros) - s_diffs_zeros
  F = safe_inverse(s_diffs + s_diffs_zeros) - s_diffs_zeros 
  dSS = s_dim.astype(A.dtype) * dS  # dS.dot(jnp.diag(s))
  SdS = _T(s_dim.astype(A.dtype)) * dS  # jnp.diag(s).dot(dS)

  s_zeros = (s == 0).astype(s.dtype)
  #s_inv = 1 / (s + s_zeros) - s_zeros
  s_inv = safe_inverse(s + s_zeros) - s_zeros
  s_inv_mat = jnp.vectorize(jnp.diag, signature='(k)->(k,k)')(s_inv)
  dUdV_diag = .5 * (dS - _H(dS)) * s_inv_mat.astype(A.dtype)
  dU = U @ (F.astype(A.dtype) * (dSS + _H(dSS)) + dUdV_diag)
  dV = V @ (F.astype(A.dtype) * (SdS + _H(SdS)))

  m, n = A.shape[-2:]
  if m > n:
    dAV = dA @ V
    dU = dU + (dAV - U @ (Ut @ dAV)) / s_dim.astype(A.dtype)
  if n > m:
    dAHU = _H(dA) @ U
    dV = dV + (dAHU - V @ (Vt @ dAHU)) / s_dim.astype(A.dtype)

  return (U, s, Vt), (dU, ds, _H(dV))




@custom_jvp
def ortho_det( U ):
  return jnp.linalg.det( U )
 
 
@ortho_det.defjvp
def ortho_det_jvp(primals, tangents):
  x, = primals
  g, = tangents
  
  y = jnp.linalg.det( x )
  z = jnp.einsum( "...ji, ...jk->...ik", y[..., None, None] * x, g )
  
  return y, jnp.trace( z, axis1=-1, axis2=-2 )
 

# Stable and differentiable procrustes projection
def orthogonal_projection_kernel(X, special=True):

  dtype = X.dtype 
  
  X = X.astype(jnp.float32)
  
  U, _, VH = safe_svd( X )

  if (X.shape[-2] == X.shape[-1] and special):
    VH = VH.at[..., -1, :].set( ortho_det(jnp.einsum("...ij, ...jk -> ...ik", U, VH, precision=jax.lax.Precision.HIGHEST))[..., None] * VH[..., -1, :])
  
  R = jnp.einsum( "...ij, ...jk -> ...ik", U, VH, precision=jax.lax.Precision.HIGHEST)
  
  R = R.astype(dtype)
  
  return R 


'''
=======================================================================================================================
================================================= Augmentations =======================================================
=======================================================================================================================
'''

def permute(key, A):

  def _perm(key, a):
    return jax.random.permutation(key, a)

  p_key = jax.random.split(key, A.shape[0])
  
  return jax.vmap(_perm, in_axes=(0, 0), out_axes=0)(p_key, A)
  
def qpatch_masks(key, H, W, B, full_chance = 0.05):

  key1, key2, key3 = jax.random.split(key, 3)

  q = jnp.concatenate((jnp.ones( (B, 2), dtype=jnp.int32), jnp.zeros( (B, 2), dtype=jnp.int32)), axis=-1)
  q0 = jnp.zeros((B, 4), dtype=jnp.int32)

  full_mask = jax.random.uniform(key1, (B, )) <= full_chance 

  q = (1.0 - full_mask)[:, None] * q + full_mask[:, None] * q0

  q = permute(key2, q)
  
  assert H % 2 == 0 and W % 2 == 0
  H2 = H // 2 
  W2 = W // 2

  q00 = q[:, 0, None, None] * jnp.ones((1, H2, W2), dtype=jnp.int32)
  q10 = q[:, 1, None, None] * jnp.ones((1, H2, W2), dtype=jnp.int32)
  q01 = q[:, 2, None, None] * jnp.ones((1, H2, W2), dtype=jnp.int32)
  q11 = q[:, 3, None, None] * jnp.ones((1, H2, W2), dtype=jnp.int32)

  top = jnp.concatenate((q00, q10), axis=-1)
  bot = jnp.concatenate((q01, q11), axis=-1)

  pmask = jnp.concatenate((top, bot), axis=-2)

  pmask = jnp.reshape(pmask, (B, -1))

  return pmask, jnp.sum(q, axis=1)


## These guys below are all from ChatGPT
def color_jitter(
    key: Any,
    x: jnp.ndarray,                   # [B, V, H, W, 3]
    p_apply: float = 0.8,
    d_bright: float = 0.2,
    d_cont: float = 0.2,
    d_sat: float = 0.2,
    d_hue: float = 0.05,
    time_invar: bool = False
) -> jnp.ndarray:
    B, V, H, W, C = x.shape
    x_flat = x.reshape((B * V, H, W, C))

    # figure out how many keys we need
    n = B if time_invar else B * V
    keys = jax.random.split(key, n)
    if time_invar:
        keys = jnp.repeat(keys, V, axis=0)

    def _jit_one(k, im):
        im = (im + 1.0) / 2.0
        k_main, k_app, *k_ops = jax.random.split(k, 6)
        perm = jax.random.permutation(k_main, 4)

        def apply_op(i, val):
            return jax.lax.switch(i, [
                lambda y: pix.random_brightness(k_ops[0], y, d_bright),
                lambda y: pix.random_contrast(k_ops[1], y, 1-d_cont, 1+d_cont),
                lambda y: pix.random_saturation(k_ops[2], y, 1-d_sat, 1+d_sat),
                lambda y: pix.random_hue(k_ops[3], y, d_hue),
            ], val)

        y = jax.lax.fori_loop(0, 4, lambda i, v: apply_op(perm[i], v), im)
        do_it = jax.random.uniform(k_app) <= p_apply
        out = jax.lax.select(do_it, y, im)
        return jnp.clip(2*(out - 0.5), -1.0, 1.0)

    x_j = jax.vmap(_jit_one)(keys, x_flat)
    return x_j.reshape((B, V, H, W, C))


def gaussian_kernel_1d(sigma: float, kernel_size: int):
    x = jnp.arange(kernel_size) - kernel_size // 2
    k = jnp.exp(-(x**2) / (2 * sigma**2))
    return k / jnp.sum(k)


def apply_blur(img: jnp.ndarray, kernel: jnp.ndarray):
    B, H, W, C = img.shape
    kH = kernel[:, None]
    kW = kernel[None, :]

    def blur_chan(channel):
        b1 = jax.lax.conv_general_dilated(
            channel[None, ..., None], kH[:, :, None, None],
            window_strides=(1,1), padding='SAME',
            dimension_numbers=('NHWC','HWIO','NHWC')
        )
        b2 = jax.lax.conv_general_dilated(
            b1, kW[:, :, None, None],
            window_strides=(1,1), padding='SAME',
            dimension_numbers=('NHWC','HWIO','NHWC')
        )
        return b2[0, ..., 0]

    return jnp.stack([
        jnp.stack([blur_chan(img[b, ..., c]) for c in range(C)], axis=-1)
        for b in range(B)
    ], axis=0)


def random_gaussian_blur(
    key: Any,
    img: jnp.ndarray,                 # [B, V, H, W, C]
    min_sigma: float = 0.1,
    max_sigma: float = 1.0,
    kernel_size: int = 5,
    p: float = 0.25,
    time_invar: bool = False
) -> jnp.ndarray:
    B, V, H, W, C = img.shape
    flat = img.reshape((B*V, H, W, C))

    k_blur, k_sig = jax.random.split(key, 2)

    n = B if time_invar else B * V
    flags = jax.random.uniform(k_blur, (n,)) < p
    if time_invar:
        flags = jnp.repeat(flags, V)
    sigs = jax.random.uniform(k_sig, (n,), minval=min_sigma, maxval=max_sigma)
    if time_invar:
        sigs = jnp.repeat(sigs, V)

    def _maybe_blur(im, s, f):
        def do_blur(_):
            kern = gaussian_kernel_1d(s, kernel_size)
            return apply_blur(im[None], kern)[0]
        return jax.lax.cond(f, do_blur, lambda _: im, operand=None)

    out_flat = jax.vmap(_maybe_blur)(flat, sigs, flags)
    out = jnp.clip(out_flat, -1.0, 1.0)
    return out.reshape((B, V, H, W, C))


def jitter_and_blur(
    key: Any,
    x: jnp.ndarray,                   # [B, V, H, W, 3]
    p_apply_jit: float = 0.8,
    p_apply_blur: float = 0.25,
    d_bright: float = 0.2,
    d_cont: float = 0.2,
    d_sat: float = 0.2,
    d_hue: float = 0.05,
    min_sigma: float = 0.1,
    max_sigma: float = 1.0,
    kernel_size: int = 5,
    time_invar: bool = False
) -> jnp.ndarray:
    jk, bk = jax.random.split(key)
    x1 = color_jitter(
        jk, x, p_apply_jit, d_bright, d_cont, d_sat, d_hue, time_invar=time_invar
    )
    x2 = random_gaussian_blur(
        bk, x1, min_sigma, max_sigma, kernel_size, p_apply_blur, time_invar=time_invar
    )
    return x2


@jax.jit 
def consistent_rotate_flip(key: Any,
                           videos: jnp.ndarray) -> jnp.ndarray:
    """
    Apply the same random rotation (0°, 90°, 180°, 270°),
    horizontal flip, and vertical flip to each video in the batch.

    Args:
      key:    PRNGKey
      videos: jnp.ndarray of shape (B, V, H, W, C)

    Returns:
      jnp.ndarray of same shape, dtype=float32
    """
    B, V, H, W, C = videos.shape

    # make B independent subkeys
    subkeys = jax.random.split(key, B + 1)[1:]  # shape (B,)

    def _transform_video(subkey, vid):
        # subkey: PRNGKey, vid: (V, H, W, C)

        # split into three streams: rotation, hflip, vflip
        sk_rot, sk_hflip, sk_vflip = jax.random.split(subkey, 3)

        # rotation index {0,1,2,3}
        rot_k = jax.random.randint(sk_rot, (), 0, 4)      # int32 scalar

        # booleans for flips
        do_hflip = jax.random.bernoulli(sk_hflip, 0.5)    # bool scalar
        do_vflip = jax.random.bernoulli(sk_vflip, 0.5)    # bool scalar

        # 1) rotation over axes (H,W)
        def r0(x): return x
        def r1(x): return jnp.rot90(x, k=1, axes=(1, 2))
        def r2(x): return jnp.rot90(x, k=2, axes=(1, 2))
        def r3(x): return jnp.rot90(x, k=3, axes=(1, 2))
        vid = jax.lax.switch(rot_k, (r0, r1, r2, r3), vid)

        # 2) horizontal flip over W axis
        vid = jax.lax.cond(do_hflip,
                           lambda x: x[:, :, ::-1, :],
                           lambda x: x,
                           vid)

        # 3) vertical flip over H axis
        vid = jax.lax.cond(do_vflip,
                           lambda x: x[:, ::-1, :, :],
                           lambda x: x,
                           vid)

        return vid.astype(jnp.float32)

    # vmap over batch dimension
    return jax.vmap(_transform_video, in_axes=(0, 0))(subkeys, videos)



