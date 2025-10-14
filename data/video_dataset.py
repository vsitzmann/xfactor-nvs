import sys
import os
import io
import logging
import gzip
import json
from functools import partial
from pathlib import Path
from pprint import pprint
import grain
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
import re
import threading
from typing import Optional
from pathlib import Path
from typing import List
from multiprocessing import Value, RLock, Manager, current_process
import ctypes
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import random 


ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
CACHE_DIR = os.environ.get("VD_CACHE_DIR")
if not CACHE_DIR:
  CACHE_DIR = "vd_cache/"
else:
  if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


# Use nest_asyncio to allow asyncio in Jupyter Notebook


def load_json(file_name):
  with open(file_name) as json_file:
    data = json.load(json_file)
  return data 

def save_json(data, file_name):
  with open(file_name, 'w') as json_file:
      json.dump(data, json_file, indent=2)


def center_crop_resize(im: np.ndarray, HW: int = 256) -> np.ndarray:
    """
    Center crop a square from an HWC numpy array and resize to (HW, HW)
    using pure NumPy bilinear interpolation (vectorized).
    """

    if not isinstance(im, np.ndarray):
      im = np.asarray(im)
      
    assert im.ndim == 3, f"expected HWC image, got {im.shape}"
    H, W, C = im.shape

    if H == W == HW:
        return im

    # ---- center crop ----
    N = min(H, W)
    dH = (H - N) // 2
    dW = (W - N) // 2
    cropped = im[dH:dH+N, dW:dW+N, :]  # (N, N, C)

    if N == HW:
        return cropped

    # ---- precompute interpolation indices/weights ----
    coords = np.linspace(0, N - 1, HW)

    x0 = np.floor(coords).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, N - 1)
    dx = (coords - x0)[None, :, None]  # (1, HW, 1)

    y0 = np.floor(coords).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, N - 1)
    dy = (coords - y0)[:, None, None]  # (HW, 1, 1)

    # gather four neighbors with advanced indexing
    Ia = cropped[y0[:, None], x0[None, :]]  # (HW, HW, C)
    Ib = cropped[y0[:, None], x1[None, :]]
    Ic = cropped[y1[:, None], x0[None, :]]
    Id = cropped[y1[:, None], x1[None, :]]

    # bilinear interpolation
    top = (1 - dx) * Ia + dx * Ib
    bot = (1 - dx) * Ic + dx * Id
    out = (1 - dy) * top + dy * bot  # (HW, HW, C)

    return out.astype(np.uint8)


##################################################################
######################## Frame samplers ##########################
##################################################################

_GLOBAL_STEP_TRACKER_MANAGER = None

def _get_global_manager():
    """
    Lazily create a single Manager in the main process and return it.
    This avoids storing the Manager object on each StepTracker instance,
    which would cause pickling errors when the instance is serialized.
    """
    global _GLOBAL_STEP_TRACKER_MANAGER
    if _GLOBAL_STEP_TRACKER_MANAGER is None:
        # Only create manager in the main process. StepTracker should be
        # constructed in main before workers spawn.
        if current_process().name != "MainProcess":
            raise RuntimeError("Global Manager must be created in the main process")
        _GLOBAL_STEP_TRACKER_MANAGER = Manager()
    return _GLOBAL_STEP_TRACKER_MANAGER


class StepTracker:
    """
    Picklable step tracker safe to pass (step_tracker.get_step) into spawn-based workers.
    Uses a single global Manager (created in main) to allocate proxies, but does NOT
    store the Manager object on the instance (so pickling doesn't try to serialize it).
    """

    def __init__(self, initial: int = 0):
        mgr = _get_global_manager()          # created only in main process
        # store only the proxies (these are picklable)
        self._val = mgr.Value('q', int(initial))  # signed 64-bit proxy
      
        self._lock = mgr.Lock()                    # proxy lock
      
    def set_step(self, step: int) -> None:
        with self._lock:
            self._val.value = int(step)

    def get_step(self) -> int:

        with self._lock:
            return int(self._val.value)
          

def _conseq_frame_sampler(num_frames, num_samples, min_step, max_step):

  assert num_frames >= num_samples

  max_step = min(max_step, num_frames // (num_samples-1))
  max_step = max(min_step, max_step)
  
  f = np.random.randint(min_step, max_step+1, size=(num_samples-1, ), dtype=np.int32)

  f = np.cumsum(f)

  max_bound = max(num_frames - f[-1], 1)
  
  f0 = np.random.randint(0, max_bound, size=(1, ), dtype=np.int32)

  f = f0[0] + f 

  f = np.concatenate((f0, f), axis=0)

  f = np.minimum(f, num_frames-1)
  
  ff = np.flip(f, axis=-1)

  fmask = (1 * (np.random.uniform(size=(1, )) > 0.5)).astype(np.int32)

  f = (1 - fmask)[0] * f + fmask[0] * ff 
   
  return f


  

def _window_frame_sampler(
    num_frames,
    num_samples,
    max_win_len,
    rng,
    min_win_len=None,
    inclusive=False,
    upweight=False,
    gap_sigma=None,
    equal=False):
  
    assert num_samples >= 2
    assert num_frames >= num_samples

    max_win_len = min(num_frames, max_win_len)

    if min_win_len is None:
        min_win_len = num_samples
    else:
        min_win_len = max(min_win_len, num_samples)

    assert num_frames >= min_win_len
    assert max_win_len >= num_samples


    # Choose window length
    possible_win_lens = np.arange(min_win_len, max_win_len + 1)

    # Upweight larger baselines
    if upweight:
        win_weights = np.linspace(0, 1, len(possible_win_lens))
        win_weights = 0.8 * np.exp( -1.0 * (((win_weights-0.7)/0.25)**2)) + 0.2
        win_weights = win_weights / win_weights.sum()
        win_weights[-1] = 1 - win_weights[:-1].sum()  # ensure total 1
        win_len = rng.choice(possible_win_lens, p=win_weights)
    else:
        win_len = rng.choice(possible_win_lens)

    win_0 = rng.integers(0, num_frames - win_len + 1)
    win_1 = win_0 + win_len - 1

    if equal or gap_sigma is None:
        # Original behavior
        if num_samples > 2:

            if not equal:
              if inclusive:
                  candidate_range = np.arange(win_0, win_1 + 1)
              else:
                  candidate_range = np.arange(win_0 + 1, win_1)
              
              interior = rng.choice(candidate_range, size=(num_samples - 2), replace=False)
              f = np.sort(np.asarray([win_0] + interior.tolist() + [win_1], dtype=np.int32))
              
            else:
              f = np.linspace(win_0, win_1, num=num_samples, dtype=int)
        else:
            f = np.asarray([win_0, win_1], dtype=np.int32)
      
    else:
      # Sample to ensure that point which will be used as reference (the point which as the minimal maximum 
      # distance from the end points), has a minimal maximum distance no greater than (1 + gap_sigma) * window_length//2
      
      alpha = 1.0 + gap_sigma
      spread = win_1 - win_0
      n = num_samples - 2  # number of interior points

      if n == 0:
        f = np.asarray([win_0, win_1], dtype=np.int32)
      else:       
        # compute integer bounds for central point
        low  = int(np.ceil(win_1 - alpha*spread/2))
        high = int(np.floor(win_0 + alpha*spread/2))

        if high < low:
          x_central = np.arange(win_0 + 1, win_1)
          x_central = x_central[x_central.shape[0]//2, None]

        else:
          x_central = rng.choice(np.arange(low, high+1), size=1, replace=False)

        # remaining interior points (unique, exclude central point and endpoints)
        remaining_pool = np.setdiff1d(np.arange(win_0+1, win_1), x_central)
        if n > 1:
            remaining = rng.choice(remaining_pool, size=n-1, replace=False)
        else:
            remaining = np.array([], dtype=np.int32)

          
        # combine, sort, and add endpoints
        interior = np.sort(np.concatenate([x_central, remaining]))
        f = np.concatenate([[win_0], interior, [win_1]])

      
    # Flip 50% of time
    if rng.random() > 0.5:
        f = np.flip(f)

    return f
  

def get_conseq_frame_sampler(num_samples, min_step, max_step):

  sampler = lambda num_frames: _conseq_frame_sampler(num_frames, num_samples, min_step, max_step)

  return sampler


def get_window_frame_sampler(num_samples, max_win_len, min_win_len=None, inclusive=False):

  sampler = lambda num_frames: _window_frame_sampler(num_frames, num_samples, max_win_len, min_win_len, inclusive=inclusive)

  return sampler


def make_dynamic_window_sampler(
    train_step_fn,    # callable that returns current step
    num_samples,
    start_max,
    end_max,
    total_iters,
    min_win_len=None,
    upweight=False,
    gap_sigma=None,
    inclusive=False,
    equal=False,
    delay=0
):
    def sampler(num_frames, rng):
        step = train_step_fn()
        if step < delay:
            curr_max = start_max
        else:
            ramp_step = min(step - delay, total_iters)
            frac = ramp_step / float(total_iters)
            curr_max = int(start_max + frac * (end_max - start_max))

        #print(step, flush=True)
        return _window_frame_sampler(
            num_frames,
            num_samples,
            rng = rng,
            max_win_len=curr_max,
            min_win_len=min_win_len,
            upweight=upweight,
            gap_sigma=gap_sigma,
            equal=equal,
            inclusive=inclusive
        )
    return sampler






'''
==================================================================================================================================
================================================== Init Datasets =================================================================
==================================================================================================================================
'''

CO3DV2_SEQ_NAMES = ["apple", "tv", "microwave", "parkingmeter", "baseballglove", "baseballbat", "hotdog", "frisbee", "pizza",
"kite", "toybus", "sandwich", "toyplane", "car", "skateboard", "cup", "donut", "stopsign", "bicycle", "banana", "handbag",
"motorcycle", "cake", "toytrain", "couch", "broccoli", "bottle", "toaster", "wineglass", "hairdryer", "bowl", "carrot",
"laptop", "hydrant", "umbrella", "bench", "suitcase", "toytruck", "ball", "orange", "keyboard", "cellphone", "toilet",
"vase", "backpack", "mouse", "chair", "plant", "teddybear", "remote", "book"]


def init_CO3DV2(source_dir, split, min_length=None):
  
  #split_path = os.path.join(source_dir, split)
  
  if min_length is None:
    min_length = 10

  min_length = max(min_length, 10)

  frame_file_train = os.path.join(CACHE_DIR, "co3dv2_train_frame_dict.json")
  frame_file_test = os.path.join(CACHE_DIR, "co3dv2_test_frame_dict.json")

  if split == "train":
    frame_file = frame_file_train
  else:
    frame_file = frame_file_test 
    
  if not os.path.exists(frame_file):
    print("No cached frame dict found, creating...", flush=True)
    
    # Create list of directories
    #dlist = os.listdir(split_path)
     
    frame_dict_train = {}
    frame_dict_test = {} 
    
    for j in tqdm(range(len(CO3DV2_SEQ_NAMES))):
      
      seq_path = os.path.join(source_dir, CO3DV2_SEQ_NAMES[j])
      
      if os.path.isdir(seq_path):

        seqs = [s for s in os.listdir(seq_path) if s and s[0].isdigit()]
        #seqs = [s for s in seqs if s[0].isdigit()]

        seqs = sorted(seqs)
        
        num_train = int(0.9*len(seqs))

        #print(len(seqs), file=sys.stderr, flush=True)
        
        for l in range(len(seqs)):

          seq_id = seqs[l] 

          image_dir = os.path.join(seq_path, seq_id, "images")

          imlist = os.listdir(image_dir)

          imlist = sorted(imlist, key=lambda f: int(f[5:11]))

          imlist[:] = [os.path.join(image_dir, name) for name in imlist]

          if l < num_train:  
            frame_dict_train[seq_id] = imlist
          else:
            frame_dict_test[seq_id] = imlist

    save_json(frame_dict_train, frame_file_train)
    save_json(frame_dict_test, frame_file_test)


  frame_dict = load_json(frame_file)

  filtered_frame_dict = {k: v for k, v in frame_dict.items() if len(v) >= min_length}

  data = list(filtered_frame_dict.keys())
  frames = filtered_frame_dict

  print("CO3Dv2 {} loaded {} videos...".format(split, len(data)), flush=True)

  return data, frames

  
def init_RE10K(source_dir, split, min_length=None):

  split_path = os.path.join(source_dir, split)

  if min_length is None:
    min_length=10

  min_length = max(min_length, 10)

  frame_file = os.path.join(CACHE_DIR, "re10k_{}_frame_dict.json".format(split))

  if not os.path.exists(frame_file):
    print("No cached frame dict found, creating...", flush=True)
    
    # Create list of directories
    dlist = os.listdir(os.path.join(split_path, "images"))

    frame_dict = {}

    skipped_vids = {
      "not a directory": 0,
      # "metadata not found": 0,
      # "frame length mismatch": 0,
    }       # {reason: n_vids}
    for vid_id in tqdm(dlist):
      # --------------------------------------------------------------------------------
      # Skip video if not a directory.
      # --------------------------------------------------------------------------------

      im_path = os.path.join(split_path, "images", vid_id)
      
      if not os.path.isdir(im_path):
        skipped_vids["not a directory"] += 1
        continue      


      imlist=os.listdir(im_path)
      imlist = sorted(imlist, key=lambda x: int(re.search(r'0*(\d+)\.png$', x).group(1)))

      imlist[:] = [os.path.join(im_path, name) for name in imlist]

      frame_dict[vid_id] = imlist

    print(f"Skipped vids: {skipped_vids}", flush=True)  # {reason: n_vids}

    save_json(frame_dict, frame_file)

  else:
    frame_dict = load_json(frame_file)

  filtered_frame_dict = {k: v for k, v in frame_dict.items() if len(v) >= min_length}

  data = list(filtered_frame_dict.keys())
  frames = filtered_frame_dict 
  
  print("RE10K {} loaded {} videos...".format(split, len(data)), flush=True)

  return data, frames


def init_DL3DV(source_dir, split, min_length=None):
  
  if min_length is None:
    min_length = 10

  min_length = max(min_length, 10)

  split_path = os.path.join(source_dir, "training_256")

  bsplit_path = os.path.join(source_dir, "test_256", "benchmark_960p")

  frame_file = os.path.join(CACHE_DIR, "dl3dv_{}_frame_dict.json".format(split))
    
  if not os.path.exists(frame_file):

    frame_dict = {}

    bmark_list = os.listdir(bsplit_path)
    part_list = os.listdir(split_path)

    
    # ========================================================================================
    # Load Videos
    # ========================================================================================
    skipped_vids = {
      "not in split": 0,
      "not a directory": 0,
      "too short": 0,
      "frame length mismatch": 0,
      "transforms.json not found": 0,
      "unknown camera model": 0,
    }       # {reason: n_vids}
    for p in part_list:
      p_path = os.path.join(split_path, p)

      if not os.path.isdir(p_path):
        continue 
        
      dlist = os.listdir(p_path)
      
      for l in tqdm(range(len(dlist))):

        # --------------------------------------------------------------------------------
        # Skip if not in split
        # --------------------------------------------------------------------------------
        vid_id = dlist[l]

        if split == "test" and vid_id not in bmark_list:
          skipped_vids["not in split"] += 1
          continue
        elif split == "train" and vid_id in bmark_list:
          skipped_vids["not in split"] += 1
          continue

        # --------------------------------------------------------------------------------
        # Skip if not a directory
        # --------------------------------------------------------------------------------
        im_dir = os.path.join(p_path, vid_id, "images_8")
        
        if not os.path.isdir(im_dir):
          skipped_vids["not a directory"] += 1
          continue
        

        # --------------------------------------------------------------------------------
        # Skip video if too short.
        # --------------------------------------------------------------------------------
        imlist = os.listdir(im_dir)
        imlist = sorted(imlist, key=lambda x: int(re.search(r'frame_(\d+)\.png$', x).group(1))) 
        
        imlist[:] = [os.path.join(im_dir, name) for name in imlist]

        # --------------------------------------------------------------------------------
        # Skip video if missing transforms.json
        # --------------------------------------------------------------------------------
        try:
          metadata = load_json(
            os.path.join(p_path, vid_id, "transforms.json")
          )
        except FileNotFoundError:
          skipped_vids["transforms.json not found"] += 1
          continue


        frame_dict[vid_id] = imlist
          
      
      print(f"Skipped vids: {skipped_vids}", flush=True)  # {reason: n_vids}

    

    save_json(frame_dict, frame_file)

  else:
    frame_dict = load_json(frame_file)

  filtered_frame_dict = {k: v for k, v in frame_dict.items() if len(v) >= min_length}

  data = list(filtered_frame_dict.keys())
  frames = filtered_frame_dict 
  
  print("DL3DV {} loaded {} videos...".format(split, len(data)), flush=True)

  return data, frames 
  
  
def init_MvImgNet(source_dir, split, min_length=None):
  source_dir = Path(source_dir)

  if min_length is None:
      min_length = 10

  min_length = max(min_length, 10)

  assert split in ["train", "test"]

  frame_file = os.path.join(CACHE_DIR, "mvimgnet_{}_frame_dict.json".format(split))

  if not os.path.exists(frame_file):

    frame_dict = {}
    
    # Avoid repeated lookups
    try:
        category_dirs = [p for p in source_dir.iterdir() if p.is_dir()]
    except Exception as e:
        raise RuntimeError(f"Failed to list categories in {source_dir}: {e}")

    skipped_vids = {
        "too short": 0,
        "no_colmap_recon": 0,
        "multiple_colmap_recons": 0,
    }


    for category_path in tqdm(category_dirs):
        try:
            scene_dirs = [p for p in category_path.iterdir() if p.is_dir()]
        except Exception:
            continue  # Skip categories that can't be read

        scene_dirs = sorted(scene_dirs)
        # Split train/test
        n = len(scene_dirs)
      
        scene_dirs = scene_dirs[:int(0.9 * n)] if split == "train" else scene_dirs[int(0.9 * n):]

        for scene_path in scene_dirs:
            sparse_dir = scene_path / "sparse"

            # if not pretrain:
            if False:
                try:
                    recon_ids = [p for p in sparse_dir.iterdir() if p.is_dir()]
                except Exception:
                    skipped_vids["no_colmap_recon"] += 1
                    continue

                if len(recon_ids) == 0:
                    skipped_vids["no_colmap_recon"] += 1
                    continue
                elif len(recon_ids) > 1:
                    skipped_vids["multiple_colmap_recons"] += 1
                    continue

            frames_dir = scene_path / "images"

            try:
                num_frames = sum(1 for f in frames_dir.iterdir() if f.suffix.lower() == ".jpg")
            except Exception:
                continue  # Skip missing or unreadable images dir

            vid_id = f"{category_path.name}/{scene_path.name}"

            frames_dir = source_dir / vid_id / "images"
            imlist = sorted([f for f in os.listdir(frames_dir) if len(f) == 7], key=lambda x: int(x[:3]))

            imlist[:] = [os.path.join(frames_dir, name) for name in imlist]

            frame_dict[vid_id] = imlist



    save_json(frame_dict, frame_file)

  else:
    frame_dict = load_json(frame_file)

  filtered_frame_dict = {k: v for k, v in frame_dict.items() if len(v) >= min_length}

  data = list(filtered_frame_dict.keys())
  frames = filtered_frame_dict 

  print("MVImgNet {} loaded {} videos...".format(split, len(data)), flush=True)

  return data, frames 


'''
==============================================================================================================================
===================================================== Dataset Sources ========================================================
==============================================================================================================================
'''

def _reset_rng_state(
    rng: np.random.Generator, op_seed: int, index: int
) -> None:
  state = rng.bit_generator.state
  state["state"]["counter"] = np.array([0, 0, op_seed, index], dtype=np.uint64)
  state["buffer"] = np.array([0, 0, 0, 0], dtype=np.uint64)
  state["buffer_pos"] = 4
  state["has_uint32"] = 0
  state["uinteger"] = 0
  rng.bit_generator.state = state
  
class RngPool:
  """RNG pool."""

  def __init__(self, seed: int):
    self._seed = seed
    self._generator_cache = []
    self._lock = threading.Lock()

  def __reduce__(self):
    return (RngPool, (self._seed,))

  def acquire_rng(self, index: int, *, op_seed: int = 0) -> np.random.Generator:
    """Acquire RNG."""
    with self._lock:
      if self._generator_cache:
        rng = self._generator_cache.pop()
      else:
        rng = np.random.Generator(np.random.Philox(self._seed))
    _reset_rng_state(rng, op_seed=op_seed, index=index)
    return rng

  def release_rng(self, rng: np.random.Generator):
    with self._lock:
      self._generator_cache.append(rng)
      

class VideoAmalgamSource(grain.sources.RandomAccessDataSource):


  def __init__(self, source_dirs, split, frame_samplers, min_lengths, weights=None, seed=0):

    # source_dirs, frame samplers, min_lengths are dicts whose keys are the dataset names: 

    self._rng_pool = RngPool(seed)
    
    self.frame_samplers = frame_samplers

    _data_inter = [] 
    _data_weight = []
    _frames = {}

    max_length = 0 

    for i, ds in enumerate(source_dirs.keys()):

      if ds == "mvimgnet":
        vid_init = init_MvImgNet
      elif ds == "co3dv2":
        vid_init = init_CO3DV2
      elif ds == "re10k":
        vid_init = init_RE10K
      elif ds == "dl3dv":
        vid_init = init_DL3DV

      data, frames = vid_init(source_dirs[ds], split, min_lengths[ds])

      num_elements = len(data)

      if num_elements > max_length:
        max_length = num_elements
              
      dataset_id = [ds]*num_elements

      data = list(zip(dataset_id, data))

      _frames[ds] = frames 

      _data_inter.append(data)

      if weights is not None:
        _data_weight.append(weights[ds])
      else:
        _data_weight.append(1)
        
    # Repeat smaller datasets so that they match the size of the largest dataset 
    _data = []
    for l in range(len(_data_inter)):

      data = _data_inter[l] 

      num_reps = (max_length // len(data)) + 1 

      data = data * num_reps 

      data = data[:max_length]

      data = data * _data_weight[l]
      
      _data = _data + data 


    self._data = _data
    self._frames = _frames 

  
  def __len__(self):
    return len(self._data)

  def __getitem__(self, idx):
    
    ds, vid_id = self._data[idx]

    frame_list = self._frames[ds][vid_id]

    num_frames = len(frame_list)

    rng = self._rng_pool.acquire_rng(idx)
    
    frame_ind = self.frame_samplers[ds](num_frames, rng)

    self._rng_pool.release_rng(rng)

    frames = [center_crop_resize(Image.open(frame_list[frame_ind[i]])) for i in range(frame_ind.shape[0])]
    
    frames = np.stack(frames, axis=0)

    #frames = (2.0 * ( (frames / 255.0) - 0.5)).astype(np.float32)
  
    frame_ind = frame_ind - frame_ind[0]

    example = {"image": frames,
               "frame_ind": frame_ind,
               "vid_id": vid_id}
  
    return example  
    

  
def dynamic_vid_amalgam_dataset(cfg, samplers):

  datasets = cfg.DATASETS

  source_dirs = {}
  min_len = {}
  weights = {}
  
  for i, ds in enumerate(datasets.keys()):

    source_dirs[ds] = datasets[ds]["path"]

    if datasets[ds]["min_win_len"] is not None:
      min_len[ds] = max(datasets[ds]["min_win_len"], cfg.NUM_SAMPLES) + 1
    else:
      min_len[ds] = cfg.NUM_SAMPLES + 1

    if "data_weight" in datasets[ds] and datasets[ds]["data_weight"] is not None:
      weights[ds] = datasets[ds]["data_weight"]
    else:
      weights[ds] = 1 
  
  vid_source_train = VideoAmalgamSource(source_dirs, "train", samplers, min_len, weights)
  vid_source_test = VideoAmalgamSource(source_dirs, "test", samplers, min_len, weights)


  # Lots of worker buffers > 8, and prefetch buffer size >= 8*batch_size 
  train_dataset = grain.DataLoader(data_source=vid_source_train,
                                  operations=[grain.transforms.Batch(batch_size=cfg.BATCH_SIZE, drop_remainder=True)],
                                  shard_options=grain.sharding.NoSharding(),
                                  sampler=grain.samplers.IndexSampler(num_records=len(vid_source_train), num_epochs=1000, shuffle=True, seed=cfg.DATA_SEED),
                                  worker_count=cfg.NUM_WORKERS,
                                  worker_buffer_size=cfg.WORKER_BUFFER_SIZE,
                                  read_options=grain.ReadOptions(num_threads=cfg.NUM_THREADS, prefetch_buffer_size=cfg.PREFETCH_FACTOR*cfg.BATCH_SIZE),
                                  enable_profiling=True
                                  )

  test_dataset = grain.DataLoader(data_source=vid_source_test,
                                  operations=[grain.transforms.Batch(batch_size=cfg.BATCH_SIZE, drop_remainder=True)],
                                  shard_options=grain.sharding.NoSharding(),
                                  sampler=grain.samplers.IndexSampler(num_records=len(vid_source_test), num_epochs=1000, shuffle=True, seed=cfg.DATA_SEED+1),
                                  worker_count=cfg.NUM_WORKERS,
                                  worker_buffer_size=cfg.WORKER_BUFFER_SIZE,
                                  read_options=grain.ReadOptions(num_threads=cfg.NUM_THREADS, prefetch_buffer_size=cfg.PREFETCH_FACTOR*cfg.BATCH_SIZE),
                                  enable_profiling=True
                                  )


  return train_dataset, test_dataset


