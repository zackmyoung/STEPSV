#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEPSV.py

STEPS-V: Sequence-based Temporal Extraction of Persistent Steps in vertical GPS time series.

Author
------
Zachary M. Young, PhD
ORCiD: https://orcid.org/0000-0002-5487-9183

Updated: April 7, 2026

Overview
--------
This script provides the full STEPS-V workflow including:
  1. Build a training dataset from station time series and known step epochs
  2. Train fold-based validation models
  3. Train a final model on the full dataset
  4. Run inference on new station files

Required Python packages
------------------------
STEPS-V requires a Python environment with the following packages available:
  - torch
  - numpy
  - scikit-learn
  - tqdm
  - matplotlib

Recommended install approach
----------------------------
For most users, the simplest approach is to create a dedicated Python environment
for STEPS-V and install the required packages there. PyTorch should be installed
using the build appropriate for your system, either CPU-only or CUDA-enabled.

Full installation guidance is provided in the README.

Expected inputs
---------------
Station files are expected as whitespace-delimited ASCII files with columns:
  [decimal_year  displacement  uncertainty  antenna_correction]

The fourth column is currently reserved for antenna-related information.
It is retained for file-format compatibility and future STEPS-V development,
but is not used by the current model. It may currently contain any value.


IMPORTANT:

The displacement column is expected to contain preprocessed vertical
displacement values prepared for STEPS-V. In the standard STEPSV v1.0
workflow, inputs are detrended, cleaned for outliers, and corrected for
NTAOL effects before dataset building or inference.



Quick examples
--------------
The examples below reflect the default/example workflow used for STEPSV V1.0.
The preferred packaged inference model is provided locally as:
  ./STEPSV_final_model_epoch_070.pt
and the matching scaler is:
  ./STEPSV_scaler.pkl

Paths and file names can be changed as needed.

Build dataset:
  python STEPSV_v1.00.py --mode build_dataset \
    --data_dir ./ML_train_stats \
    --step_epochs_file ./ML_step_epochs.txt \
    --dataset_out ./train_v1_dataset.pt \
    --metadata_out ./metadata_v1.npy \
    --scaler_out ./STEPSV_scaler.pkl \
    --apply_scaling 1 \
    --make_scaler 1

Train fold models:
  python STEPSV_v1.00.py --mode train_folds \
    --dataset_pt ./train_v1_dataset.pt \
    --train_out_dir ./models_v1_folds \
    --version_tag STEPSV_v1 \
    --k_folds 5 \
    --num_epochs 200 \
    --train_batch_size 256 \
    --val_batch_size 256 \
    --linear_size 16 \
    --hidden_size 32 \
    --num_layers 1 \
    --dropout 0 \
    --bidirectional 1 \
    --max_length 5500 \
    --pad_start 0 \
    --pad_end 0

Train final model:
  python STEPSV_v1.00.py --mode train_final \
    --dataset_pt ./train_v1_dataset.pt \
    --train_out_dir ./models_v1_final \
    --version_tag STEPSV_v1 \
    --num_epochs 200 \
    --train_batch_size 256 \
    --val_batch_size 256 \
    --linear_size 16 \
    --hidden_size 32 \
    --num_layers 1 \
    --dropout 0 \
    --bidirectional 1 \
    --max_length 5500 \
    --pad_start 0 \
    --pad_end 0

Run inference:
  python3 STEPSV_v1.00.py \
    --mode inference \
    --data_dir "./GPS_data" \
    --output_dir "./example_inference_out" \
    --model "./STEPSV_final_model_epoch_070.pt" \
    --scaler_path "./STEPSV_scaler.pkl" \
    --apply_scaling 1 \
    --max_length 5500 \
    --pad_start 0 \
    --pad_end 0 \
    --linear_size 16 \
    --hidden_size 32 \
    --num_layers 1 \
    --dropout 0 \
    --bidirectional 1 \
    --mc_samples 0 \
    --infer_batch_size 128 \
    --infer_workers 4

Inference outputs
-----------------
Inference writes one NumPy .npz file per station to the specified output directory.
Files are named:

  STATION_inference.npz

These files contain:
  time      decimal-year time vector
  prob      model step probability
  prob_std  optional probability uncertainty when Monte Carlo dropout is used

The .npz outputs are intended to be parsed with the MATLAB helper scripts
`load_inference_npz` and `pick_steps_from_prob` for postprocessing and step selection.

Example end-to-end MATLAB workflows are provided in:
  run_examples.m

Notes
-----
- Use the README for full setup instructions and argument descriptions.
- The model checkpoint, scaler, and feature configuration must remain consistent.
- Some parser defaults are more general than the settings shown here; the examples above reflect
  the practical STEPSV V1.0 workflow used in testing and evaluation.

Disclaimer
----------
STEPS-V is provided as a research tool for automated step detection in vertical GPS time series.
Model outputs should be independently reviewed before scientific, operational, engineering, or other
decision-making use. Users are responsible for validating results within their own workflow.
"""

import argparse
import os
import time  
import pickle
from concurrent.futures import ProcessPoolExecutor 
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import random
import math 
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler
import json
import datetime 

# ----------------------------------------------------------------------
# Device / global torch settings
# ----------------------------------------------------------------------
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
print("Using device:", device)


# ----------------------------------------------------------------------
# Distributed (DDP) helpers
# ----------------------------------------------------------------------
class _DDPState:
    """Lightweight container for DDP runtime state."""
    def __init__(self, enabled: bool, rank: int = 0, local_rank: int = 0, world_size: int = 1, device=None):
        self.enabled = bool(enabled)
        self.rank = int(rank)
        self.local_rank = int(local_rank)
        self.world_size = int(world_size)
        if device is None:
            if self.enabled and torch.cuda.is_available():
                device = torch.device(f"cuda:{self.local_rank}")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_ddp(self) -> bool:
        return self.enabled

def _set_seeds(seed: int, rank: int = 0):
    s = int(seed) + int(rank)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def ddp_init_if_needed(args):
    """Initialize torch.distributed if launched with torchrun."""
    use_ddp_flag = getattr(args, "ddp", False)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    enabled = bool(use_ddp_flag) and world_size > 1 and torch.cuda.is_available()
    if enabled:
        backend = getattr(args, "ddp_backend", "nccl")
        dist.init_process_group(backend=backend, init_method="env://")
        torch.cuda.set_device(local_rank)

    # Safe performance knobs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    _set_seeds(getattr(args, "seed", 1234), rank=(rank if enabled else 0))
    return _DDPState(enabled, rank=rank, local_rank=local_rank, world_size=world_size if enabled else 1)



def ddp_mean(x, device=None):
    """All-reduce mean for scalar (float or 0-dim tensor) in DDP."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return float(x) if not isinstance(x, torch.Tensor) else float(x.detach().cpu().item())
    import torch.distributed as dist
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(x, torch.Tensor):
        t = x.detach().float().to(device)
        if t.numel() != 1:
            t = t.mean()
    else:
        t = torch.tensor(float(x), device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t = t / dist.get_world_size()
    return float(t.detach().cpu().item())


def _ddp_barrier_and_cleanup(state):
    """Sync ranks and destroy process group (safe even if already destroyed)."""
    if not getattr(state, "enabled", False):
        return
    import torch.distributed as dist
    try:
        dist.barrier()
    except Exception:
        pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass



def _adjust_num_workers(num_workers: int, state: _DDPState) -> int:
    if not state.enabled:
        return int(num_workers)
    return max(1, int(num_workers) // max(1, state.world_size))


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def compute_median_diff(time, disp, num_points):
    """Compute median difference before and after each epoch within a given
    number of data points.

    Windows (no overlap):
      before: [i-num_points, ..., i-1]
      after : [i, ..., i+num_points-1]
    Edge windows shrink gracefully.
    """
    disp = np.asarray(disp)
    n = disp.shape[0]
    out = np.zeros(n, dtype=np.float32)

    if n == 0 or num_points <= 0:
        return out

    # Vectorized interior (full windows on both sides)
    if n >= num_points:
        w = np.lib.stride_tricks.sliding_window_view(disp, num_points)  # (n-num_points+1, num_points)
        med = np.median(w, axis=1)

        lo = num_points
        hi = n - num_points  # inclusive
        if hi >= lo:
            # out[i] = median(disp[i:i+num_points]) - median(disp[i-num_points:i])
            out[lo:hi+1] = (med[lo:hi+1] - med[0:hi+1-lo]).astype(np.float32)

    # Left edge
    left_end = min(num_points, n)
    for i in range(0, left_end):
        bs = 0
        be = i
        a_s = i
        a_e = min(n, i + num_points)
        before_vals = disp[bs:be]
        after_vals = disp[a_s:a_e]
        before_med = np.median(before_vals) if before_vals.size else disp[i]
        after_med  = np.median(after_vals)  if after_vals.size  else disp[i]
        out[i] = after_med - before_med

    # Right edge
    right_start = max(n - num_points + 1, left_end)  # +1 because hi is inclusive
    for i in range(right_start, n):
        bs = max(0, i - num_points)
        be = i
        a_s = i
        a_e = n
        before_vals = disp[bs:be]
        after_vals = disp[a_s:a_e]
        before_med = np.median(before_vals) if before_vals.size else disp[i]
        after_med  = np.median(after_vals)  if after_vals.size  else disp[i]
        out[i] = after_med - before_med

    return out


def log_scale(data):
    """Signed log scaling: log(1 + |y|) * sign(y)."""
    return np.sign(data) * np.log1p(np.abs(data))


# ----------------------------------------------------------------------
# Scaler I/O helpers (feature-safe)
# ----------------------------------------------------------------------
def save_scaler_bundle(path, scaler, scaled_cols, n_total_features):
    """Save scaler plus metadata about which feature columns were scaled."""
    bundle = {
        "scaler": scaler,
        "scaled_cols": list(map(int, scaled_cols)),
        "n_total_features": int(n_total_features),
        "bundle_version": 1,
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_scaler_bundle(path):
    """Load scaler bundle. Supports both bundled dicts and raw sklearn scalers."""
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "scaler" in obj:
        return obj

    # Back-compat: raw sklearn scaler object
    n_in = getattr(obj, "n_features_in_", None)
    bundle = {
        "scaler": obj,
        "scaled_cols": list(range(int(n_in))) if n_in is not None else None,
        "n_total_features": int(n_in) if n_in is not None else None,
        "bundle_version": 0,
    }
    return bundle




def _append_loss_row_csv(csv_path, loss_rows):
    """Append one or more loss rows to a CSV file.

    loss_rows: iterable of [epoch, train_loss, val_loss, lr]
    """
    if csv_path is None or loss_rows is None:
        return
    if len(loss_rows) == 0:
        return
    with open(csv_path, "a") as f:
        for row in loss_rows:
            if row is None or len(row) < 4:
                continue
            epoch, train_loss, val_loss, lr = row[0], row[1], row[2], row[3]
            tl = float(train_loss) if train_loss is not None else float("nan")
            vl = float(val_loss) if val_loss is not None else float("nan")
            lr_f = float(lr) if lr is not None else float("nan")
            f.write(f"{int(epoch)},{tl:.8g},{vl:.8g},{lr_f:.8g}\n")

def dataloader_common_kwargs(num_workers: int, pin_memory: bool, prefetch_factor: int, persistent_workers: int):
    """Build DataLoader kwargs safely (prefetch_factor/persistent_workers only valid when num_workers > 0)."""
    kw = dict(num_workers=int(num_workers), pin_memory=bool(pin_memory))
    if int(num_workers) > 0:
        kw["prefetch_factor"] = int(prefetch_factor)
        kw["persistent_workers"] = bool(int(persistent_workers))
    return kw


def _dataloader_set_epoch(loader, epoch: int) -> None:
    """Safely set epoch on samplers that support it (e.g., DistributedSampler)."""
    sampler = getattr(loader, 'sampler', None)
    if sampler is not None and hasattr(sampler, 'set_epoch'):
        sampler.set_epoch(epoch)
        return
    batch_sampler = getattr(loader, 'batch_sampler', None)
    if batch_sampler is not None and hasattr(batch_sampler, 'set_epoch'):
        batch_sampler.set_epoch(epoch)

class BucketBatchSampler(torch.utils.data.Sampler):
    """Length-aware batching to reduce padding waste.

    Buckets samples by sequence length, then forms batches within each bucket.
    Optionally shuffles within buckets and shuffles batch order.

    DDP support: shards *batches* across ranks (rank takes every world_size-th batch) and
    drops extra batches so every rank has the same number of steps.
    """

    def __init__(
        self,
        lengths,
        batch_size: int,
        num_buckets: int = 50,
        shuffle: bool = True,
        shuffle_batches: bool = True,
        drop_last: bool = False,
        seed: int = 0,
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__(None)
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.num_buckets = int(max(1, num_buckets))
        self.shuffle = bool(shuffle)
        self.shuffle_batches = bool(shuffle_batches)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.world_size = int(max(1, world_size))
        self.rank = int(rank)
        self.epoch = 0

        self._sorted_indices = np.argsort(self.lengths, kind="mergesort")
        self._bucket_slices = np.array_split(self._sorted_indices, self.num_buckets)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        bs = self.batch_size

        for bucket in self._bucket_slices:
            if bucket.size == 0:
                continue
            idx = bucket.copy()
            if self.shuffle:
                rng.shuffle(idx)

            n_full = (idx.size // bs) * bs
            for s in range(0, n_full, bs):
                batches.append(idx[s:s+bs].tolist())

            if (not self.drop_last) and (n_full < idx.size):
                batches.append(idx[n_full:].tolist())

        if self.shuffle_batches:
            rng.shuffle(batches)

        if self.world_size > 1:
            n = (len(batches) // self.world_size) * self.world_size
            batches = batches[:n]
            batches = batches[self.rank::self.world_size]

        for b in batches:
            yield b

    def __len__(self):
        n_batches = int(np.ceil(len(self.lengths) / self.batch_size))
        if self.world_size > 1:
            return (n_batches // self.world_size)
        return n_batches





def _subset_lengths(subset):
    """Return a list of sequence lengths for a Dataset or torch.utils.data.Subset.

    We try to avoid calling __getitem__ for every sample (slow) by reading from the
    underlying dataset's cached `data` container when available.  In this project
    `data[i]` may be either:
      - a (T,F) numpy array / torch tensor, OR
      - a tuple/list where the first element is the (T,F) features array/tensor.
    """
    def _len_from_entry(entry):
        # entry can be (T,F) array/tensor OR tuple/list where entry[0] is (T,F)
        if isinstance(entry, (tuple, list)):
            entry = entry[0]
        # torch tensor / numpy array
        shp = getattr(entry, "shape", None)
        if shp is not None and len(shp) >= 1:
            return int(shp[0])
        # last resort: try python len()
        return int(len(entry))

    # torch.utils.data.Subset has `.dataset` and `.indices`
    base = getattr(subset, "dataset", None)
    idxs = getattr(subset, "indices", None)
    if idxs is None:
        base = subset
        idxs = range(len(subset))

    data = getattr(base, "data", None)
    if data is not None:
        return [_len_from_entry(data[i]) for i in idxs]

    # Fallback: call __getitem__
    out = []
    for i in idxs:
        item = subset[i] if base is subset else base[i]
        out.append(_len_from_entry(item[0]))
    return out

def read_step_epochs_file(step_epochs_file):
    """Read a step-epoch list into dict: station -> [epoch1, epoch2, ...].
    Expects whitespace-delimited lines like:
        STATION 2016.1234
    or optionally with extra columns (ignored).
    Ignores blank lines and lines starting with '#'.
    """
    step_epochs = {}
    if step_epochs_file is None:
        return step_epochs
    if not os.path.isfile(step_epochs_file):
        raise FileNotFoundError(f"step_epochs_file not found: {step_epochs_file}")
    with open(step_epochs_file, "r") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            sta = parts[0]
            try:
                epoch = float(parts[1])
            except ValueError:
                continue
            step_epochs.setdefault(sta, []).append(epoch)
    # ensure sorted epochs per station
    for sta in list(step_epochs.keys()):
        step_epochs[sta] = sorted(step_epochs[sta])
    return step_epochs

 

# ----------------------------------------------------------------------
# Station loading helper (for ProcessPoolExecutor)
# ----------------------------------------------------------------------

def load_station_data(station_file, data_dir, scaler, step_epochs):
    """Load a single station file and build features/labels.

    TIME PRECISION
    --------------
    Uses fixed-point time (int64 = round(dec_year * 10000)) for exact step labeling.

    Feature vector (12), in this order:
      0  t_rel_yrs                 years since first sample (physical time)
      1  t_pos                     0–1 position in record
      2  dt_yrs                    sample-to-sample spacing (years; first = median dt)
      3  sin(2π phase)             phase = fractional year
      4  cos(2π phase)
      5  sin(4π phase)
      6  cos(4π phase)
      7  disp                      mm (assumed detrended by upstream generator)
      8  sig                       mm
      9  median_diff_5             point-window median diff feature
      10 median_diff_15
      11 disp_cumsum_detrended     cumsum(disp) detrended (remove intercept+linear) + median-centered

    Notes
    -----
    - This is decade-agnostic: it does NOT normalize by any fixed calendar-year window.
    - Keep `scaler` argument for API compatibility; scaling is handled outside this loader.
    """
    station_path = os.path.join(data_dir, station_file)
    if not os.path.isfile(station_path):
        return None

    # Parse only needed columns: [time, disp, sig, ant]
    try:
        station_data = np.loadtxt(station_path, dtype=np.float64, usecols=(0, 1, 2, 3))
    except Exception as e:
        print(f"Failed to read {station_path}: {e}")
        return None

    station_name = os.path.basename(station_path).split('.')[0]

    time64 = station_data[:, 0].astype(np.float64, copy=False)
    disp   = station_data[:, 1].astype(np.float32, copy=False)
    sig    = station_data[:, 2].astype(np.float32, copy=False)
    ant    = station_data[:, 3].astype(np.float32, copy=False)  # kept for compatibility if needed elsewhere

    # Fixed-point time (exact at 4 decimals)
    time_i = np.rint(time64 * 10000.0).astype(np.int64)

    # Sanity checks
    if time_i.size >= 2 and (not np.all(np.diff(time_i) > 0)):
        print(f"Warning: Time array for {station_name} is not strictly increasing!")
    if time_i.size != np.unique(time_i).size:
        print(f"Warning: Duplicate time values in {station_name}!")

    # Median-diff features (points-based windows) -- keep existing behavior
    time_scales = (3,7,15 ,30 )
    median_diffs = np.column_stack([compute_median_diff(time64, disp, s) for s in time_scales]).astype(np.float32)

    # Time features
    t_rel_yrs = ((time_i - time_i[0]) / 10000.0).astype(np.float32, copy=False)



    span = float(t_rel_yrs[-1]) if t_rel_yrs.size else 0.0
    if span > 0.0:
        t_pos = (t_rel_yrs / span).astype(np.float32, copy=False)
    else:
        t_pos = np.zeros_like(t_rel_yrs, dtype=np.float32)

    dt_yrs = (np.diff(time_i) / 10000.0).astype(np.float32)
    if dt_yrs.size:
        dt0 = np.median(dt_yrs).astype(np.float32)
    else:
        dt0 = np.float32(0.0)
    dt_yrs = np.insert(dt_yrs, 0, dt0).astype(np.float32, copy=False)

    # Seasonal phase (fractional year) from fixed-point time (stable)
    # phase in [0,1): fractional part of decimal year
    frac_i = (time_i % 10000).astype(np.float32)
    phase = frac_i / 10000.0
    sin1 = np.sin(2.0 * np.pi * phase).astype(np.float32)
    cos1 = np.cos(2.0 * np.pi * phase).astype(np.float32)
#    sin2 = np.sin(4.0 * np.pi * phase).astype(np.float32)
#    cos2 = np.cos(4.0 * np.pi * phase).astype(np.float32)

    # Cumsum feature: cumsum(disp) then detrend that cumsum (remove intercept+linear) and median-center.
    # (Do NOT re-detrend disp here; generator already did.)
    csum = np.cumsum(disp).astype(np.float32)
    if csum.size >= 2:
        A = np.vstack([np.ones_like(t_rel_yrs, dtype=np.float32), t_rel_yrs]).T  # [1, t_rel]
        m = np.linalg.lstsq(A, csum, rcond=None)[0].astype(np.float32)
        csum_dt = csum - (A @ m).astype(np.float32)
    else:
        csum_dt = csum
    csum_dt = csum_dt - np.median(csum_dt).astype(np.float32)


    time_i = np.rint(time64 * 10000).astype(np.int64)          # canonical time grid
    t_rel_yrs = ((time_i - time_i[0]) / 10000.0).astype(np.float32)

# scale to ~[0,1] for a ~15-year record (5500 daily-ish points)
    TMAX_YRS = np.float64(5500.0 / 365.25)                       # ~15.06
    t0_frac = (time_i[0] % 10000) / 10000.0                   # start-year fraction [0,1)
    t_norm = (t_rel_yrs + np.float32(t0_frac)) / np.float32(TMAX_YRS)


    features = np.column_stack([
        t_norm,
        t_pos,
        sin1,
        cos1,
        disp,
        sig,
        median_diffs,      
        csum_dt,
    ]).astype(np.float32, copy=False)

    # Labels: 1 at first epoch affected by each step, else 0 (integer-time exact)
    labels = np.zeros((len(time_i), 1), dtype=np.float32)
    if station_name in step_epochs:
        step_list64 = np.asarray(step_epochs[station_name], dtype=np.float64)
        step_list64 = np.unique(step_list64)
        step_i = np.rint(step_list64 * 10000.0).astype(np.int64)

        for s_i in step_i:
            mask = time_i >= s_i
            if np.any(mask):
                first_index = int(np.argmax(mask))
                labels[first_index, 0] = 1.0
            else:
                print(f"Step {s_i/10000.0:.4f} not found in time array! station {station_name}")

        # Optional consistency check
        step_mask = labels[:, 0].astype(bool)
        picked_i = time_i[step_mask]
        if step_i.size and (picked_i.size != step_i.size or (not np.array_equal(step_i, picked_i))):
            print("\nThe indices are NOT equivalent! (v155 int-time check)\n")
            print(f"Step mask sum: {step_mask.sum()} (should be >0 if steps exist)")
            print(f"Time at step mask: {(picked_i / 10000.0)} station {station_name}")
            print(f"Expected step times: {(step_i / 10000.0)} station {station_name}")
            print(station_name)

    time_out = (time_i / 10000.0).astype(np.float64, copy=False)
    return features, labels, station_name, time_out
def read_step_epochs(step_epochs_file):
    """Backward-compatible alias.

    Older code expects `read_step_epochs(...)`. The canonical implementation in this
    composite script is `read_step_epochs_file(...)`.
    """
    return read_step_epochs_file(step_epochs_file)

class GPSTimeSeriesDataset(Dataset):
    def __init__(
        self,
        data_dir,
        station_files,
        step_epochs=None,
        scaler=False,
        save_scaler_path=None,
        load_scaler_path=None,
        apply_scaling=1,
        make_scaler=False,
        step_epochs_file=None,
        max_length=None,
        figure_on=0,
        max_workers=8,
        progress_log=None,
        progress_every=2000,
        seed=0,
        **kwargs,
    ):
        self.data_dir = data_dir
        self.station_files = station_files

        if step_epochs is None and step_epochs_file is not None:
            step_epochs = read_step_epochs(step_epochs_file)
        self.step_epochs = step_epochs if step_epochs is not None else {}

        self.apply_scaling = bool(apply_scaling)
        self.make_scaler = bool(make_scaler)
        self.max_length = max_length
        self.figure_on = figure_on

        self.progress_log = progress_log
        self.progress_every = int(progress_every) if progress_every is not None else 2000
        self.seed = int(seed)

        self.scaler_flag = scaler
        self.save_scaler_path = save_scaler_path
        self.load_scaler_path = load_scaler_path
        self.max_workers = int(max_workers)

        self.data, self.timeall, self.scaler, self.statall = self.load_data()

    def load_data(self):
        """Load all station files (parallel), fit/load scaler, and return per-station sequences.

        Returns
        -------
        data : list of (features, labels) where each element is a variable-length sequence
        timeall : list of time arrays (unscaled, as loaded)
        scaler : RobustScaler (or None if apply_scaling==0)
        statall : list of station names
        """
        from time import time as _time

        t0 = _time()
        n_total = len(self.station_files)

        results = []

        # Parallel station loading
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            it = executor.map(
                load_station_data,
                self.station_files,
                [self.data_dir] * n_total,
                [None] * n_total,                 # keep signature stable
                [self.step_epochs] * n_total,
                chunksize=50,
            )

            # Optional progress logging
            if self.progress_log is not None:
                try:
                    with open(self.progress_log, "w") as f:
                        f.write("n_done,n_total,elapsed_sec\n")
                except Exception:
                    pass

            for ii, out in enumerate(it, start=1):
                if out is not None:
                    results.append(out)

                if (self.progress_log is not None) and (ii % self.progress_every == 0 or ii == n_total):
                    try:
                        with open(self.progress_log, "a") as f:
                            f.write(f"{ii},{n_total},{_time()-t0:.2f}\n")
                    except Exception:
                        pass

        if len(results) == 0:
            raise RuntimeError("No station files were successfully loaded. Check paths and file formats.")

        features_list, labels_list, statall, timeall = zip(*results)
        features_list = list(features_list)
        labels_list = list(labels_list)
        statall = list(statall)
        timeall = list(timeall)

                
        # Fit / load scaler (global, across all samples)
        #
        # IMPORTANT:
        # - We intentionally **do not** scale column 0 (t_norm) here, because it is already
        #   normalized by construction (roughly 0–1 over the model span). This avoids the
        #   signed-log + robust-scaling pipeline warping time in unintuitive ways.
        # - All other feature columns are scaled with signed-log then RobustScaler.
        #
        # Scaler files are saved as a *bundle* that includes which columns were scaled, so
        # applying the scaler to an independent dataset remains consistent and cannot
        # silently refit on the wrong feature stack.

        n_feat = int(features_list[0].shape[1])
        scaled_cols = list(range(4, n_feat))  # scale everything except t_norm in col 0
        n_scaled = len(scaled_cols)

        scaler_obj = None

        # ---- load existing scaler bundle (apply mode) ----
        if (not self.make_scaler) and (self.load_scaler_path is not None) and os.path.isfile(self.load_scaler_path):
            bundle = load_scaler_bundle(self.load_scaler_path)
            scaler_obj = bundle.get("scaler", None)
            saved_cols = bundle.get("scaled_cols", None)

            if scaler_obj is None:
                raise RuntimeError(f"Scaler file '{self.load_scaler_path}' did not contain a valid scaler object.")

            # Validate column mapping
            if saved_cols is None:
                # back-compat fallback: assume saved scaler was fit to exactly the columns we are scaling now
                saved_cols = scaled_cols

            if len(saved_cols) != n_scaled:
                raise RuntimeError(
                    f"Scaler mismatch: saved scaler expects {len(saved_cols)} scaled features, "
                    f"but current feature stack requires {n_scaled} scaled features. "
                    "This usually means the feature vector changed. Rebuild the scaler with --make_scaler 1."
                )

            if max(saved_cols) >= n_feat:
                raise RuntimeError(
                    f"Scaler mismatch: saved scaler references column {max(saved_cols)} but current data has only {n_feat} features."
                )

            # Use saved_cols to transform (keeps exact training-time mapping)
            scaled_cols = list(map(int, saved_cols))

        # ---- fit scaler (build mode) ----
        if scaler_obj is None:
            all_features_sub = np.vstack([f[:, scaled_cols] for f in features_list])
            scaler_obj = RobustScaler()
            scaler_obj.fit(log_scale(all_features_sub))
            del all_features_sub

            if self.save_scaler_path is not None:
                save_scaler_bundle(self.save_scaler_path, scaler_obj, scaled_cols, n_feat)

        # -------------------------------
        # Apply scaling (in-place) and return
        # -------------------------------
        if self.apply_scaling and (scaler_obj is not None):
            for i, feats in enumerate(features_list):
                feats = feats.astype(np.float32, copy=False)
                sub = feats[:, scaled_cols]
                sub = log_scale(sub)
                feats[:, scaled_cols] = scaler_obj.transform(sub).astype(np.float32, copy=False)
                features_list[i] = feats

        data = list(zip(features_list, labels_list))
        return data, timeall, scaler_obj, statall

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return one sample.

        Returns:
          feats  : (T, F) float32 tensor
          labels : (T, 1) float32 tensor
          station_name : str
          time   : (T,) float32 numpy array (kept as numpy for convenience)
        """
        feats_np, labels_np = self.data[idx]
        feats = torch.tensor(feats_np, dtype=torch.float32)
        labels = torch.tensor(labels_np, dtype=torch.float32)

        station_name = self.statall[idx]
        time = self.timeall[idx]
        return feats, labels, station_name, time



def collate_fn(batch, pad_start=0, pad_end=0, max_length=None):
    """Pad (and optionally truncate) sequences in batch to equal length.

    max_length:
        If provided, each sequence is truncated to at most max_length samples
        BEFORE padding. Truncation keeps the *most recent* samples (tail) so that
        events late in a series are preserved.
    """
    # Sort by sequence length (desc) for pack_padded_sequence
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)

    # Optional truncation
    if max_length is not None and int(max_length) > 0:
        ml = int(max_length)
        new_batch = []
        for features, labels, station_name, time in batch:
            if features.shape[0] > ml:
                features = features[-ml:, :]
                labels   = labels[-ml:, :]
                time     = time[-ml:]
            new_batch.append((features, labels, station_name, time))
        batch = new_batch

    max_original_len = max(x[0].shape[0] for x in batch)
    max_len = max_original_len + int(pad_start) + int(pad_end)

    padded_features = []
    padded_labels = []
    lengths = []
    masks = []
    station_names = []

    for features, labels, station_name, time in batch:
        # Ensure torch tensors
        if not torch.is_tensor(features):
            features_t = torch.as_tensor(features, dtype=torch.float32)
        else:
            features_t = features.to(dtype=torch.float32)

        if not torch.is_tensor(labels):
            labels_t = torch.as_tensor(labels, dtype=torch.float32)
        else:
            labels_t = labels.to(dtype=torch.float32)

        original_len = features_t.shape[0]
        F = features_t.shape[1]
        L = labels_t.shape[1]

        # Padding before/after (torch-native to avoid numpy dtype issues)
        if int(pad_start) > 0:
            pad_before = torch.zeros((int(pad_start), F), dtype=features_t.dtype)
            pad_before_y = torch.zeros((int(pad_start), L), dtype=labels_t.dtype)
        else:
            pad_before = None
            pad_before_y = None

        if int(pad_end) > 0:
            pad_after = torch.zeros((int(pad_end), F), dtype=features_t.dtype)
            pad_after_y = torch.zeros((int(pad_end), L), dtype=labels_t.dtype)
        else:
            pad_after = None
            pad_after_y = None

        parts_x = [p for p in [pad_before, features_t, pad_after] if p is not None]
        parts_y = [p for p in [pad_before_y, labels_t, pad_after_y] if p is not None]
        padded_seq = torch.cat(parts_x, dim=0)
        padded_seq_labels = torch.cat(parts_y, dim=0)

        # Right-pad to max_len
        pad_total = max_len - padded_seq.shape[0]
        if pad_total > 0:
            padded_seq = torch.cat([padded_seq, torch.zeros((pad_total, F), dtype=features_t.dtype)], dim=0)
            padded_seq_labels = torch.cat([padded_seq_labels, torch.zeros((pad_total, L), dtype=labels_t.dtype)], dim=0)

        padded_features.append(padded_seq)
        padded_labels.append(padded_seq_labels)

        lengths.append(original_len + int(pad_start) + int(pad_end))

        mask = torch.zeros(max_len, dtype=torch.float32)
        mask[int(pad_start):int(pad_start) + original_len] = 1.0
        masks.append(mask)

        station_names.append(station_name)

    return (
        torch.stack(padded_features, dim=0),
        torch.stack(padded_labels, dim=0),
        torch.tensor(lengths, dtype=torch.int64),
        torch.stack(masks, dim=0),
        station_names,
    )



def mode_build_dataset(args):
    """Build dataset and save it (torch.save) + metadata + scaler."""

    # Back-compat arg aliases
    dataset_out = getattr(args, "dataset_out", None) or getattr(args, "dataset_pt", None)
    metadata_out = getattr(args, "metadata_out", None) or getattr(args, "metadata_npy", None)
    scaler_out = getattr(args, "scaler_out", None) or getattr(args, "scaler_path", None)

    if dataset_out is None:
        raise ValueError("Missing dataset output path. Use --dataset_out (recommended) or --dataset_pt.")

    if metadata_out is None:
        raise ValueError("Missing metadata output path. Use --metadata_out (recommended) or --metadata_npy.")

    # Station list
    station_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith(".data")])
    if len(station_files) == 0:
        raise RuntimeError(f"No .data station files found in data_dir={args.data_dir}")

    # Progress log (optional)
    progress_log = getattr(args, "progress_log", None) or getattr(args, "log_path", None)

    # Workers
    n_workers = int(getattr(args, "n_workers", None) or getattr(args, "max_workers", None) or 8)

    dataset = GPSTimeSeriesDataset(
        data_dir=args.data_dir,
        station_files=station_files,
        step_epochs_file=args.step_epochs_file,
        apply_scaling=int(args.apply_scaling),
        make_scaler=int(args.make_scaler),
        save_scaler_path=scaler_out if int(args.make_scaler) == 1 else None,
        load_scaler_path=args.scaler_path if (int(args.make_scaler) == 0 and args.scaler_path is not None) else None,
        max_workers=n_workers,
        progress_log=progress_log,
        progress_every=int(args.progress_every),
        seed=int(args.seed),
    )

    # Save dataset + metadata
    torch.save(dataset, dataset_out)

    # Metadata: station names (and optionally lengths)
    meta = {
        "station_names": dataset.statall,
        "n_stations": len(dataset.statall),
    }
    np.save(metadata_out, meta, allow_pickle=True)

    print(f"Saved dataset: {dataset_out}")
    print(f"Saved metadata: {metadata_out}")
    if scaler_out is not None and int(args.make_scaler) == 1:
        print(f"Saved scaler: {scaler_out}")



def mode_train_folds(args):
    """Train K folds (same core training loop as before), optionally with DDP."""
    state = ddp_init_if_needed(args)
    max_norm = 1.0

    if args.dataset_pt is None:
        raise ValueError("--dataset_pt is required for train_folds")

    dataset = torch.load(args.dataset_pt, map_location="cpu")

    # Optionally enforce a max sequence length at collate time (no change if <=0/None)
    max_length = None
    if getattr(args, "max_length", None) is not None and int(args.max_length) > 0:
        max_length = int(args.max_length)

    # Infer input feature size
    x0, _, _, _ = dataset[0]
    input_size = x0.shape[1]

    # Training hyperparams
    linear_size = int(args.linear_size)
    hidden_size = int(args.hidden_size)
    bidirectional = bool(int(args.bidirectional))
    lr = float(args.learning_rate)
    weight_decay = float(args.weight_decay)
    n_epochs = int(args.num_epochs)

    train_bs = int(args.train_batch_size)
    val_bs   = int(args.val_batch_size)
    num_workers = int(args.train_workers)

    # Output locations
    out_dir = args.train_out_dir
    os.makedirs(out_dir, exist_ok=True)
    version = str(args.version_tag)

    # Path for per-epoch loss CSV (written by rank0 only)
    loss_csv_path = os.path.join(out_dir, f"loss_{version}.csv") if state.is_main else None
    if loss_csv_path is not None and (not os.path.exists(loss_csv_path)):
        with open(loss_csv_path, "w") as fcsv:
            fcsv.write("epoch,train_loss,val_loss,lr\n")


    # --- Fold splitting (GROUPED by station prefix to avoid leakage across siblings) ---
    all_indices = np.arange(len(dataset))

    # Get station names aligned with dataset items
    station_names = []
    if hasattr(dataset, "statall") and dataset.statall is not None and len(dataset.statall) == len(dataset):
        station_names = [str(s) for s in dataset.statall]
    else:
        # Fallback: dataset[i] returns (x, y, length, station_name)
        for i in range(len(dataset)):
            _x, _y, _l, st = dataset[i]
            station_names.append(str(st))

    # Group id = first 4 chars (your station family key)
    groups = np.array([s[:4] for s in station_names], dtype=object)

    unique_groups = np.unique(groups)
    k_folds = int(args.k_folds)
    if k_folds > len(unique_groups):
        raise ValueError(f"k_folds={k_folds} > number of unique station groups={len(unique_groups)}")

    rng = np.random.RandomState(int(args.fold_seed))
    rng.shuffle(unique_groups)
    fold_group_bins = np.array_split(unique_groups, k_folds)

    # Collate partial
    from functools import partial
    collate = partial(collate_fn, max_length=max_length, pad_start=int(args.pad_start), pad_end=int(args.pad_end))


    for fold, val_groups in enumerate(fold_group_bins, start=1):

        val_mask = np.isin(groups, val_groups)
        train_idx = all_indices[~val_mask]
        val_idx   = all_indices[val_mask]

        if state.is_main:
            print(f"[train_folds] Entering fold {fold}", flush=True)

        train_subset = Subset(dataset, train_idx)
        val_subset   = Subset(dataset, val_idx)

        if int(getattr(args, "bucket_by_length", 0)) == 1:
            lengths_train = _subset_lengths(train_subset)
            train_batch_sampler = BucketBatchSampler(
                lengths_train,
                batch_size=int(args.train_batch_size),
                num_buckets=int(getattr(args, "bucket_num_buckets", 50)),
                shuffle=bool(int(getattr(args, "bucket_shuffle", 1))),
                shuffle_batches=bool(int(getattr(args, "bucket_shuffle_batches", 1))),
                drop_last=bool(int(getattr(args, "bucket_drop_last", 0) or state.enabled)),
                seed=int(getattr(args, "seed", 0)),
                world_size=(state.world_size if state.enabled else 1),
                rank=(state.rank if state.enabled else 0),
            )
            train_sampler = None
            train_loader = DataLoader(
                train_subset,
                batch_sampler=train_batch_sampler,
                **dataloader_common_kwargs(args.train_workers, bool(args.pin_memory), args.prefetch_factor, args.persistent_workers),
                collate_fn=collate,
            )
        else:
            if state.enabled:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_subset, num_replicas=state.world_size, rank=state.rank, shuffle=True, drop_last=False
                )
            else:
                train_sampler = None

            train_loader = DataLoader(
                train_subset,
                batch_size=int(args.train_batch_size),
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                **dataloader_common_kwargs(args.train_workers, bool(args.pin_memory), args.prefetch_factor, args.persistent_workers),
                collate_fn=collate,
                drop_last=False,
            )


        val_sampler = None
        if state.enabled:
            val_sampler = DistributedSampler(
                val_subset,
                num_replicas=state.world_size,
                rank=state.rank,
                shuffle=False
            )

        val_workers = _adjust_num_workers(args.val_workers, state)


        val_loader = DataLoader(
            val_subset,
            batch_size=val_bs,
            shuffle=False,
            sampler=val_sampler,
            **dataloader_common_kwargs(val_workers, bool(args.pin_memory), args.prefetch_factor, args.persistent_workers),
            collate_fn=collate,
            drop_last=False,
        )

        # Model definition (same architecture, now parameterized)
        class LSTMEncoderDecoder(nn.Module):
            def __init__(self, input_size, linear_size, hidden_size, num_layers, output_size, dropout=0.2, bidirectional=True):
                super().__init__()
                self.bidirectional = bool(bidirectional)
                self.num_directions = 2 if self.bidirectional else 1
                self.linear_in = nn.Linear(input_size, linear_size)

                # For num_layers==1, PyTorch's LSTM dropout is ignored; use explicit dropout modules so MC-dropout works.
                self._explicit_dropout = (int(num_layers) == 1) and (float(dropout) > 0.0)
                self._drop = nn.Dropout(float(dropout)) if self._explicit_dropout else None
                lstm_dropout = 0.0 if int(num_layers) == 1 else float(dropout)

                self.encoder = nn.LSTM(
                    input_size=linear_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=lstm_dropout,
                    bidirectional=self.bidirectional,
                )

                self.decoder = nn.LSTM(
                    input_size=hidden_size * self.num_directions,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=lstm_dropout,
                    bidirectional=self.bidirectional,
                )

                self.output_layer = nn.Linear(hidden_size * self.num_directions, output_size)

                init.kaiming_uniform_(self.linear_in.weight, nonlinearity='relu')
                init.zeros_(self.linear_in.bias)

                for name, param in self.encoder.named_parameters():
                    if 'weight' in name:
                        init.orthogonal_(param)
                    elif 'bias' in name:
                        init.zeros_(param)

                for name, param in self.decoder.named_parameters():
                    if 'weight' in name:
                        init.orthogonal_(param)
                    elif 'bias' in name:
                        init.zeros_(param)

                init.kaiming_uniform_(self.output_layer.weight, nonlinearity='linear')
                init.zeros_(self.output_layer.bias)

            def forward(self, x, lengths):
                x = torch.relu(self.linear_in(x))
                if self._drop is not None:
                    x = self._drop(x)

                packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
                packed_enc_out, (h, c) = self.encoder(packed_x)
                enc_out, _ = nn.utils.rnn.pad_packed_sequence(packed_enc_out, batch_first=True)
                if self._drop is not None:
                    enc_out = self._drop(enc_out)

                packed_dec_in = nn.utils.rnn.pack_padded_sequence(enc_out, lengths.cpu(), batch_first=True, enforce_sorted=False)
                packed_dec_out, _ = self.decoder(packed_dec_in, (h, c))
                dec_out, _ = nn.utils.rnn.pad_packed_sequence(packed_dec_out, batch_first=True)
                if self._drop is not None:
                    dec_out = self._drop(dec_out)

                return self.output_layer(dec_out)

        model = LSTMEncoderDecoder(
            input_size=input_size,
            linear_size=linear_size,
            hidden_size=hidden_size,
            num_layers=int(args.num_layers),
            output_size=1,
            dropout=float(args.dropout),
            bidirectional=bidirectional,
        ).to(state.device)

        # --- Optional torch.compile (per-rank) ---
        if int(args.compile) == 1 and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode=args.compile_mode)
            except Exception as e:
                if state.is_main:
                    print(f"torch.compile disabled due to error: {e}")

        # --- AMP settings ---
        use_amp = (int(args.amp) == 1) and (state.device.type == "cuda")
        amp_dtype = torch.bfloat16 if str(args.amp_dtype).lower() == "bf16" else torch.float16
        scaler = GradScaler(enabled=use_amp and amp_dtype == torch.float16)

        if state.enabled:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[state.local_rank], output_device=state.local_rank
            )

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- Optional LR scheduler ---
        scheduler = None
        if int(getattr(args, "use_lr_scheduler", 0) or 0) == 1 and str(getattr(args, "lr_scheduler", "plateau")).lower() == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(getattr(args, "lr_factor", 0.5)),                 # default: 0.5
                patience=int(getattr(args, "lr_patience", 20)),                # default: 20
                threshold=float(getattr(args, "lr_threshold", 0.01)),           # default: 1% rel
                threshold_mode=str(getattr(args, "lr_threshold_mode", "rel")),  # default: rel
                cooldown=int(getattr(args, "lr_cooldown", 10)),                 # default: 10
                min_lr=float(getattr(args, "lr_min_lr", 1e-5)),                 # default: 1e-5
                eps=float(getattr(args, "lr_eps", 1e-8)),
            )


        # loss (same as before; masked BCE over valid timesteps)
            pos_weight = torch.tensor([float(args.pos_weight)], device=state.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

        # Logs
            best_val = float("inf")
            loss_rows = []

        # Early stopping trackers (optional)
            es_enabled = int(getattr(args, "early_stop", 0) or 0) == 1
            es_metric = str(getattr(args, "early_stop_metric", "train")).lower()
            es_patience = int(getattr(args, "early_stop_patience", 50))
            es_min_delta = float(getattr(args, "early_stop_min_delta", 1e-6))
            es_warmup = int(getattr(args, "early_stop_warmup", 10))
            es_loss_thresh = float(getattr(args, "early_stop_loss_threshold", 5e-5))
            best_metric = float("inf")
            bad_epochs = 0

            for epoch in range(1, n_epochs + 1):

                if state.is_ddp and train_sampler is not None:
                    train_sampler.set_epoch(epoch)

                if state.enabled:
                    _dataloader_set_epoch(train_loader, epoch)

                model.train()
                running_loss = 0.0
                n_batches = 0

                for batch in tqdm(train_loader, desc=f"Fold {fold} | Epoch {epoch}/{n_epochs} (train)", disable=not state.is_main, dynamic_ncols=True, leave=False, mininterval=1.0):
                    inputs, step_labels, lengths, mask, _ = batch
                    inputs = inputs.to(state.device, non_blocking=True)
                    step_labels = step_labels.to(state.device, non_blocking=True)
                    mask = mask.to(state.device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)


                    with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):



                            step_outputs = model(inputs, lengths)
                            step_outputs = step_outputs.clamp(min=-20.0, max=20.0)



                            loss_raw = criterion(step_outputs, step_labels)



                            mask_exp = mask.unsqueeze(-1).expand_as(loss_raw)



                            loss = (loss_raw * mask_exp).sum() / mask_exp.sum().clamp_min(1.0)



                    if scaler.is_enabled():


                        scaler.scale(loss).backward()
                        
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


                        scaler.step(optimizer)


                        scaler.update()


                    else:


                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                        optimizer.step()

                    running_loss += loss.detach().item()
                    n_batches += 1

                train_loss = running_loss / max(n_batches, 1)

            # Validation
                model.eval()
                v_running = 0.0
                v_batches = 0
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f"Fold {fold} | Epoch {epoch}/{n_epochs} (val)", disable=not state.is_main, dynamic_ncols=True, leave=False, mininterval=1.0):
                        inputs, step_labels, lengths, mask, _ = batch
                        inputs = inputs.to(state.device, non_blocking=True)
                        step_labels = step_labels.to(state.device, non_blocking=True)
                        mask = mask.to(state.device, non_blocking=True)

                        step_outputs = model(inputs, lengths)
                        loss_raw = criterion(step_outputs, step_labels)
                        mask_exp = mask.unsqueeze(-1).expand_as(loss_raw)
                        vloss = (loss_raw * mask_exp).sum() / mask_exp.sum().clamp_min(1.0)

                        v_running += vloss.detach().item()
                        v_batches += 1

                val_loss = v_running / max(v_batches, 1)

            # DDP reduce losses
                if state.enabled:
                    train_loss = ddp_mean(train_loss, state.device)
                    val_loss   = ddp_mean(val_loss, state.device)

            # --- LR scheduler step (after DDP reductions) ---
                metric_for_sched = train_loss
                if scheduler is not None and str(getattr(args, "lr_monitor", "train")).lower() == "val":
                    if (val_loader is not None) and (not np.isnan(val_loss)):
                        metric_for_sched = val_loss
                if scheduler is not None:
                    scheduler.step(metric_for_sched)


                lr_now = float(optimizer.param_groups[0].get("lr", float("nan")))
                if state.is_main:
                    loss_rows.append([epoch, train_loss, val_loss, lr_now])
                    print(f"[epoch {epoch}/{n_epochs}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={lr_now:.3e}", flush=True)

            # checkpoint best
                if val_loss < best_val:
                    best_val = val_loss
                    if state.is_main:
                        best_path = os.path.join(out_dir, f"{version}_fold{fold:02d}_best.pt")
                        torch.save(
                            {"model": model.module.state_dict() if state.enabled else model.state_dict(),
                             "args": vars(args)},
                            best_path
                        )

            # save full checkpoint EVERY epoch (fold + epoch)
                if state.is_main:
                    epoch_path = os.path.join(out_dir, f"{version}_fold{fold:02d}_epoch{epoch:04d}.pt")
                    torch.save(
                        {"model": model.module.state_dict() if state.enabled else model.state_dict(),
                         "args": vars(args)},
                        epoch_path
                    )

                # save loss log every epoch
                    loss_csv = os.path.join(out_dir, f"{version}_fold{fold:02d}_loss.csv")
                    np.savetxt(
                        loss_csv,
                        np.array(loss_rows),
                        delimiter=",",
                        header="epoch,train_loss,val_loss,lr",
                        comments=""
                    )

            # Optional stop: terminate once lr hits lr_min_lr
                if getattr(args, "stop_when_min_lr", False):
                    min_lr = float(getattr(args, "lr_min_lr", 0.0) or 0.0)
                    if min_lr > 0.0 and (not np.isnan(lr_now)) and (lr_now <= (min_lr + 1e-12)):
                        if state.is_main:
                            print(f"[STOP] lr reached lr_min_lr={min_lr:.3e} (lr={lr_now:.3e})", flush=True)
                        break


            if state.is_main:
            # Save last checkpoint too
                last_path = os.path.join(out_dir, f"{version}_fold{fold:02d}_last.pt")
                torch.save({"model": model.module.state_dict() if state.enabled else model.state_dict(),
                            "args": vars(args)}, last_path)

    _ddp_barrier_and_cleanup(state)



def mode_train_final(args):
    """Train a single final model on the full dataset (optionally hold out a small val split)."""
    state = ddp_init_if_needed(args)
    max_norm = 1.0


    if args.dataset_pt is None:
        raise ValueError("--dataset_pt is required for train_final")

    dataset = torch.load(args.dataset_pt, map_location="cpu")

    max_length = None
    if getattr(args, "max_length", None) is not None and int(args.max_length) > 0:
        max_length = int(args.max_length)

    # Infer input feature size
    x0, _, _, _ = dataset[0]
    input_size = x0.shape[1]

# --- Early-stop enable flag (default OFF) ---
    es_enabled = bool(int(getattr(args, "early_stop", 0) or 0))



    # Hyperparams
    linear_size = int(args.linear_size)
    hidden_size = int(args.hidden_size)
    bidirectional = bool(int(args.bidirectional))
    lr = float(args.learning_rate)
    weight_decay = float(args.weight_decay)
    n_epochs = int(args.num_epochs)

    train_bs = int(args.train_batch_size)
    val_bs   = int(args.val_batch_size)
    num_workers = int(args.train_workers)

    out_dir = args.train_out_dir
    os.makedirs(out_dir, exist_ok=True)
    version = str(args.version_tag)

    # Loss CSV (rank0 only)
    loss_csv_path = None
    if state.is_main:
        loss_csv_path = os.path.join(out_dir, f"loss_{version}_final.csv")
        with open(loss_csv_path, "w") as f:
            f.write("epoch,train_loss,val_loss,lr\n")


    # Validation handling:
    # - If --val_dataset_pt is provided (non-empty), it is used as the validation dataset (recommended for independent synthetic sets).
    # - Otherwise, val_fraction can hold out a random subset from the training dataset.
    val_dataset_pt = str(getattr(args, "val_dataset_pt", "") or "").strip()
    if val_dataset_pt != "":
        val_dataset = torch.load(val_dataset_pt, map_location="cpu")
        train_subset = dataset
        val_subset = val_dataset
        if state.is_main:
            print(f"[train_final] Using external validation dataset: {val_dataset_pt} (n={len(val_subset)})")
    else:
        # Optional val split (0.0 keeps legacy "train on all" behavior)
        val_fraction = float(getattr(args, "val_fraction", 0.0))
        all_indices = np.arange(len(dataset))
        rng = np.random.RandomState(int(args.seed))
        rng.shuffle(all_indices)

        if val_fraction > 0:
            n_val = max(1, int(round(val_fraction * len(dataset))))
            val_idx = all_indices[:n_val]
            train_idx = all_indices[n_val:]
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
        else:
            train_subset = dataset
            val_subset = None

    from functools import partial
    collate = partial(collate_fn, max_length=max_length, pad_start=int(args.pad_start), pad_end=int(args.pad_end))

    if int(getattr(args, "bucket_by_length", 0)) == 1:
        lengths_train = _subset_lengths(train_subset)
        train_batch_sampler = BucketBatchSampler(
            lengths_train,
            batch_size=int(args.train_batch_size),
            num_buckets=int(getattr(args, "bucket_num_buckets", 50)),
            shuffle=bool(int(getattr(args, "bucket_shuffle", 1))),
            shuffle_batches=bool(int(getattr(args, "bucket_shuffle_batches", 1))),
            drop_last=bool(int(getattr(args, "bucket_drop_last", 0) or state.enabled)),
            seed=int(getattr(args, "seed", 0)),
            world_size=(state.world_size if state.enabled else 1),
            rank=(state.rank if state.enabled else 0),
        )
        train_sampler = None
        train_loader = DataLoader(
            train_subset,
            batch_sampler=train_batch_sampler,
            **dataloader_common_kwargs(args.train_workers, bool(args.pin_memory), args.prefetch_factor, args.persistent_workers),
            collate_fn=collate,
        )
    else:
        if state.enabled:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_subset, num_replicas=state.world_size, rank=state.rank, shuffle=True, drop_last=False
            )
        else:
            train_sampler = None

        train_loader = DataLoader(
            train_subset,
            batch_size=int(args.train_batch_size),
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            **dataloader_common_kwargs(args.train_workers, bool(args.pin_memory), args.prefetch_factor, args.persistent_workers),
            collate_fn=collate,
            drop_last=False,
        )



    if val_subset is not None:
        if state.enabled:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_subset, num_replicas=state.world_size, rank=state.rank, shuffle=False, drop_last=False
            )
        else:
            val_sampler = None
        # Validation workers
        val_workers = int(getattr(args, 'val_workers', 0) or 0)
        val_workers = val_workers if val_workers > 0 else int(args.train_workers)

        val_loader = DataLoader(
            val_subset,
            batch_size=val_bs,
            shuffle=False,
            sampler=val_sampler,
            **dataloader_common_kwargs(val_workers, bool(args.pin_memory), args.prefetch_factor, args.persistent_workers),
            collate_fn=collate,
            drop_last=False,
        )
    else:
        val_loader = None

    class LSTMEncoderDecoder(nn.Module):
        def __init__(self, input_size, linear_size, hidden_size, num_layers, output_size, dropout=0.2, bidirectional=True):
            super().__init__()
            self.bidirectional = bool(bidirectional)
            self.num_directions = 2 if self.bidirectional else 1
            self.linear_in = nn.Linear(input_size, linear_size)

            # For num_layers==1, PyTorch's LSTM dropout is ignored; use explicit dropout modules so MC-dropout works.
            self._explicit_dropout = (int(num_layers) == 1) and (float(dropout) > 0.0)
            self._drop = nn.Dropout(float(dropout)) if self._explicit_dropout else None
            lstm_dropout = 0.0 if int(num_layers) == 1 else float(dropout)

            self.encoder = nn.LSTM(
                input_size=linear_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=lstm_dropout,
                bidirectional=self.bidirectional,
            )

            self.decoder = nn.LSTM(
                input_size=hidden_size * self.num_directions,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=lstm_dropout,
                bidirectional=self.bidirectional,
            )

            self.output_layer = nn.Linear(hidden_size * self.num_directions, output_size)

            init.kaiming_uniform_(self.linear_in.weight, nonlinearity='relu')
            init.zeros_(self.linear_in.bias)

            for name, param in self.encoder.named_parameters():
                if 'weight' in name:
                    init.orthogonal_(param)
                elif 'bias' in name:
                    init.zeros_(param)

            for name, param in self.decoder.named_parameters():
                if 'weight' in name:
                    init.orthogonal_(param)
                elif 'bias' in name:
                    init.zeros_(param)

            init.kaiming_uniform_(self.output_layer.weight, nonlinearity='linear')
            init.zeros_(self.output_layer.bias)

        def forward(self, x, lengths):
            x = torch.relu(self.linear_in(x))
            if self._drop is not None:
                x = self._drop(x)

            packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_enc_out, (h, c) = self.encoder(packed_x)
            enc_out, _ = nn.utils.rnn.pad_packed_sequence(packed_enc_out, batch_first=True)
            if self._drop is not None:
                enc_out = self._drop(enc_out)

            packed_dec_in = nn.utils.rnn.pack_padded_sequence(enc_out, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_dec_out, _ = self.decoder(packed_dec_in, (h, c))
            dec_out, _ = nn.utils.rnn.pad_packed_sequence(packed_dec_out, batch_first=True)
            if self._drop is not None:
                dec_out = self._drop(dec_out)

            return self.output_layer(dec_out)

    model = LSTMEncoderDecoder(
        input_size=input_size,
        linear_size=linear_size,
        hidden_size=hidden_size,
        num_layers=int(args.num_layers),
        output_size=1,
        dropout=float(args.dropout),
        bidirectional=bidirectional,
    ).to(state.device)

    # --- Optional torch.compile (per-rank) ---
    if int(args.compile) == 1 and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=args.compile_mode)
        except Exception as e:
            if state.is_main:
                print(f"torch.compile disabled due to error: {e}")

    # --- AMP settings ---
    use_amp = (int(args.amp) == 1) and (state.device.type == "cuda")
    amp_dtype = torch.bfloat16 if str(args.amp_dtype).lower() == "bf16" else torch.float16
    scaler = GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    if state.enabled:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[state.local_rank], output_device=state.local_rank
        )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- Optional LR scheduler (v99-style) ---
    scheduler = None
    if int(getattr(args, "use_lr_scheduler", 0) or 0) == 1 and str(getattr(args, "lr_scheduler", "plateau")).lower() == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(getattr(args, "lr_factor", 0.5)),                 # default: 0.5
            patience=int(getattr(args, "lr_patience", 20)),                # default: 20
            threshold=float(getattr(args, "lr_threshold", 0.01)),           # default: 1% rel
            threshold_mode=str(getattr(args, "lr_threshold_mode", "rel")),  # default: rel
            cooldown=int(getattr(args, "lr_cooldown", 10)),                 # default: 10
            min_lr=float(getattr(args, "lr_min_lr", 1e-5)),                 # default: 1e-5
            eps=float(getattr(args, "lr_eps", 1e-8)),
        )


    pos_weight = torch.tensor([float(args.pos_weight)], device=state.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    best_val = float("inf")
    loss_rows = []

    for epoch in range(1, n_epochs + 1):
        if state.is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if state.enabled:
            _dataloader_set_epoch(train_loader, epoch)

        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Final | Epoch {epoch}/{n_epochs} (train)", disable=not state.is_main, dynamic_ncols=True, leave=False, mininterval=1.0):
            inputs, step_labels, lengths, mask, _ = batch
            inputs = inputs.to(state.device, non_blocking=True)
            step_labels = step_labels.to(state.device, non_blocking=True)
            mask = mask.to(state.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)


            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):



                    step_outputs = model(inputs, lengths)
                    step_outputs = step_outputs.clamp(min=-20.0, max=20.0)



                    loss_raw = criterion(step_outputs, step_labels)



                    mask_exp = mask.unsqueeze(-1).expand_as(loss_raw)



                    loss = (loss_raw * mask_exp).sum() / mask_exp.sum().clamp_min(1.0)



            if scaler.is_enabled():


                scaler.scale(loss).backward()
                
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                
                scaler.step(optimizer)
                scaler.update()


            else:


                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


                optimizer.step()

            running_loss += loss.detach().item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)

        # Validation (optional)
        if val_loader is not None:
            model.eval()
            v_running = 0.0
            v_batches = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Final | Epoch {epoch}/{n_epochs} (val)", disable=not state.is_main, dynamic_ncols=True, leave=False, mininterval=1.0):
                    inputs, step_labels, lengths, mask, _ = batch
                    inputs = inputs.to(state.device, non_blocking=True)
                    step_labels = step_labels.to(state.device, non_blocking=True)
                    mask = mask.to(state.device, non_blocking=True)

                    step_outputs = model(inputs, lengths)
                    loss_raw = criterion(step_outputs, step_labels)
                    mask_exp = mask.unsqueeze(-1).expand_as(loss_raw)
                    vloss = (loss_raw * mask_exp).sum() / mask_exp.sum().clamp_min(1.0)

                    v_running += vloss.detach().item()
                    v_batches += 1
            val_loss = v_running / max(v_batches, 1)
        else:
            val_loss = float("nan")

        if state.enabled:
            train_loss = ddp_mean(train_loss, state.device)
            if not np.isnan(val_loss):
                val_loss = ddp_mean(val_loss, state.device)

        # --- LR scheduler step (after DDP reductions) ---
        metric_for_sched = train_loss
        if scheduler is not None and str(getattr(args, "lr_monitor", "train")).lower() == "val":
            if (val_loader is not None) and (not np.isnan(val_loss)):
                metric_for_sched = val_loss
        if scheduler is not None:
            scheduler.step(metric_for_sched)

        # --- Early stopping check (rank-0 decides, then broadcast) ---
        stop_now = False
        if es_enabled:
            metric_for_stop = train_loss
            if es_metric == "val" and (val_loader is not None) and (not np.isnan(val_loss)):
                metric_for_stop = val_loss

            # immediate threshold stop (v99-style)
            if es_loss_thresh > 0 and metric_for_stop < es_loss_thresh:
                stop_now = True

            # patience-based stop (relative improvement)
            if epoch > es_warmup:
                # Initialize on first eligible epoch
                if not np.isfinite(best_metric):
                    best_metric = metric_for_stop
                rel_improve = (best_metric - metric_for_stop) / max(abs(best_metric), 1e-12)
                if rel_improve > es_min_delta:
                    best_metric = metric_for_stop
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= es_patience:
                        stop_now = True
            else:
                # during warmup, keep best_metric updated
                if metric_for_stop < best_metric:
                    best_metric = metric_for_stop


        # Optional stop: terminate once lr hits lr_min_lr (useful with ReduceLROnPlateau)
        if getattr(args, "stop_when_min_lr", False):
            lr_now = float(optimizer.param_groups[0].get("lr", float("nan")))
            min_lr = float(getattr(args, "lr_min_lr", 0.0) or 0.0)
            if min_lr > 0.0 and (not np.isnan(lr_now)) and (lr_now <= (min_lr + 1e-12)):
                if state.is_main:
                    print(f"[STOP] lr reached lr_min_lr={min_lr:.3e} (lr={lr_now:.3e})", flush=True)
                stop_now = True
        if state.enabled:
            tstop = torch.tensor([1 if (stop_now and state.is_main) else 0], device=state.device, dtype=torch.int32)
            dist.broadcast(tstop, src=0)
            stop_now = bool(int(tstop.item()))


        if state.is_main:
            lr_now = optimizer.param_groups[0].get("lr", float("nan"))
            loss_rows.append([epoch, train_loss, val_loss, lr_now])
            if state.is_main:
                lr_now = optimizer.param_groups[0].get("lr", float("nan"))
                print(f"[epoch {epoch}/{n_epochs}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={lr_now:.3e}", flush=True)

# Optional: stop once LR hits the configured minimum (ReduceLROnPlateau will otherwise keep running)
        # Save best + rolling checkpoint + log (rank0 only)
        if state.is_main:
            if val_loader is None or (not np.isnan(val_loss) and val_loss < best_val):
                if val_loader is not None and not np.isnan(val_loss):
                    best_val = float(val_loss)
                best_path = os.path.join(out_dir, f"{version}_final_best.pt")
                torch.save({
                    "model": (model.module.state_dict() if state.enabled else model.state_dict()),
                    "args": vars(args),
                    "epoch": epoch,
                    "best_val": best_val,
                }, best_path)

            ckpt_path = os.path.join(out_dir, f"{version}_final_ckpt.pt")
            torch.save({
                "model": (model.module.state_dict() if state.enabled else model.state_dict()),
                "optimizer": optimizer.state_dict(),
                "scheduler": (scheduler.state_dict() if scheduler is not None else None),
                "args": vars(args),
                "epoch": epoch,
                "best_val": best_val,
            }, ckpt_path)
            # Save a full per-epoch snapshot (matches legacy behavior)
            epoch_path = os.path.join(out_dir, f"{version}_final_epoch{epoch:04d}.pt")
            torch.save({
                "model": (model.module.state_dict() if state.enabled else model.state_dict()),
                "optimizer": optimizer.state_dict(),
                "scheduler": (scheduler.state_dict() if scheduler is not None else None),
                "args": vars(args),
                "epoch": epoch,
                "best_val": best_val,
            }, epoch_path)

            if loss_csv_path is not None:
                _append_loss_row_csv(loss_csv_path, loss_rows)
        if stop_now:
            if state.is_main:
                lr_now = optimizer.param_groups[0].get("lr", float("nan"))
                print(f"[early_stop] stopping at epoch {epoch} | metric={metric_for_sched:.6e} | lr={lr_now:.3e}", flush=True)
            break


    if state.is_main:
        last_path = os.path.join(out_dir, f"{version}_final_last.pt")
        torch.save({"model": model.module.state_dict() if state.enabled else model.state_dict(),
                    "args": vars(args)}, last_path)

    _ddp_barrier_and_cleanup(state)



def mode_inference(args):
    """Run model inference over all station files in --data_dir.

    v142 changes (inference only):
      - If a station time series is longer than --max_length, run fixed-length window inference
        (window_len=max_length) with overlap, then merge into a single probability series with
        the original time axis and full length.
      - Window starts are chosen to be balanced (evenly spaced between 0 and T-window_len),
        avoiding a tiny tail-only window.
      - Merging uses overlap-weighted averaging with taper weights to minimize seam artifacts.
      - Dataset build + training behavior is unchanged.
    """
    state = ddp_init_if_needed(args)

    data_dir = getattr(args, "data_dir", None)
    output_dir = getattr(args, "output_dir", None) or getattr(args, "out_dir", None)
    model_path = getattr(args, "model_path", None) or getattr(args, "model", None)
    scaler_path = getattr(args, "scaler_path", None) or getattr(args, "scaler_file", None)

    if not data_dir:
        raise ValueError("--data_dir is required for inference (directory of station .data files).")
    if not output_dir:
        raise ValueError("--output_dir (or --out_dir) is required for inference.")
    if not model_path:
        raise ValueError("--model is required for inference (path to trained model checkpoint).")
    if not scaler_path:
        raise ValueError("--scaler_path (or --scaler_file) is required for inference.")

    os.makedirs(output_dir, exist_ok=True)

    # Load scaler bundle (supports both bundled dicts and raw sklearn scalers)
    bundle = load_scaler_bundle(scaler_path)
    scaler_obj = bundle["scaler"]
    scaled_cols = bundle.get("scaled_cols", None)

    # Discover station files
    station_files = sorted([fn for fn in os.listdir(data_dir) if fn.endswith(".data") or fn.endswith(".txt")])
    if len(station_files) == 0:
        raise FileNotFoundError(f"No station files found in {data_dir} (expected *.data or *.txt).")

    # Under DDP, shard station files by rank so outputs are not duplicated and all windows for a station
    # stay on the same rank (no cross-rank merging).
    if state.enabled:
        station_files = station_files[state.rank::state.world_size]

    # Window length (trained max regime)
    max_length = None
    if getattr(args, "max_length", None) is not None and int(args.max_length) > 0:
        max_length = int(args.max_length)

    # Overlap for long-series windowing (in samples)
    infer_overlap = int(getattr(args, "infer_overlap", 1000) or 1000)
    if max_length is not None:
        infer_overlap = max(0, min(infer_overlap, max_length - 1))

    pad_start = int(getattr(args, "pad_start", 0) or 0)
    pad_end   = int(getattr(args, "pad_end", 0) or 0)

    def _balanced_window_starts(T, L, overlap):
        """Balanced window starts so each window has length L and coverage is even."""
        if T <= L:
            return [0]
        stride = max(1, L - max(0, overlap))
        nwin = int(math.ceil((T - L) / stride)) + 1
        starts = np.linspace(0, T - L, nwin)
        starts = np.round(starts).astype(np.int64)
        starts = np.unique(starts)
        if starts[0] != 0:
            starts = np.insert(starts, 0, 0)
        if starts[-1] != (T - L):
            starts = np.append(starts, T - L)
        starts = np.unique(starts)
        return starts.tolist()

    def _merge_windows(T, L, starts, probs_by_win, p2_by_win=None):
        """Merge window probs back to length T with taper weights. If p2_by_win given, returns std."""
        w = np.hanning(L).astype(np.float64)
        # Clip so endpoints still contribute if only covered once
        w = np.clip(w, 1e-3, None)

        num = np.zeros(T, dtype=np.float64)
        den = np.zeros(T, dtype=np.float64)
        if p2_by_win is not None:
            num2 = np.zeros(T, dtype=np.float64)

        for s, pw in zip(starts, probs_by_win):
            i0 = int(s)
            i1 = i0 + L
            num[i0:i1] += w * pw.astype(np.float64, copy=False)
            den[i0:i1] += w

        if p2_by_win is not None:
            for s, p2w in zip(starts, p2_by_win):
                i0 = int(s)
                i1 = i0 + L
                num2[i0:i1] += w * p2w.astype(np.float64, copy=False)

        den = np.where(den == 0, 1.0, den)
        p = (num / den).astype(np.float32)

        if p2_by_win is None:
            return p, None

        p2 = (num2 / den).astype(np.float64)
        var = np.maximum(0.0, p2 - (p.astype(np.float64) ** 2))
        std = np.sqrt(var).astype(np.float32)
        return p, std

    def _build_one(filename):
        out = load_station_data(filename, data_dir, scaler=None, step_epochs={})
        if out is None:
            return None
        feats, labs, stname, time = out  # time is float64 (v140)
        if int(getattr(args, "apply_scaling", 1)) == 1:
            # Apply the same preprocessing used during dataset build:
            #   1) log_scale only the scaled columns
            #   2) apply the saved scaler to those columns
            cols = scaled_cols
            if cols is None:
                cols = list(range(feats.shape[1]))
            sub = log_scale(feats[:, cols])
            feats = feats.astype(np.float32, copy=False)
            feats[:, cols] = scaler_obj.transform(sub).astype(np.float32, copy=False)
        return feats.astype(np.float32, copy=False), labs.astype(np.float32, copy=False), stname, time

    # Build features
    build_workers = int(getattr(args, "n_workers", 0) or 0)
    built = []
    if build_workers < 2:
        for fn in tqdm(station_files, desc="Inference | build features", disable=not state.is_main, dynamic_ncols=True):
            item = _build_one(fn)
            if item is not None:
                built.append(item)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=build_workers) as ex:
            for item in tqdm(ex.map(_build_one, station_files), total=len(station_files),
                             desc="Inference | build features", disable=not state.is_main, dynamic_ncols=True):
                if item is not None:
                    built.append(item)

    if len(built) == 0:
        raise RuntimeError("Inference build produced 0 valid station series. Check data_dir and file formats.")

    time_by_station = {st: t for (_f, _y, st, t) in built}

    # Split into short vs long series
    short_items = []
    long_items = []
    for f, y, st, t in built:
        if max_length is not None and f.shape[0] > max_length:
            long_items.append((f, y, st, t))
        else:
            short_items.append((f, y, st, t))

    # Infer input_dim from any sample
    input_dim = int(built[0][0].shape[-1])

    class LSTMEncoderDecoder(nn.Module):
        def __init__(self, input_size, linear_size, hidden_size, num_layers, output_size, dropout=0.2, bidirectional=True):
            super().__init__()
            self.bidirectional = bool(bidirectional)
            self.num_directions = 2 if self.bidirectional else 1
            self.linear_in = nn.Linear(input_size, linear_size)

            # For num_layers==1, PyTorch's LSTM dropout is ignored; use explicit dropout modules so MC-dropout works.
            self._explicit_dropout = (int(num_layers) == 1) and (float(dropout) > 0.0)
            self._drop = nn.Dropout(float(dropout)) if self._explicit_dropout else None
            lstm_dropout = 0.0 if int(num_layers) == 1 else float(dropout)

            self.encoder = nn.LSTM(
                input_size=linear_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=lstm_dropout,
                bidirectional=self.bidirectional,
            )

            self.decoder = nn.LSTM(
                input_size=hidden_size * self.num_directions,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=lstm_dropout,
                bidirectional=self.bidirectional,
            )

            self.output_layer = nn.Linear(hidden_size * self.num_directions, output_size)

            init.kaiming_uniform_(self.linear_in.weight, nonlinearity='relu')
            init.zeros_(self.linear_in.bias)

            for name, param in self.encoder.named_parameters():
                if 'weight' in name:
                    init.orthogonal_(param)
                elif 'bias' in name:
                    init.zeros_(param)

            for name, param in self.decoder.named_parameters():
                if 'weight' in name:
                    init.orthogonal_(param)
                elif 'bias' in name:
                    init.zeros_(param)

            init.kaiming_uniform_(self.output_layer.weight, nonlinearity='linear')
            init.zeros_(self.output_layer.bias)

        def forward(self, x, lengths):
            x = torch.relu(self.linear_in(x))
            if self._drop is not None:
                x = self._drop(x)

            packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_enc_out, (h, c) = self.encoder(packed_x)
            enc_out, _ = nn.utils.rnn.pad_packed_sequence(packed_enc_out, batch_first=True)
            if self._drop is not None:
                enc_out = self._drop(enc_out)

            packed_dec_in = nn.utils.rnn.pack_padded_sequence(enc_out, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_dec_out, _ = self.decoder(packed_dec_in, (h, c))
            dec_out, _ = nn.utils.rnn.pad_packed_sequence(packed_dec_out, batch_first=True)
            if self._drop is not None:
                dec_out = self._drop(dec_out)

            return self.output_layer(dec_out)
    model = LSTMEncoderDecoder(
        input_size=input_dim,
        hidden_size=args.hidden_size,
        linear_size=args.linear_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=bool(args.bidirectional),
        output_size=1,
    ).to(state.device)

    # Load checkpoint
    ckpt = torch.load(model_path, map_location="cpu")

    # --- canonical state_dict extraction ---
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        else:
            # If it looks like a raw state_dict (tensor values), use it
            if all(hasattr(v, "shape") for v in ckpt.values() if v is not None):
                state_dict = ckpt
            else:
                raise ValueError(f"Checkpoint dict keys {list(ckpt.keys())} does not contain 'model' or 'state_dict'.")
    else:
        # raw state_dict case
        state_dict = ckpt

    # --- strip DDP 'module.' prefix if present ---
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)

    if state.enabled:
        model = DDP(model, device_ids=[state.local_rank], output_device=state.local_rank, find_unused_parameters=False)

    mc_samples = int(getattr(args, "mc_samples", 0) or 0)
    do_mc = mc_samples > 0
    if do_mc:
        model.train()
        if state.is_main:
            print(f"[Inference] MC-dropout enabled: mc_samples={mc_samples}")
    else:
        model.eval()

    def _predict_windows(feats_batch):
        """
        feats_batch: np.ndarray (B, L, F) float32
        returns:
          p_mean: np.ndarray (B, L) float32
          p2_mean: np.ndarray (B, L) float32 or None
        """
        X = torch.from_numpy(feats_batch).to(state.device, non_blocking=True)
        lengths = torch.full((X.shape[0],), X.shape[1], dtype=torch.int64, device=state.device)

        if do_mc:
            probs_samples = []
            probs2_samples = []
            with torch.no_grad():
                for _ in range(mc_samples):
                    logits = model(X, lengths)
                    p = torch.sigmoid(logits).squeeze(-1).float()
                    probs_samples.append(p)
                    probs2_samples.append(p * p)
                stack = torch.stack(probs_samples, dim=0)   # (S, B, L)
                stack2 = torch.stack(probs2_samples, dim=0) # (S, B, L)
                p_mean = stack.mean(dim=0).cpu().numpy().astype(np.float32, copy=False)
                p2_mean = stack2.mean(dim=0).cpu().numpy().astype(np.float32, copy=False)
            return p_mean, p2_mean
        else:
            with torch.no_grad():
                logits = model(X, lengths)
                p = torch.sigmoid(logits).squeeze(-1).cpu().numpy().astype(np.float32, copy=False)
            return p, None

    # ------------------------------------------------------------
    # SHORT series: original batching/collate behavior (<= max_length)
    # ------------------------------------------------------------
    if len(short_items) > 0:
        class _InMemInfer(torch.utils.data.Dataset):
            def __init__(self, items):
                self.items = items
            def __len__(self):
                return len(self.items)
            def __getitem__(self, idx):
                f, y, st, t = self.items[idx]
                return torch.tensor(f, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), st, t

        infer_ds = _InMemInfer(short_items)

        from functools import partial
        collate = partial(
            collate_fn,
            pad_start=pad_start,
            pad_end=pad_end,
            max_length=max_length,
        )

        infer_loader = DataLoader(
            infer_ds,
            batch_size=int(getattr(args, "infer_batch_size", 32)),
            shuffle=False,
            sampler=None,
            collate_fn=collate,
            **dataloader_common_kwargs(int(getattr(args, "infer_workers", 4)), bool(args.pin_memory),
                                       int(getattr(args, "prefetch_factor", 2)), int(getattr(args, "persistent_workers", 0))),
        )

        with torch.no_grad():
            for X, _Y, lengths, masks, station_names in tqdm(infer_loader, desc="Inference | batches", disable=not state.is_main, dynamic_ncols=True, leave=False, mininterval=1.0):
                X = X.to(state.device, non_blocking=True)
                lengths = lengths.to(state.device, non_blocking=True)

                if do_mc:
                    probs_samples = []
                    probs2_samples = []
                    for _ in range(mc_samples):
                        logits = model(X, lengths)
                        p = torch.sigmoid(logits).float().squeeze(-1)
                        probs_samples.append(p)
                        probs2_samples.append(p * p)
                    stack = torch.stack(probs_samples, dim=0)    # (S, B, T)
                    stack2 = torch.stack(probs2_samples, dim=0)  # (S, B, T)
                    probs_mean = stack.mean(dim=0).cpu().numpy()
                    probs2_mean = stack2.mean(dim=0).cpu().numpy()
                else:
                    logits = model(X, lengths)
                    probs_mean = torch.sigmoid(logits).cpu().numpy()[:, :, 0]
                    probs2_mean = None

                masks_np = masks.cpu().numpy().astype(bool)
                for i, st in enumerate(station_names):
                    t = time_by_station.get(st)
                    if t is None:
                        # fallback
                        t = np.arange(int(masks_np[i].sum()), dtype=np.float64)
                    out = {
                        "time": t,
                        "prob": probs_mean[i, :][masks_np[i]].astype(np.float32, copy=False),
                    }
                    if probs2_mean is not None:
                        p = out["prob"].astype(np.float64)
                        p2 = probs2_mean[i, :][masks_np[i]].astype(np.float64, copy=False)
                        std = np.sqrt(np.maximum(0.0, p2 - p * p)).astype(np.float32)
                        out["prob_std"] = std
                    np.savez(os.path.join(output_dir, f"{st}_inference.npz"), **out)

    # ------------------------------------------------------------
    # LONG series: window inference + merge (T > max_length)
    # ------------------------------------------------------------
    if (max_length is not None) and (len(long_items) > 0):
        # Batch windows within each station for throughput
        win_batch = int(getattr(args, "infer_batch_size", 32))

        for feats_full, _y, st, t_full in long_items:
            T = int(feats_full.shape[0])
            L = int(max_length)

            starts = _balanced_window_starts(T, L, infer_overlap)
            overlaps = []
            for j in range(len(starts)-1):
                overlaps.append(L - (starts[j+1] - starts[j]))
            if state.is_main:
                print(f"[Inference] {st}: length={T} > max_length={L} -> windows={len(starts)} overlap~{overlaps} starts={starts}")

            # Build all windows (nwin, L, F)
            nwin = len(starts)
            F = feats_full.shape[1]
            windows = np.empty((nwin, L, F), dtype=np.float32)
            for j, s in enumerate(starts):
                s = int(s)
                windows[j, :, :] = feats_full[s:s+L, :]

            probs_by_win = []
            p2_by_win = [] if do_mc else None

            # Predict in batches of windows
            for j0 in range(0, nwin, win_batch):
                j1 = min(nwin, j0 + win_batch)
                p_mean_b, p2_mean_b = _predict_windows(windows[j0:j1, :, :])
                probs_by_win.extend([p_mean_b[k, :] for k in range(p_mean_b.shape[0])])
                if do_mc and p2_mean_b is not None:
                    p2_by_win.extend([p2_mean_b[k, :] for k in range(p2_mean_b.shape[0])])

            prob_full, std_full = _merge_windows(T, L, starts, probs_by_win, p2_by_win=p2_by_win)

            out = {
                "time": t_full,
                "prob": prob_full,
            }
            if std_full is not None:
                out["prob_std"] = std_full
            np.savez(os.path.join(output_dir, f"{st}_inference.npz"), **out)

    if state.enabled:
        dist.barrier()
        dist.destroy_process_group()

def build_parser():
    p = argparse.ArgumentParser(description="ML offset detection (composite)", conflict_handler="resolve")
    p.add_argument("--mode", required=True,
                   choices=["build_dataset", "train_folds", "train_final", "inference"],
                   help="Which pipeline stage to run.")

    # DDP / performance knobs (training modes only)
    p.add_argument("--ddp", action="store_true",
               help="Enable DistributedDataParallel when launched via torchrun (single-node multi-GPU).")
    p.add_argument("--ddp_backend", type=str, default="nccl", choices=["nccl", "gloo"],
               help="DDP backend (nccl recommended for CUDA).")
    p.add_argument("--seed", type=int, default=1234, help="Random seed (rank offset is applied under DDP).")
    p.add_argument("--pin_memory", type=int, default=1, choices=[0,1], help="Use pinned-memory DataLoader transfers (CUDA only).")

    # training performance knobs (defaults preserve prior behavior)
    p.add_argument("--train_batch_size", type=int, default=64, help="Batch size for training (folds/final).")
    p.add_argument("--num_epochs", type=int, default=100, help="Training epochs (folds/final).")
    p.add_argument("--learning_rate", type=float, default=1e-4, help="Optimizer learning rate.")
    p.add_argument("--weight_decay", type=float, default=0.0, help="Optimizer weight decay.")
    # LR scheduler (v99-style ReduceLROnPlateau; optional)
    p.add_argument("--use_lr_scheduler", type=int, default=1, choices=[0,1],
                   help="Enable LR scheduler (v99 style ReduceLROnPlateau).")
    p.add_argument("--lr_scheduler", type=str, default="plateau", choices=["plateau","none"],
                   help="LR scheduler type (currently only 'plateau' is supported).")
    p.add_argument("--lr_monitor", type=str, default="train", choices=["train","val"],
                   help="Metric to monitor for LR scheduling ('val' requires val_fraction>0).")
    p.add_argument("--lr_factor", type=float, default=0.5, help="ReduceLROnPlateau factor.")
    p.add_argument("--lr_patience", type=int, default=3, help="ReduceLROnPlateau patience (epochs).")
    p.add_argument("--lr_threshold", type=float, default=0.01, help="ReduceLROnPlateau threshold.")
    p.add_argument("--lr_threshold_mode", type=str, default="rel", choices=["rel","abs"], help="ReduceLROnPlateau threshold_mode.")
    p.add_argument("--lr_cooldown", type=int, default=0, help="ReduceLROnPlateau cooldown.")
    p.add_argument("--lr_min_lr", type=float, default=0.0, help="ReduceLROnPlateau min_lr.")
    p.add_argument("--stop_when_min_lr", action="store_true", help="Stop training early once LR reaches lr_min_lr.")
    p.add_argument("--lr_eps", type=float, default=1e-8, help="ReduceLROnPlateau eps.")

    # Early stopping (optional; safe defaults)
    p.add_argument("--early_stop", type=int, default=0, choices=[0,1], help="Enable early stopping.")
    p.add_argument("--early_stop_metric", type=str, default="train", choices=["train","val"],
                   help="Metric for early stopping ('val' requires val_fraction>0).")
    p.add_argument("--early_stop_patience", type=int, default=50, help="Patience epochs with no improvement before stopping.")
    p.add_argument("--early_stop_min_delta", type=float, default=1e-6, help="Minimum relative improvement to reset patience.")
    p.add_argument("--early_stop_warmup", type=int, default=10, help="Do not early-stop before this many epochs.")
    p.add_argument("--early_stop_loss_threshold", type=float, default=5e-5,
                   help="Stop immediately if monitored metric drops below this value (set <=0 to disable).")

    p.add_argument("--train_workers", type=int, default=32, help="DataLoader num_workers for training (0 disables multiprocessing).")
    p.add_argument("--val_workers", type=int, default=0,
                   help="Validation DataLoader workers (0 => uses train_workers).")
    p.add_argument("--bucket_by_length", type=int, choices=[0,1], default=0,
                   help="If 1, use length-aware bucketing to reduce padding within batches.")
    p.add_argument("--bucket_num_buckets", type=int, default=50,
                   help="Number of buckets used when bucket_by_length=1.")
    p.add_argument("--bucket_shuffle", type=int, choices=[0,1], default=1,
                   help="Shuffle indices within each bucket each epoch.")
    p.add_argument("--bucket_shuffle_batches", type=int, choices=[0,1], default=1,
                   help="Shuffle order of batches each epoch.")
    p.add_argument("--bucket_drop_last", type=int, choices=[0,1], default=0,
                   help="Drop last incomplete batch (recommended for DDP).")
    p.add_argument("--prefetch_factor", type=int, default=4, help="DataLoader prefetch_factor (only when train_workers>0).")
    p.add_argument("--persistent_workers", type=int, default=1, choices=[0,1], help="Keep DataLoader workers alive between epochs (train_workers>0).")
    p.add_argument("--amp", type=int, default=0, choices=[0,1], help="Enable mixed precision autocast on CUDA.")
    p.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16","fp16"], help="Autocast dtype (bf16 recommended on Ada GPUs; fp16 uses GradScaler).")
    p.add_argument("--compile", type=int, default=0, choices=[0,1], help="Enable torch.compile for the model (PyTorch 2.x).")
    p.add_argument("--compile_mode", type=str, default="default", choices=["default","reduce-overhead","max-autotune"], help="torch.compile mode.")
    p.add_argument("--k_folds", type=int, default=5, help="Number of folds for train_folds.")
    p.add_argument("--fold_seed", type=int, default=42, help="Random seed for KFold shuffling.")

    # Shared paths
    p.add_argument("--dataset_pt", default="train_v90_dataset.pt",
                   help="Path to saved torch dataset (.pt) for training modes.")
    p.add_argument("--metadata_npy", default="metadata_v90.npy",
                   help="Path to saved metadata (.npy) for training modes.")

    # build_dataset options
    p.add_argument("--data_dir", default="./ML_trainrand_stats_v90/",
                   help="Folder containing station .data files (for build_dataset and inference).")
    p.add_argument("--step_epochs_file", default="ML_steplisttest_rand_v90",
                   help="Step epoch list file used by the dataset builder.")
    p.add_argument("--max_length", type=int, default=8000, help="Max padded sequence length.")
    p.add_argument("--apply_scaling", type=int, default=1, help="1 to apply scaler, 0 to skip.")
    p.add_argument("--scaler_path", "--scaler_file", default="STEPSV_scaler.pkl", help="Scaler .pkl path (input if apply_scaling=1, or default output if make_scaler=1).")
    p.add_argument("--scaler_out", default="", help="(alias) Output path for scaler when make_scaler=1. If blank, uses scaler_path.")
    p.add_argument("--make_scaler", type=int, default=0, help="1 to compute a new scaler, else 0.")
    p.add_argument("--figure_on", type=int, default=0, help="1 to show debug figures during build.")
    p.add_argument("--dataset_out", default="train_v90_dataset.pt", help="Output dataset .pt file.")
    p.add_argument("--metadata_out", default="metadata_v90.npy", help="Output metadata .npy file.")
    p.add_argument("--progress_log", default=None, help="Optional path to append build_dataset progress logs.")
    p.add_argument("--progress_every", type=int, default=50000, help="Log every N stations during build_dataset scaling/packing.")
    p.add_argument("--n_workers", type=int, default=0, help="Number of worker processes for station loading (build_dataset). 0 = serial.")
    p.add_argument("--log_path", default="", help="(alias) Log file path for build_dataset progress; same as progress_log.")

    # inference options
    p.add_argument("--model", dest="model_path", default=None, help="Model checkpoint (.pth) for inference.")
    p.add_argument("--output_dir", "--out_dir", default="./inference_out/", help="Where to write inference outputs.")
    p.add_argument("--use_gpu", type=int, default=1, help="1 to use GPU if available.")
    p.add_argument("--gpu_id", type=int, default=0, help="GPU id for inference.")

    # --- Training/inference hyperparameters (parameterized; defaults keep prior behavior) ---
    p.add_argument("--version_tag", type=str, default="v100", help="Tag used in saved model/log filenames.")
    p.add_argument("--train_out_dir", type=str, default="./train_outputs", help="Directory for saved models/logs.")
    p.add_argument("--val_batch_size", type=int, default=8, help="Validation batch size.")
    p.add_argument("--linear_size", type=int, default=16, help="Linear projection size before LSTM.")
    p.add_argument("--hidden_size", type=int, default=32, help="LSTM hidden size.")
    p.add_argument("--bidirectional", type=int, default=1, help="1=bidirectional LSTM, 0=unidirectional.")
    p.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers.")
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout in LSTMs.")
    p.add_argument("--pos_weight", type=float, default=1.0, help="pos_weight for BCEWithLogitsLoss.")
    p.add_argument("--pad_start", type=int, default=0, help="Pad this many samples at start of each sequence.")
    p.add_argument("--pad_end", type=int, default=0, help="Pad this many samples at end of each sequence.")
    p.add_argument("--save_every", type=int, default=1, help="Save an epoch checkpoint every N epochs (train_final).")
    p.add_argument("--val_dataset_pt", default="", help="Optional .pt dataset to use as validation set in train_final (overrides val_fraction when provided).")
    p.add_argument("--val_metadata_npy", default="", help="Optional metadata .npy for the validation dataset (currently unused; kept for symmetry).")
    p.add_argument("--val_fraction", type=float, default=0.0, help="Hold out this fraction for validation in train_final (0 disables).")

    # Inference loader settings
    p.add_argument("--infer_batch_size", type=int, default=32, help="Inference batch size.")
    p.add_argument("--infer_workers", type=int, default=4, help="DataLoader workers for inference.")

    p.add_argument("--mc_samples", type=int, default=0, help="Monte Carlo dropout samples at inference (0 disables).")
    return p





def main():
    args = build_parser().parse_args()

    # Backward-compatible aliases (do not overwrite existing explicit args)
    if getattr(args, "scaler_file", None) and not getattr(args, "scaler_path", None):
        args.scaler_path = args.scaler_file
    if getattr(args, "out_dir", None) and not getattr(args, "output_dir", None):
        args.output_dir = args.out_dir

    if args.mode == "build_dataset":
        mode_build_dataset(args)
    elif args.mode == "train_folds":
        mode_train_folds(args)
    elif args.mode == "train_final":
        mode_train_final(args)
    elif args.mode == "inference":
        mode_inference(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")




if __name__ == '__main__':
    main()
