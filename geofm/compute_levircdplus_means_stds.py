import numpy as np
from pathlib import Path
from math import sqrt
import tifffile
from tqdm import tqdm

# def compute_dataset_mean_std(images_dir: Path, glob='*_stacked.tif', sample_limit=None):
#     """
#     Compute per-channel mean and std for multi-channel images stored as HWC or (H,W,C) TIFFs.
#     Returns (means, stds) in 0..1 range (i.e. divided by 255).
#     - images_dir: Path to folder with stacked tifs
#     - sample_limit: optional int to only use N images (useful for quick estimate)
#     """
#     files = sorted(images_dir.glob(glob))
#     if sample_limit:
#         files = files[:sample_limit]
#     if len(files) == 0:
#         raise RuntimeError(f"No files found in {images_dir} with pattern {glob}")

#     # compute channel-wise sum and sum of squares in dtype float64
#     first = tifffile.imread(str(files[0]))
#     # tifffile returns HWC (H,W,C) typically; normalize to (C,H,W) then convert to 0..255 values
#     if first.ndim == 3 and first.shape[2] <= 64:
#         C = first.shape[2]
#     elif first.ndim == 3 and first.shape[0] <= 64:
#         # sometimes tifffile gives (C,H,W)
#         C = first.shape[0]
#     else:
#         # fallback
#         C = first.shape[-1]

#     chan_sum = np.zeros((C,), dtype=np.float64)
#     chan_sumsq = np.zeros((C,), dtype=np.float64)
#     total_pixels = 0  # total per-channel pixel count

#     for p in tqdm(files, desc="Computing mean/std"):
#         im = tifffile.imread(str(p))  # (H,W,C) or (C,H,W)
#         if im.ndim == 3 and im.shape[0] <= 64 and im.shape[0] != im.shape[2]:
#             # assume (C,H,W) -> convert to (H,W,C)
#             im = np.transpose(im, (1,2,0))
#         if im.ndim == 2:
#             im = im[..., None]  # single channel
#         H, W, ch = im.shape
#         im = im.astype(np.float64)  # important for sums
#         # reshape to (pixels, C)
#         arr = im.reshape(-1, ch)  # (H*W, C)
#         chan_sum += arr.sum(axis=0)
#         chan_sumsq += (arr ** 2).sum(axis=0)
#         total_pixels += H * W

#     means_255 = chan_sum / total_pixels           # means in 0..255 range
#     variances_255 = (chan_sumsq / total_pixels) - (means_255 ** 2)
#     stds_255 = np.sqrt(np.maximum(variances_255, 0.0))

#     # convert to 0..1 range for albumentations Normalize (divide by 255)
#     means = (means_255 / 255.0).tolist()
#     stds = (stds_255 / 255.0).tolist()

#     return means, stds

def compute_stats_for_stacked_tifs(images_dir, split="train"):
    """
    Reads up to max_samples *_stacked.tif files from root_dir/split/images and
    returns:
      - means6: ndarray shape (6,) for the six bands in file order
      - stds6:  ndarray shape (6,)
      - means3: ndarray shape (3,) averaged over A and B (per-band)
      - stds3:  ndarray shape (3,)
    """
    img_dir = Path(images_dir) / split / "images"
    paths = sorted(list(img_dir.glob("*_stacked.tif")))
    if len(paths) == 0:
        raise RuntimeError("No stacked TIFFs found in " + str(img_dir))
    
    sum_px = None
    sum_sq_px = None
    total_pixels = 0

    for p in tqdm(paths, desc="Computing mean/std"):
        arr = tifffile.imread(str(p))  # likely (H,W,6) or (6,H,W)
        arr = np.asarray(arr)
        # normalize shape to (bands, H, W)
        if arr.ndim == 3 and arr.shape[2] > 1 and arr.shape[0] != 6:
            arr = np.transpose(arr, (2,0,1))   # (6,H,W)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]   # (1,H,W) if single band

        bands, H, W = arr.shape
        # flatten per band
        flat = arr.reshape(bands, -1).astype(np.float64)  # use float for accumulation

        if sum_px is None:
            sum_px = flat.sum(axis=1)
            sum_sq_px = (flat**2).sum(axis=1)
        else:
            sum_px += flat.sum(axis=1)
            sum_sq_px += (flat**2).sum(axis=1)
        total_pixels += H * W

    means6 = sum_px / total_pixels
    var6 = (sum_sq_px / total_pixels) - (means6**2)
    stds6 = np.sqrt(np.maximum(var6, 1e-12))

    # assume the 6 bands are stacked as [A_R,A_G,A_B, B_R,B_G,B_B]
    if len(means6) >= 6:
        means3 = 0.5 * (means6[:3] + means6[3:6])
        stds3  = 0.5 * (stds6[:3] + stds6[3:6])   # approximate (better would combine variances but this is reasonable)
    else:
        means3 = means6
        stds3 = stds6

    return means3.tolist(), stds3.tolist()
