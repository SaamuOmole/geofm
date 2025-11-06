import numpy as np
from pathlib import Path
from math import sqrt
import tifffile
from tqdm import tqdm

# Script to compute means and stds of stacked tiff images
# in the training directory per channel for normalising the images

def compute_stats_for_stacked_tifs(images_dir, split="train"):
    """
    Reads *_stacked.tif files from root_dir/split/images and
    computes means and stds per channel

    Args:
        images_dir: Path to stacked images
        split (str, optional): The particular subfolder e.g., train, test. Defaults to "train".

    Raises:
        RuntimeError: Raises when no images are present

    Returns:
        The means and stds per channel 
    """
    img_dir = Path(images_dir) / split / "images"
    paths = sorted(list(img_dir.glob("*_stacked.tif")))
    if len(paths) == 0:
        raise RuntimeError("No stacked TIFFs found in " + str(img_dir))
    
    sum_px = None
    sum_sq_px = None
    total_pixels = 0

    for p in tqdm(paths, desc="Computing mean/std"):
        arr = tifffile.imread(str(p))
        arr = np.asarray(arr)
        
        # Normalize shape to (bands, H, W)
        if arr.ndim == 3 and arr.shape[2] > 1 and arr.shape[0] != 6:
            arr = np.transpose(arr, (2,0,1)) # (6,H,W)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...] # (1,H,W) if single band

        bands, H, W = arr.shape
        
        # Flatten per band
        flat = arr.reshape(bands, -1).astype(np.float64) # Use float

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

    # If there are >= 6 in the list
    # Approximate calculation
    if len(means6) >= 6:
        means3 = 0.5 * (means6[:3] + means6[3:6])
        stds3  = 0.5 * (stds6[:3] + stds6[3:6])
    else:
        means3 = means6
        stds3 = stds6

    return means3.tolist(), stds3.tolist()
