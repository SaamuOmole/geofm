import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
from pathlib import Path

# Script to plot random examples of stacked images and mask labels for the levircdplus dataset

img_path = Path("/Users/samuel.omole/Desktop/repos/geofm_datasets/levircdplus_restructured/train/images") / sorted(list(Path("/Users/samuel.omole/Desktop/repos/geofm_datasets/levircdplus_restructured/train/images").glob("*_stacked.tif")))[100].name # Change index to plot random sample
mask_path = Path("/Users/samuel.omole/Desktop/repos/geofm_datasets/levircdplus_restructured/train/labels") / (img_path.stem.replace("_stacked","") + "_mask.png")

print("img:", img_path)
print("mask:", mask_path)

arr = tifffile.imread(str(img_path))
# Normalize to H,W,C
if arr.ndim == 3 and arr.shape[0] <= 64 and arr.shape[0] != arr.shape[2]:
    arr = np.transpose(arr, (1,2,0))

H,W,C = arr.shape
print("image shape:", arr.shape)

# Handle the stacked layout: 6 channels (R1,G1,B1,R2,G2,B2)
if C == 6:
    A_rgb = arr[..., :3]
    B_rgb = arr[..., 3:6]
elif C == 3:
    A_rgb = arr
    B_rgb = None
else:
    # Split in half as two timesteps
    t = C // 2
    A_rgb = arr[..., :t]
    B_rgb = arr[..., t:2*t]

mask = np.array(Image.open(mask_path).convert("L"))

# Ensure mask same size
if mask.shape != (H,W):
    mask = np.array(Image.fromarray(mask).resize((W,H), resample=Image.NEAREST))

# Show plot 
fig, axs = plt.subplots(1, 3 if B_rgb is not None else 2, figsize=(12,4))
axs[0].imshow(A_rgb)
axs[0].set_title("A (t1)")
axs[0].axis("off")
if B_rgb is not None:
    axs[1].imshow(B_rgb)
    axs[1].set_title("B (t2)")
    axs[1].axis("off")
    ax_mask = axs[2]
else:
    ax_mask = axs[1]
ax_mask.imshow(mask, cmap="gray")
ax_mask.set_title("Mask")
ax_mask.axis("off")
plt.tight_layout()
plt.show()
