#!/usr/bin/env python3
"""
Convert LEVIR-CD+ folder layout (train/{A,B,label}, test/{A,B,label})
into a directory structure compatible with TerraTorch GenericNonGeoSegmentationDataModule.

Output layout:
  out_root/
    train/images   (stacked tiff: A channels then B channels, e.g. 6 channels)
    train/labels   (single-channel 0/255 PNG)
    val/...
    test/...

Usage:
  python convert_levircdplus_to_genericnongeosegdatamodule.py \
    --levir_root /Users/samuel.omole/Desktop/repos/geofm_datasets/levircdplus/LEVIR-CD+ \
    --out_dir /Users/samuel.omole/Desktop/repos/geofm_datasets/levir_restructured \
    --val_fraction 0.2
"""
from pathlib import Path
import argparse
from PIL import Image
import numpy as np
import sys
import random
import rasterio

# try tifffile then rasterio
# HAS_TIFFFILE = False
# HAS_RASTERIO = False
# try:
#     import tifffile
#     HAS_TIFFFILE = True
# except Exception:
#     try:
#         import rasterio
#         from rasterio.transform import from_origin
#         HAS_RASTERIO = True
#     except Exception:
#         pass

def read_rgb_png(path: Path):
    """Return numpy array shape (3, H, W), dtype=uint8"""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)  # (H,W,3)
    return np.transpose(arr, (2,0,1))   # (3,H,W)

# def write_stacked_tiff(dest: Path, array_bhw):
#     """
#     Write stacked array with shape (bands, H, W).
#     Uses tifffile (preferred) or rasterio fallback.
#     """
#     bands, H, W = array_bhw.shape
#     if HAS_TIFFFILE:
#         # tifffile expects (H, W, channels) for multi-channel
#         arr_hw_c = np.transpose(array_bhw, (1,2,0))
#         tifffile.imwrite(str(dest), arr_hw_c)
#         return
#     if HAS_RASTERIO:
#         # create a minimal rasterio profile (no CRS/transform because PNGs are not georef)
#         import rasterio
#         profile = {
#             "driver": "GTiff",
#             "height": H,
#             "width": W,
#             "count": bands,
#             "dtype": array_bhw.dtype,
#             "compress": "deflate",
#         }
#         with rasterio.open(str(dest), "w", **profile) as dst:
#             dst.write(array_bhw)
#         return
#     raise RuntimeError("Neither 'tifffile' nor 'rasterio' is installed. Install tifffile (`pip install tifffile`).")

def write_stacked_tiff(dest: Path, array_bhw):
    """
    Write stacked array with shape (bands, H, W) using rasterio (bands-first).
    This ensures rasterio / rioxarray open returns an xarray.DataArray (with .to_numpy()).
    """
    bands, H, W = array_bhw.shape
    profile = {
        "driver": "GTiff",
        "height": H,
        "width": W,
        "count": bands,
        "dtype": array_bhw.dtype,
        "compress": "deflate",
    }
    with rasterio.open(str(dest), "w", **profile) as dst:
        dst.write(array_bhw)   # expects shape (bands, H, W)


def write_mask_png(dest: Path, mask_arr):
    """Save single-channel mask (0/255) as PNG"""
    img = Image.fromarray(mask_arr.astype(np.uint8))
    img.save(dest)

# def process_split(split_dir: Path, out_images_dir: Path, out_labels_dir: Path, stems=None, suffix="_stacked.tif"):
#     """
#     Process a directory with subfolders A/, B/, label/
#     For each filename in A, find same filename in B and label, stack and save.
#     """
#     A_dir = split_dir / "A"
#     B_dir = split_dir / "B"
#     L_dir = split_dir / "label"

#     if not (A_dir.exists() and B_dir.exists() and L_dir.exists()):
#         raise RuntimeError(f"Expected subfolders A/, B/, label/ in {split_dir}")

#     out_images_dir.mkdir(parents=True, exist_ok=True)
#     out_labels_dir.mkdir(parents=True, exist_ok=True)

#     # get all filenames and stems from A_dir if stems not provided
#     if stems is None:
#         filenames = sorted([p.name for p in A_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
#         stems = sorted([p.stem for p in A_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    
#     # if stems are known beforehand
#     else:
#         filenames = sorted([f"{stem}.png" for stem in stems])
        
#     # start counter for missing files
#     missing = 0
#     for fname in filenames:
#         a_path = A_dir / fname
#         b_path = B_dir / fname
#         l_path = L_dir / fname

#         if not a_path.exists():
#             print(f"WARNING: missing A/{fname}; skipping")
#             missing += 1
#             continue
#         if not b_path.exists():
#             print(f"WARNING: missing B/{fname}; skipping")
#             missing += 1
#             continue
#         if not l_path.exists():
#             print(f"WARNING: missing label/{fname}; skipping")
#             missing += 1
#             continue
    
#     if missing:
#         print(f"Processed {len(filenames)-missing} samples; skipped {missing} samples due to missing pairs.")
        
#     for stem in stems:
#         out_img_path = out_images_dir / f"{stem}{suffix}"
#         out_lbl_path = out_labels_dir / f"{stem}_mask.png"

#         # read & stack
#         a_stem_path = A_dir / f"{stem}.png"
#         b_stem_path = B_dir / f"{stem}.png"
#         l_stem_path = L_dir / f"{stem}.png"
#         a = read_rgb_png(a_stem_path)  # (3,H,W)
#         b = read_rgb_png(b_stem_path)
#         # check shapes
#         if a.shape[1:] != b.shape[1:]:
#             print(f"WARNING: spatial mismatch for {stem}.png (A vs B); resizing B to A using PIL (bilinear).")
#             # resize b using PIL to match A
#             b_img = Image.open(b_path).convert("RGB").resize((a.shape[2], a.shape[1]), resample=Image.BILINEAR)
#             b = np.transpose(np.array(b_img, dtype=np.uint8), (2,0,1))

#         stacked = np.concatenate([a, b], axis=0)  # (6,H,W)

#         # write stacked tiff
#         write_stacked_tiff(out_img_path, stacked)

#         # process label: convert to single-channel 0/255
#         lbl = Image.open(l_stem_path).convert("L")
#         lbl_arr = np.array(lbl)
#         # if mask uses 255 vs 1, convert anything >0 to 255
#         if not set(np.unique(lbl_arr)).issubset({0, 255}):
#             lbl_arr = np.where(lbl_arr > 0, 255, 0).astype(np.uint8)
#         write_mask_png(out_lbl_path, lbl_arr)

def process_split(split_dir: Path, out_images_dir: Path, out_labels_dir: Path, stems=None, suffix="_stacked.tif"):
    """
    Corrected: Process a directory with subfolders A/, B/, label/
    For each filename in A, find same filename in B and label, stack and save.
    """
    A_dir = split_dir / "A"
    B_dir = split_dir / "B"
    L_dir = split_dir / "label"

    if not (A_dir.exists() and B_dir.exists() and L_dir.exists()):
        raise RuntimeError(f"Expected subfolders A/, B/, label/ in {split_dir}")

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    # collect stems and keep their original extension mapping
    # mapping: stem -> ext (including the dot, e.g. ".png")
    stem_to_ext = {}
    for p in sorted(A_dir.iterdir()):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            stem_to_ext[p.stem] = p.suffix.lower()

    if stems is None:
        stems = sorted(stem_to_ext.keys())
    else:
        # ensure stems exist in A_dir
        stems = [s for s in stems if s in stem_to_ext]
        # missing_stems = set(stems) ^ set(stem_to_ext.keys())
        # we don't raise here; we'll warn per-stem below

    if len(stems) == 0:
        print(f"Warning: no stems to process in {split_dir}")
        return

    missing = 0
    processed = 0
    for stem in stems:
        ext = stem_to_ext.get(stem, ".png")  # default to .png if unknown
        a_path = A_dir / f"{stem}{ext}"
        b_path = B_dir / f"{stem}{ext}"
        l_path = L_dir / f"{stem}{ext}"

        if not a_path.exists():
            print(f"WARNING: missing A/{stem}{ext}; skipping")
            missing += 1
            continue
        if not b_path.exists():
            print(f"WARNING: missing B/{stem}{ext}; skipping")
            missing += 1
            continue
        if not l_path.exists():
            print(f"WARNING: missing label/{stem}{ext}; skipping")
            missing += 1
            continue

        # read & stack
        a = read_rgb_png(a_path)  # (3,H,W)
        b = read_rgb_png(b_path)  # (3,H,W)

        # check shapes and resize b if needed
        if a.shape[1:] != b.shape[1:]:
            print(f"WARNING: spatial mismatch for {stem} (A vs B); resizing B to A using PIL (bilinear).")
            b_img = Image.open(b_path).convert("RGB").resize((a.shape[2], a.shape[1]), resample=Image.BILINEAR)
            b = np.transpose(np.array(b_img, dtype=np.uint8), (2,0,1))

        stacked = np.concatenate([a, b], axis=0)  # (6,H,W)
        out_img_path = out_images_dir / f"{stem}{suffix}"
        write_stacked_tiff(out_img_path, stacked)

        # process label: open original label (grayscale) then force 0/255
        lbl_img = Image.open(l_path).convert("L")
        lbl_arr = np.array(lbl_img)
        if not set(np.unique(lbl_arr)).issubset({0, 255}):
            lbl_arr = np.where(lbl_arr > 0, 255, 0).astype(np.uint8)
        out_lbl_path = out_labels_dir / f"{stem}_mask.png"
        write_mask_png(out_lbl_path, lbl_arr)

        processed += 1

    print(f"Processed {processed} samples; skipped {missing} samples due to missing pairs.")


def main():
    parser = argparse.ArgumentParser(description="Convert LEVIR-CD+ to TerraTorch GenericNonGeoSegmentation layout")
    parser.add_argument("--levir_root", required=True, type=Path,
                        help="Path to LEVIR-CD+ root (contains train/ and test/)")
    parser.add_argument("--out_dir", required=True, type=Path, help="Output root directory")
    parser.add_argument("--val_fraction", type=float, default=0.2, help="Fraction of train to use as validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    levir_root = args.levir_root.expanduser().resolve()
    out_root = args.out_dir.expanduser().resolve()

    train_dir = levir_root / "train"
    test_dir = levir_root / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise RuntimeError("Expected 'train' and 'test' directories under levir_root")

    # collect train filenames (by stem) from train/A
    train_A = sorted([p for p in (train_dir / "A").glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    train_stems = [p.stem for p in train_A]

    # create split indices
    n = len(train_stems)
    if n == 0:
        raise RuntimeError("No images found in directory train/A")
    n_val = max(1, int(n * args.val_fraction)) if 0 < args.val_fraction < 1 else 0
    random.seed(args.seed)
    shuffled = list(train_stems)
    random.shuffle(shuffled)
    train_stems_list = shuffled[:-n_val] if n_val > 0 else shuffled
    val_stems_list = shuffled[-n_val:] if n_val > 0 else []

    print(f"Found {n} train samples -> {len(train_stems_list)} train / {len(val_stems_list)} val")
    print(f"Test samples will be read from: {test_dir}")

    # prepare output directories
    for split in ("train", "val", "test"):
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)

    if len(train_stems_list) > 0:
        print("Processing TRAIN split...")
        process_split(train_dir, out_root / "train" / "images", out_root / "train" / "labels", stems=train_stems_list)

    if len(val_stems_list) > 0:
        print("Processing VAL split...")
        # process_split uses split_dir argument for A/B/label locations; we still point to original train_dir,
        # but pass only val_stems so only selected items are processed.
        process_split(train_dir, out_root / "val" / "images", out_root / "val" / "labels", stems=val_stems_list)
    else:
        # create empty val dirs if val_fraction == 0
        print("No validation split requested; val folder will remain empty.")

    # process test entirely
    print("Processing TEST split...")
    process_split(test_dir, out_root / "test" / "images", out_root / "test" / "labels", stems=None)
    
    
    # print results of conversion
    print("Conversion complete. Output layout:")
    print(f"  {out_root}/train/images  {len(list((out_root / 'train' / 'images').glob('*')))}  files")
    print(f"  {out_root}/train/labels  {len(list((out_root / 'train' / 'labels').glob('*')))}  files")
    print(f"  {out_root}/val/images  {len(list((out_root / 'val' / 'images').glob('*')))}  files")
    print(f"  {out_root}/val/labels  {len(list((out_root / 'val' / 'labels').glob('*')))}  files")
    print(f"  {out_root}/test/images  {len(list((out_root / 'test' / 'images').glob('*')))}  files")
    print(f"  {out_root}/test/labels  {len(list((out_root / 'test' / 'labels').glob('*')))}  files")
    
    
if __name__ == "__main__":
    main()