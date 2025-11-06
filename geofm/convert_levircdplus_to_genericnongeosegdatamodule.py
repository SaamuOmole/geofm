"""
Convert LEVIR-CD+ folder layout (train/{A,B,label}, test/{A,B,label})
into a directory structure compatible with TerraTorch GenericNonGeoSegmentationDataModule.

Output layout:
  out_root/
    train/images (stacked channels)
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

def read_rgb_png(path: Path):
    """
    Read and return arr of shape C, H, W

    Args:
        path (Path): Path to image

    Returns:
        _type_: Array of shape C, H, W
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)  # (H,W,C)
    return np.transpose(arr, (2,0,1))   # (C,H,W)

def write_stacked_tiff(dest: Path, array_bhw):
    """
    Write stacked array with shape (bands, H, W) using rasterio

    Args:
        dest (Path): Path to write stacked array to
        array_bhw (_type_): Stacked array
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
    """
    Save single-channel mask (0/255) as PNG

    Args:
        dest (Path): Path to write mask array to
        mask_arr (_type_): Mask array
    """
    img = Image.fromarray(mask_arr.astype(np.uint8))
    img.save(dest)

def process_split(split_dir: Path, out_images_dir: Path, out_labels_dir: Path, stems=None, suffix="_stacked.tif"):
    """
    Process directory with subfolders A/, B/, label/. For each
    filename in A, find same filename in B and label, stack and save.

    Args:
        split_dir (Path): Path where A/, B/, label/ are located 
        out_images_dir (Path): Path to stacked images
        out_labels_dir (Path): Path to mask labels
        stems (optional): All file names. Defaults to None.
        suffix (str, optional): Appended suffix to stacked images. Defaults to "_stacked.tif".

    Raises:
        RuntimeError: Raises when subfolders not in the expected A/, B/, label/ structure 
    """
    A_dir = split_dir / "A"
    B_dir = split_dir / "B"
    L_dir = split_dir / "label"

    if not (A_dir.exists() and B_dir.exists() and L_dir.exists()):
        raise RuntimeError(f"Expected subfolders A/, B/, label/ in {split_dir}")

    # Creates directories
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    # Collect stems and keep their original extension mapping
    stem_to_ext = {}
    for p in sorted(A_dir.iterdir()):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            stem_to_ext[p.stem] = p.suffix.lower()

    if stems is None:
        stems = sorted(stem_to_ext.keys())
    else:
        # Ensure stems exist in A_dir
        stems = [s for s in stems if s in stem_to_ext]
        # missing_stems = set(stems) ^ set(stem_to_ext.keys())

    if len(stems) == 0:
        print(f"Warning: no stems to process in {split_dir}")
        return

    missing = 0
    processed = 0
    for stem in stems:
        ext = stem_to_ext.get(stem, ".png")  # Defaults to .png if unknown
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

        # Read & stack
        a = read_rgb_png(a_path)  # (3,H,W)
        b = read_rgb_png(b_path)  # (3,H,W)

        # Check shapes and resize b if needed
        if a.shape[1:] != b.shape[1:]:
            print(f"WARNING: spatial mismatch for {stem} (A vs B); resizing B to A using PIL (bilinear).")
            b_img = Image.open(b_path).convert("RGB").resize((a.shape[2], a.shape[1]), resample=Image.BILINEAR)
            b = np.transpose(np.array(b_img, dtype=np.uint8), (2,0,1))

        # Stack A and B images and write to directory
        stacked = np.concatenate([a, b], axis=0)  # (6,H,W) 
        out_img_path = out_images_dir / f"{stem}{suffix}"
        write_stacked_tiff(out_img_path, stacked)

        # Open grayscale label then force to 0/255
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

    # Collect train filenames by stem from dir train/A
    train_A = sorted([p for p in (train_dir / "A").glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    train_stems = [p.stem for p in train_A]

    # Create split indices
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

    # Prepare output directories
    for split in ("train", "val", "test"):
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)

    if len(train_stems_list) > 0:
        print("Processing TRAIN split...")
        process_split(train_dir, out_root / "train" / "images", out_root / "train" / "labels", stems=train_stems_list)

    if len(val_stems_list) > 0:
        print("Processing VAL split...")
        # Pass only val_stems so only selected items are processed
        process_split(train_dir, out_root / "val" / "images", out_root / "val" / "labels", stems=val_stems_list)
    else:
        # Create empty val dirs if val_fraction == 0
        print("No validation split requested; val folder will remain empty.")

    # Process test set
    print("Processing TEST split...")
    process_split(test_dir, out_root / "test" / "images", out_root / "test" / "labels", stems=None)
    
    
    # Print results of conversion
    print("Conversion complete. Output layout:")
    print(f"  {out_root}/train/images  {len(list((out_root / 'train' / 'images').glob('*')))}  files")
    print(f"  {out_root}/train/labels  {len(list((out_root / 'train' / 'labels').glob('*')))}  files")
    print(f"  {out_root}/val/images  {len(list((out_root / 'val' / 'images').glob('*')))}  files")
    print(f"  {out_root}/val/labels  {len(list((out_root / 'val' / 'labels').glob('*')))}  files")
    print(f"  {out_root}/test/images  {len(list((out_root / 'test' / 'images').glob('*')))}  files")
    print(f"  {out_root}/test/labels  {len(list((out_root / 'test' / 'labels').glob('*')))}  files")
    
if __name__ == "__main__":
    main()