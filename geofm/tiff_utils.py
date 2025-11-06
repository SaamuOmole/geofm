import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple

# Script contains helper functions to visualise downloaded Sentinel TIFF images

def normalize_for_display(band: np.ndarray, percentile: Tuple[float, float] = (2, 98)) -> np.ndarray:
    """
    Normalize band for display and clipping with percentiles.

    Args:
        band (np.ndarray): Input band array
        percentile (Tuple[float, float], optional): Tuple of (min_percentile, max_percentile). Defaults to (2, 98).

    Returns:
        np.ndarray: Normalized array scaled to 0-1
    """

    # Remove any NaN or infinite values
    band_clean = band[np.isfinite(band)]
    
    if len(band_clean) == 0:
        return np.zeros_like(band)
    
    # Calculate percentiles
    p_min, p_max = np.percentile(band_clean, percentile)
    
    # Clip and normalize
    band_normalized = np.clip(band, p_min, p_max)
    band_normalized = (band_normalized - p_min) / (p_max - p_min + 1e-8) # To avoid zero division
    
    return band_normalized


def visualize_sentinel2(
    image_path: str,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
):
    """
    Visualize Sentinel-2 multi-band image.
    
    Shows:
    - True color RGB bands

    Args:
        image_path (str): Path to the Sentinel-2 GeoTIFF
        figsize (Tuple[int, int], optional): Figure size. Defaults to (15, 10).
        save_path (Optional[str], optional): Optional path to save the visualization. Defaults to None.
    """
    print(f"\nVisualizing Sentinel-2 image: {image_path}")
    
    with rasterio.open(image_path) as src:
        # Read metadata
        print(f"  Bands: {src.count}")
        print(f"  Size: {src.width} x {src.height}")
        print(f"  CRS: {src.crs}")
        print(f"  Bounds: {src.bounds}")
        
        # Read all bands
        data = src.read()  # Shape: (bands, height, width)
        
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        fig.suptitle(f'Sentinel-2 Visualization\n{Path(image_path).name}', fontsize=14, fontweight='bold')
        
        # True Color RGB (B04, B03, B02) with indices 3, 2, 1
        if src.count >= 4:
            rgb = np.stack([
                normalize_for_display(data[3]),  # Red (B04)
                normalize_for_display(data[2]),  # Green (B03)
                normalize_for_display(data[1])   # Blue (B02)
            ], axis=-1)
            
            axes.imshow(rgb)
            axes.set_title('True Color RGB\n(B04-B03-B02)', fontweight='bold')
            axes.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved visualization to: {save_path}")
        
        plt.show()

def visualize_sentinel1(
    image_path: str,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
):
    """
    Visualize Sentinel-1 SAR image

    Shows:
    - VV band
    - VH band
    
    Args:
        image_path (str): Path to the Sentinel-1 GeoTIFF
        figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 5).
        save_path (Optional[str], optional): Optional path to save the visualization. Defaults to None.
    """

    print(f"\nVisualizing Sentinel-1 image: {image_path}")
    
    with rasterio.open(image_path) as src:
        # Read metadata
        print(f"  Bands (polarizations): {src.count}")
        print(f"  Size: {src.width} x {src.height}")
        print(f"  CRS: {src.crs}")
        print(f"  Bounds: {src.bounds}")
        
        # Read all bands (typically VV and VH for Sentinel-1)
        data = src.read()  # Shape: (bands, height, width)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'Sentinel-1 SAR Visualization\n{Path(image_path).name}', fontsize=14, fontweight='bold')
        
        # VV band
        if src.count >= 1:
            vv = data[0]
            # Convert to dB if not already (SAR data is often in linear scale)
            vv_db = 10 * np.log10(np.abs(vv) + 1e-10)
            vv_normalized = normalize_for_display(vv_db, percentile=(1, 99))
            
            axes[0].imshow(vv_normalized, cmap='gray')
            axes[0].set_title('VV Polarization', fontweight='bold')
            axes[0].axis('off')
            
            # Add stats as text
            axes[0].text(0.02, 0.98, f'Min: {vv_db.min():.1f} dB\nMax: {vv_db.max():.1f} dB\nMean: {vv_db.mean():.1f} dB',
                        transform=axes[0].transAxes, fontsize=8,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # VH band
        if src.count >= 2:
            vh = data[1]
            vh_db = 10 * np.log10(np.abs(vh) + 1e-10)
            vh_normalized = normalize_for_display(vh_db, percentile=(1, 99))
            
            axes[1].imshow(vh_normalized, cmap='gray')
            axes[1].set_title('VH Polarization', fontweight='bold')
            axes[1].axis('off')
            
            # Add stats as text
            axes[1].text(0.02, 0.98, f'Min: {vh_db.min():.1f} dB\nMax: {vh_db.max():.1f} dB\nMean: {vh_db.mean():.1f} dB',
                        transform=axes[1].transAxes, fontsize=8,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved visualization to: {save_path}")
        
        plt.show()


def visualize_downloaded_images(output_dir: str):
    """
    Automatically find and visualize all Sentinel images in a directory.

    Args:
        output_dir (str): Directory containing downloaded images
    """

    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Error: Directory not found: {output_dir}")
        return
    
    # Find all TIFF files in the directory
    tiff_files = list(output_path.glob("*.tif")) + list(output_path.glob("*.tiff"))
    
    if not tiff_files:
        print(f"No TIFF files found in: {output_dir}")
        return
    
    print(f"\nFound {len(tiff_files)} image(s) in {output_dir}")
    
    # Separate S2 and S1 files
    s2_files = [f for f in tiff_files if 'S2' in f.name.upper()]
    s1_files = [f for f in tiff_files if 'S1' in f.name.upper()]
    
    # Visualize Sentinel-2 images
    for s2_file in s2_files:
        visualize_sentinel2(
            str(s2_file),
            # save_path=str(output_path / f"{s2_file.stem}_visualization.png")
        )
    
    # Visualize Sentinel-1 images
    for s1_file in s1_files:
        visualize_sentinel1(
            str(s1_file),
            # save_path=str(output_path / f"{s1_file.stem}_visualization.png")
        )

def quick_view(s2_path: Optional[str] = None, s1_path: Optional[str] = None):
    """
    Quick function to view one or both image types.

    Args:
        s2_path (Optional[str], optional): Path to Sentinel-2 image. Defaults to None.
        s1_path (Optional[str], optional): Path to Sentinel-1 image. Defaults to None.
    """
    if s2_path:
        visualize_sentinel2(s2_path)
    
    if s1_path:
        visualize_sentinel1(s1_path)

if __name__ == "__main__":
    # Visualize all downloaded images in a directory
    # visualize_downloaded_images("/Users/samuel.omole/Desktop/repos/geofm_datasets/test_satellite_data/")
    
    # Visualize specific images with known names
    quick_view(
        s2_path="/Users/samuel.omole/Desktop/repos/geofm_datasets/test_satellite_data/S2L2A_20240626.tif",
        s1_path="/Users/samuel.omole/Desktop/repos/geofm_datasets/test_satellite_data/S1GRD_20240630.tif"
    )

    # Visualize individual TIFF images
    # visualize_sentinel2(
    #     "/Users/samuel.omole/Desktop/repos/geofm_datasets/test_satellite_data/S2L2A_20240626.tif",
    #     figsize=(18, 12),
    # )