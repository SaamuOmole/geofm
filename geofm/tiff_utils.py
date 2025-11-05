import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple

def normalize_for_display(band: np.ndarray, percentile: Tuple[float, float] = (2, 98)) -> np.ndarray:
    """
    Normalize band data for display using percentile stretching.
    
    Args:
        band: Input band array
        percentile: Tuple of (min_percentile, max_percentile) for stretching
    
    Returns:
        Normalized array scaled to 0-1
    """
    # Remove any NaN or infinite values
    band_clean = band[np.isfinite(band)]
    
    if len(band_clean) == 0:
        return np.zeros_like(band)
    
    # Calculate percentiles
    p_min, p_max = np.percentile(band_clean, percentile)
    
    # Clip and normalize
    band_normalized = np.clip(band, p_min, p_max)
    band_normalized = (band_normalized - p_min) / (p_max - p_min + 1e-8)
    
    return band_normalized


def visualize_sentinel2(
    image_path: str,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
):
    """
    Visualize Sentinel-2 multi-band image.
    
    Shows:
    - True color RGB (bands 4, 3, 2)
    - False color infrared (bands 8, 4, 3)
    - Individual band samples
    
    Args:
        image_path: Path to the Sentinel-2 GeoTIFF
        figsize: Figure size
        save_path: Optional path to save the visualization
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
        
        # Band mapping for Sentinel-2 L2A
        # Typical order: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
        # For RGB we want: B04 (Red), B03 (Green), B02 (Blue)
        # These are typically at indices 3, 2, 1
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Sentinel-2 Visualization\n{Path(image_path).name}', fontsize=14, fontweight='bold')
        
        # True Color RGB (B04, B03, B02) - indices 3, 2, 1
        if src.count >= 4:
            try:
                rgb = np.stack([
                    normalize_for_display(data[3]),  # Red (B04)
                    normalize_for_display(data[2]),  # Green (B03)
                    normalize_for_display(data[1])   # Blue (B02)
                ], axis=-1)
                
                axes[0, 0].imshow(rgb)
                axes[0, 0].set_title('True Color RGB\n(B04-B03-B02)', fontweight='bold')
                axes[0, 0].axis('off')
            except Exception as e:
                axes[0, 0].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
                axes[0, 0].axis('off')
        
        # False Color Infrared (B08, B04, B03) - indices 7, 3, 2
        if src.count >= 8:
            try:
                false_color = np.stack([
                    normalize_for_display(data[7]),  # NIR (B08)
                    normalize_for_display(data[3]),  # Red (B04)
                    normalize_for_display(data[2])   # Green (B03)
                ], axis=-1)
                
                axes[0, 1].imshow(false_color)
                axes[0, 1].set_title('False Color (NIR)\n(B08-B04-B03)', fontweight='bold')
                axes[0, 1].axis('off')
            except Exception as e:
                axes[0, 1].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
                axes[0, 1].axis('off')
        
        # NDVI visualization (if NIR and Red are available)
        if src.count >= 8:
            try:
                nir = data[7].astype(float)  # B08
                red = data[3].astype(float)  # B04
                
                # Calculate NDVI: (NIR - Red) / (NIR + Red)
                ndvi = (nir - red) / (nir + red + 1e-8)
                
                im = axes[0, 2].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
                axes[0, 2].set_title('NDVI\n(Vegetation Index)', fontweight='bold')
                axes[0, 2].axis('off')
                plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
            except Exception as e:
                axes[0, 2].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
                axes[0, 2].axis('off')
        
        # Individual bands
        band_indices = [1, 3, 7] if src.count >= 8 else [0, 1, 2]
        band_names = ['B02 (Blue)', 'B04 (Red)', 'B08 (NIR)'] if src.count >= 8 else [f'Band {i+1}' for i in band_indices]
        
        for idx, (band_idx, band_name) in enumerate(zip(band_indices, band_names)):
            if band_idx < src.count:
                try:
                    band_data = normalize_for_display(data[band_idx])
                    axes[1, idx].imshow(band_data, cmap='gray')
                    axes[1, idx].set_title(band_name, fontweight='bold')
                    axes[1, idx].axis('off')
                except Exception as e:
                    axes[1, idx].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
                    axes[1, idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved visualization to: {save_path}")
        
        plt.show()


def visualize_sentinel1(
    image_path: str,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
):
    """
    Visualize Sentinel-1 SAR image.
    
    Shows:
    - VV polarization
    - VH polarization
    - VV/VH ratio
    
    Args:
        image_path: Path to the Sentinel-1 GeoTIFF
        figsize: Figure size
        save_path: Optional path to save the visualization
    """
    print(f"\nVisualizing Sentinel-1 image: {image_path}")
    
    with rasterio.open(image_path) as src:
        # Read metadata
        print(f"  Bands (polarizations): {src.count}")
        print(f"  Size: {src.width} x {src.height}")
        print(f"  CRS: {src.crs}")
        print(f"  Bounds: {src.bounds}")
        
        # Read all bands (typically VV and VH)
        data = src.read()  # Shape: (bands, height, width)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(f'Sentinel-1 SAR Visualization\n{Path(image_path).name}', fontsize=14, fontweight='bold')
        
        # VV polarization (band 1)
        if src.count >= 1:
            try:
                vv = data[0]
                # Convert to dB if not already (SAR data is often in linear scale)
                vv_db = 10 * np.log10(np.abs(vv) + 1e-10)
                vv_normalized = normalize_for_display(vv_db, percentile=(1, 99))
                
                axes[0].imshow(vv_normalized, cmap='gray')
                axes[0].set_title('VV Polarization', fontweight='bold')
                axes[0].axis('off')
                
                # Add statistics
                axes[0].text(0.02, 0.98, f'Min: {vv_db.min():.1f} dB\nMax: {vv_db.max():.1f} dB\nMean: {vv_db.mean():.1f} dB',
                           transform=axes[0].transAxes, fontsize=8,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except Exception as e:
                axes[0].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
                axes[0].axis('off')
        
        # VH polarization (band 2)
        if src.count >= 2:
            try:
                vh = data[1]
                vh_db = 10 * np.log10(np.abs(vh) + 1e-10)
                vh_normalized = normalize_for_display(vh_db, percentile=(1, 99))
                
                axes[1].imshow(vh_normalized, cmap='gray')
                axes[1].set_title('VH Polarization', fontweight='bold')
                axes[1].axis('off')
                
                # Add statistics
                axes[1].text(0.02, 0.98, f'Min: {vh_db.min():.1f} dB\nMax: {vh_db.max():.1f} dB\nMean: {vh_db.mean():.1f} dB',
                           transform=axes[1].transAxes, fontsize=8,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except Exception as e:
                axes[1].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
                axes[1].axis('off')
        
        # VV/VH ratio (useful for classification)
        if src.count >= 2:
            try:
                vv = data[0]
                vh = data[1]
                
                # Calculate ratio in dB space
                ratio = vv / (vh + 1e-10)
                ratio_db = 10 * np.log10(np.abs(ratio) + 1e-10)
                ratio_normalized = normalize_for_display(ratio_db, percentile=(1, 99))
                
                im = axes[2].imshow(ratio_normalized, cmap='RdYlBu_r')
                axes[2].set_title('VV/VH Ratio', fontweight='bold')
                axes[2].axis('off')
                plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
                
                # Add statistics
                axes[2].text(0.02, 0.98, f'Min: {ratio_db.min():.1f} dB\nMax: {ratio_db.max():.1f} dB\nMean: {ratio_db.mean():.1f} dB',
                           transform=axes[2].transAxes, fontsize=8,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except Exception as e:
                axes[2].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
                axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved visualization to: {save_path}")
        
        plt.show()


def visualize_downloaded_images(output_dir: str):
    """
    Automatically find and visualize all Sentinel images in a directory.
    
    Args:
        output_dir: Directory containing downloaded images
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Error: Directory not found: {output_dir}")
        return
    
    # Find all TIFF files
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
            save_path=str(output_path / f"{s2_file.stem}_visualization.png")
        )
    
    # Visualize Sentinel-1 images
    for s1_file in s1_files:
        visualize_sentinel1(
            str(s1_file),
            save_path=str(output_path / f"{s1_file.stem}_visualization.png")
        )


# ============================================================================
# Quick visualization function
# ============================================================================

def quick_view(s2_path: Optional[str] = None, s1_path: Optional[str] = None):
    """
    Quick function to view one or both image types.
    
    Args:
        s2_path: Path to Sentinel-2 image
        s1_path: Path to Sentinel-1 image
    """
    if s2_path:
        visualize_sentinel2(s2_path)
    
    if s1_path:
        visualize_sentinel1(s1_path)


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    # Option 1: Visualize all images in a directory
    # visualize_downloaded_images("./test_satellite_data")
    
    # Option 2: Visualize specific images
    quick_view(
        s2_path="/Users/samuel.omole/Desktop/repos/geofm_datasets/test_satellite_data/S2L2A_20240626.tif",
        s1_path="/Users/samuel.omole/Desktop/repos/geofm_datasets/test_satellite_data/S1GRD_20240630.tif"
    )
    
    # Option 3: Visualize individual images with more control
    # visualize_sentinel2(
    #     "./test_satellite_data/S2L2A_20240615.tif",
    #     figsize=(18, 12),
    #     save_path="./my_s2_visualization.png"
    # )