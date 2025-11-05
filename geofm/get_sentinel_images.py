from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import os
from datetime import datetime, timedelta
from langchain_core.tools import tool
import pystac_client
import planetary_computer
import rasterio
from rasterio.windows import Window
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import shutil

def search_sentinel_data(
    bbox: List[float],  # [min_lon, min_lat, max_lon, max_lat]
    start_date: str,
    end_date: str,
    collections: List[str],
    max_cloud_cover: float = 20.0
) -> List[pystac_client.item_search.Item]:
    """
    Search for Sentinel data using Microsoft Planetary Computer STAC API.
    
    Args:
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD'
        collections: List of collection names (e.g., ['sentinel-2-l2a', 'sentinel-1-grd'])
        max_cloud_cover: Maximum cloud cover percentage (for Sentinel-2)
    
    Returns:
        List of STAC items matching the search criteria
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    # Build query
    query_params = {}
    if 'sentinel-2-l1c' in collections or 'sentinel-2-l2a' in collections:
        query_params["eo:cloud_cover"] = {"lt": max_cloud_cover}
    
    search = catalog.search(
        collections=collections,
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query=query_params if query_params else None,
    )
    
    items = list(search.items())
    print(f"Found {len(items)} items matching search criteria")
    return items


# def download_and_crop_image(
#     item: pystac_client.item_search.Item,
#     asset_keys: List[str],
#     output_path: str,
#     tile_size: Tuple[int, int] = (512, 512),
#     center_crop: bool = True
# ) -> Optional[str]:
#     """
#     Download image from STAC item and crop to specified size.
    
#     Args:
#         item: STAC item containing the image
#         asset_keys: List of asset keys to download (e.g., ['B02', 'B03', 'B04'] for RGB)
#         output_path: Path to save the cropped image
#         tile_size: Size of the output tile (width, height)
#         center_crop: If True, crop from center; otherwise crop from top-left
    
#     Returns:
#         Path to saved image or None if failed
#     """
#     try:
#         # Get the first available asset
#         asset_key = None
#         for key in asset_keys:
#             if key in item.assets:
#                 asset_key = key
#                 break
        
#         if not asset_key:
#             print(f"None of the requested assets {asset_keys} found in item")
#             return None
        
#         asset = item.assets[asset_key]
#         href = asset.href
        
#         # Sign the URL if needed (for Planetary Computer)
#         if hasattr(planetary_computer, 'sign'):
#             href = planetary_computer.sign(href)
        
#         print(f"Downloading from: {asset_key}")
        
#         # Open the raster with rasterio
#         with rasterio.open(href) as src:
#             # Get image dimensions
#             height, width = src.height, src.width
            
#             # Calculate crop window
#             if center_crop:
#                 col_off = max(0, (width - tile_size[0]) // 2)
#                 row_off = max(0, (height - tile_size[1]) // 2)
#             else:
#                 col_off = 0
#                 row_off = 0
            
#             # Ensure we don't exceed image bounds
#             actual_width = min(tile_size[0], width - col_off)
#             actual_height = min(tile_size[1], height - row_off)
            
#             window = Window(col_off, row_off, actual_width, actual_height)
            
#             # Read the data
#             data = src.read(window=window)
            
#             # Save as GeoTIFF with same profile
#             profile = src.profile.copy()
#             profile.update({
#                 'height': actual_height,
#                 'width': actual_width,
#                 'transform': rasterio.windows.transform(window, src.transform)
#             })
            
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
#             with rasterio.open(output_path, 'w', **profile) as dst:
#                 dst.write(data)
            
#             print(f"Saved cropped image to: {output_path}")
#             return output_path
            
#     except Exception as e:
#         print(f"Error downloading/cropping image: {e}")
#         return None


def download_multiband_sentinel2(
    item: pystac_client.item_search.Item,
    output_path: str,
    bands: Optional[List[str]] = None,
    tile_size: Tuple[int, int] = (512, 512)
) -> Optional[str]:
    """
    Download multi-band Sentinel-2 image.
    
    Args:
        item: STAC item for Sentinel-2
        output_path: Path to save the image
        bands: List of band names (e.g., ['B02', 'B03', 'B04', 'B08']). If None, downloads all bands.
        tile_size: Size of output tile
    
    Returns:
        Path to saved image or None if failed
    """
    try:
        # Default to all visible and NIR bands if not specified
        if bands is None:
            bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
        
        available_bands = [b for b in bands if b in item.assets]
        
        if not available_bands:
            print(f"No bands available from requested: {bands}")
            return None
        
        print(f"Downloading {len(available_bands)} bands: {available_bands}")
        
        # Read first band to get dimensions and profile
        first_asset = item.assets[available_bands[0]]
        first_href = planetary_computer.sign(first_asset.href)
        
        with rasterio.open(first_href) as src:
            height, width = src.height, src.width
            
            # Calculate center crop window
            col_off = max(0, (width - tile_size[0]) // 2)
            row_off = max(0, (height - tile_size[1]) // 2)
            actual_width = min(tile_size[0], width - col_off)
            actual_height = min(tile_size[1], height - row_off)
            
            window = Window(col_off, row_off, actual_width, actual_height)
            
            # Prepare output array
            band_data = []
            
            # Read each band
            for band in available_bands:
                asset = item.assets[band]
                href = planetary_computer.sign(asset.href)
                
                with rasterio.open(href) as band_src:
                    data = band_src.read(1, window=window)
                    band_data.append(data)
            
            # Stack bands
            stacked = np.stack(band_data, axis=0)
            
            # Create output profile
            profile = src.profile.copy()
            profile.update({
                'count': len(available_bands),
                'height': actual_height,
                'width': actual_width,
                'transform': rasterio.windows.transform(window, src.transform)
            })
            
            # Save
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(stacked)
            
            print(f"Saved {len(available_bands)}-band image to: {output_path}")
            return output_path
            
    except Exception as e:
        print(f"Error downloading Sentinel-2 multi-band image: {e}")
        return None


def download_sentinel1(
    item: pystac_client.item_search.Item,
    output_path: str,
    polarizations: Optional[List[str]] = None,
    tile_size: Tuple[int, int] = (512, 512)
) -> Optional[str]:
    """
    Download Sentinel-1 SAR image.
    
    Args:
        item: STAC item for Sentinel-1
        output_path: Path to save the image
        polarizations: List of polarizations (e.g., ['vv', 'vh']). If None, downloads all available.
        tile_size: Size of output tile
    
    Returns:
        Path to saved image or None if failed
    """
    try:
        # Default polarizations
        if polarizations is None:
            polarizations = ['vv', 'vh']
        
        available_pols = [p for p in polarizations if p in item.assets]
        
        if not available_pols:
            print(f"No polarizations available from requested: {polarizations}")
            return None
        
        print(f"Downloading {len(available_pols)} polarizations: {available_pols}")
        
        # Read first polarization to get dimensions
        first_asset = item.assets[available_pols[0]]
        first_href = planetary_computer.sign(first_asset.href)
        
        with rasterio.open(first_href) as src:
            height, width = src.height, src.width
            
            # Calculate center crop window
            col_off = max(0, (width - tile_size[0]) // 2)
            row_off = max(0, (height - tile_size[1]) // 2)
            actual_width = min(tile_size[0], width - col_off)
            actual_height = min(tile_size[1], height - row_off)
            
            window = Window(col_off, row_off, actual_width, actual_height)
            
            # Read each polarization
            pol_data = []
            for pol in available_pols:
                asset = item.assets[pol]
                href = planetary_computer.sign(asset.href)
                
                with rasterio.open(href) as pol_src:
                    data = pol_src.read(1, window=window)
                    pol_data.append(data)
            
            # Stack polarizations
            stacked = np.stack(pol_data, axis=0)
            
            # Create output profile
            profile = src.profile.copy()
            profile.update({
                'count': len(available_pols),
                'height': actual_height,
                'width': actual_width,
                'transform': rasterio.windows.transform(window, src.transform)
            })
            
            # Save
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(stacked)
            
            print(f"Saved Sentinel-1 image with {len(available_pols)} polarizations to: {output_path}")
            return output_path
            
    except Exception as e:
        print(f"Error downloading Sentinel-1 image: {e}")
        return None

def find_and_download_sentinel_images(
    location: str,
    start_date: str,
    end_date: Optional[str],
    output_dir: str,
    max_cloud_cover: float,
    tile_size: int,
) -> Dict[str, Any]:
    """
    Find and download Sentinel-2 (S2L1C/L2A) and Sentinel-1 (S1GRD) images from Microsoft Planetary Computer.
    
    Args:
        location: Location as string (city name) or bounding box as "min_lon,min_lat,max_lon,max_lat"
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD' (optional, defaults to start_date + 7 days)
        output_dir: Directory to save downloaded images
        max_cloud_cover: Maximum cloud cover percentage for Sentinel-2 (0-100)
        tile_size: Size of the output tiles (will be tile_size x tile_size)
    
    Returns:
        Dictionary containing paths to downloaded Sentinel-2 and Sentinel-1 images
    """
    try:
        # Parse location to bbox
        # Possible to use a proper geocoding API to extract bbox for location
        if ',' in location and location.count(',') == 3:
            # Already a bbox
            bbox = [float(x.strip()) for x in location.split(',')]
        else:
            # For now, provide some example coordinates
            print(f"Location name provided: {location}")
            print("Please provide bbox as 'min_lon,min_lat,max_lon,max_lat' for more accurate results")
            
            # Defaults to ~London
            bbox = [-0.15, 51.48, -0.05, 51.53]
            print(f"Using default bbox: {bbox}")
        
        # Set end_date if not provided
        if end_date is None:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = start_dt + timedelta(days=7)
            end_date = end_dt.strftime('%Y-%m-%d')
        
        print("\nSearching for Sentinel data:")
        print(f"  Location (bbox): {bbox}")
        print(f"  Date range: {start_date} to {end_date}")
        print(f"  Max cloud cover: {max_cloud_cover}%")
        print(f"  Tile size: {tile_size}x{tile_size}")
        
        # Create output directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            "sentinel2_paths": [],
            "sentinel1_paths": [],
            "metadata": []
        }
        
        # Search for Sentinel-2 L2A (atmospherically corrected)
        print("\n" + "="*60)
        print("Searching for Sentinel-2 L2A data...")
        print("="*60)
        
        s2_items = search_sentinel_data(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            collections=['sentinel-2-l2a'],
            max_cloud_cover=max_cloud_cover
        )
        
        if s2_items:
            # Download the most recent item
            item = s2_items[0]
            print(f"\nDownloading Sentinel-2 from: {item.datetime}")
            print(f"Cloud cover: {item.properties.get('eo:cloud_cover', 'N/A')}%")
            
            # Define bands to download (all 13 bands for S2L2A)
            s2_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
            
            s2_filename = f"S2L2A_{item.datetime.strftime('%Y%m%d')}.tif"
            s2_path = os.path.join(output_dir, s2_filename)
            
            downloaded_s2 = download_multiband_sentinel2(
                item=item,
                output_path=s2_path,
                bands=s2_bands,
                tile_size=(tile_size, tile_size)
            )
            
            if downloaded_s2:
                results["sentinel2_paths"].append(downloaded_s2)
                results["metadata"].append({
                    "type": "Sentinel-2 L2A",
                    "date": item.datetime.isoformat(),
                    "cloud_cover": item.properties.get('eo:cloud_cover'),
                    "path": downloaded_s2
                })
        
        # Search for Sentinel-1 GRD
        print("\n" + "="*60)
        print("Searching for Sentinel-1 GRD data...")
        print("="*60)
        
        s1_items = search_sentinel_data(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            collections=['sentinel-1-grd'],
            max_cloud_cover=100  # Not applicable for SAR
        )
        
        if s1_items:
            # Download the most recent item
            item = s1_items[0]
            print(f"\nDownloading Sentinel-1 from: {item.datetime}")
            
            s1_filename = f"S1GRD_{item.datetime.strftime('%Y%m%d')}.tif"
            s1_path = os.path.join(output_dir, s1_filename)
            
            downloaded_s1 = download_sentinel1(
                item=item,
                output_path=s1_path,
                polarizations=['vv', 'vh'],
                tile_size=(tile_size, tile_size)
            )
            
            if downloaded_s1:
                results["sentinel1_paths"].append(downloaded_s1)
                results["metadata"].append({
                    "type": "Sentinel-1 GRD",
                    "date": item.datetime.isoformat(),
                    "path": downloaded_s1
                })
        
        print("\n" + "="*60)
        print("Download Summary")
        print("="*60)
        print(f"Sentinel-2 images: {len(results['sentinel2_paths'])}")
        print(f"Sentinel-1 images: {len(results['sentinel1_paths'])}")
        
        return results
        
    except Exception as e:
        print(f"Error in find_and_download_sentinel_images: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "sentinel2_paths": [],
            "sentinel1_paths": []
        }

if __name__ == "__main__":
    result = find_and_download_sentinel_images(
        location="-0.1,51.5,-0.05,51.52",  # for ~London
        start_date="2024-06-01",
        end_date="2024-06-30",
        output_dir="/Users/samuel.omole/Desktop/repos/geofm_datasets/test_satellite_data",
        max_cloud_cover=30.0,
        tile_size=512
    )
    
    print("\n" + "="*60)
    print("Final Results:")
    print("="*60)
    print(f"S2 images: {result['sentinel2_paths']}")
    print(f"S1 images: {result['sentinel1_paths']}")