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
import traceback

# Script to get Sentinel 2 and Sentinel 1 images from the Microsoft Planetary Computer STAC API
# Only Sentinel-2-L2A imnages, with 12 bands, are provided by this API. To get the Sentinel-2-L1C
# images in all 13 bands, the Google Earth Engine or Copernicus API may be used. However, this
# script has extracted the Sentinel-2-L2A images and added an extra dummy band for test purposes. 

def search_sentinel_data(
    bbox: List[float],
    start_date: str,
    end_date: str,
    collections: List[str],
    max_cloud_cover: float = 20.0
) -> List[pystac_client.item_search.Item]:
    """
    Search for Sentinel data using Microsoft Planetary Computer STAC API.

    Args:
        bbox (List[float]): Bounding box [min_lon, min_lat, max_lon, max_lat]
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        collections (List[str]): List of collection names (e.g., ['sentinel-2-l2a', 'sentinel-1-grd'])
        max_cloud_cover (float, optional): Maximum cloud cover percentage (for Sentinel-2). Defaults to 20.0.

    Returns:
        List[pystac_client.item_search.Item]: List of STAC items matching the search criteria
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
    
def download_multiband_sentinel2(
    item: pystac_client.item_search.Item,
    output_path: str,
    bands: Optional[List[str]] = None,
    tile_size: Tuple[int, int] = (512, 512)
) -> Optional[str]:
    """
    Download multi-band Sentinel-2 image.

    Args:
        item (pystac_client.item_search.Item): STAC item for Sentinel-2
        output_path (str): Path to save the image
        bands (Optional[List[str]], optional): List of band names. If None, downloads all bands. Defaults to None.
        tile_size (Tuple[int, int], optional): Size of output tile. Defaults to (512, 512).

    Returns:
        Optional[str]: Path to saved image or None if failed
    """
    try:
        # Default to all visible and NIR bands if not specified
        if bands is None:
            bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        
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
            
            # Prepare output array to add to
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
        item (pystac_client.item_search.Item): STAC item for Sentinel-1
        output_path (str): Path to save the image
        polarizations (Optional[List[str]], optional): List of polarizations (e.g., ['vv', 'vh']). If None, downloads all available. Defaults to None.
        tile_size (Tuple[int, int], optional): Size of output tile. Defaults to (512, 512).

    Returns:
        Optional[str]: Path to saved image or None if failed
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
    Find and download Sentinel-2 (S2L2A) and Sentinel-1 (S1GRD) images from Microsoft Planetary Computer.

    Args:
        location (str): Location as bounding box in format "min_lon,min_lat,max_lon,max_lat"
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (Optional[str]): End date in format 'YYYY-MM-DD' (optional, defaults to start_date + 7 days)
        output_dir (str): Directory to save downloaded images
        max_cloud_cover (float): Maximum cloud cover percentage for Sentinel-2 (0-100)
        tile_size (int): Size of the output tiles (will be tile_size x tile_size)

    Returns:
        Dict[str, Any]: Dictionary containing paths to downloaded Sentinel-2 and Sentinel-1 images
    """
    try:
        # Parse location to bbox
        if ',' in location and location.count(',') == 3:
            # Already a bbox
            bbox = [float(x.strip()) for x in location.split(',')]
        else:
            # If location not properly given, then uses a default
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
        
        # Search for Sentinel-2 L2A
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
            
            # Define bands to download
            s2_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
            
            s2_filename = f"S2L2A_{item.datetime.strftime('%Y%m%d')}.tif"
            s2_path = os.path.join(output_dir, s2_filename)
            
            # Download the S2 bands 
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