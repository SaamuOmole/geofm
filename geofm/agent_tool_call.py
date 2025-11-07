from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from geofm.model_inference_pair import predict_pair
from geofm.get_sentinel_images import find_and_download_sentinel_images

# This is a more complicated example of expanding the agent_pair.py tool call.
# Here, 2 tools namely, get_satellite_data and predict_pair_tool are provided
# to the language model. The get_satellite_data uses the Microsoft Planetary
# Computer STAC API to download the Sentinel-2 and Sentinel-1 images when a
# location and date parameters are provided as part of a prompt.
# The predict_pair_tool then grabs these image pair to make predictions using
# a known model saved at an endpoint. 

@tool
def get_satellite_data(location: str,
                       start_date: str,
                       end_date: str,
                       output_dir: str,
                       max_cloud_cover: float,
                       tile_size: int,
                       ):
    """
    Download Sentinel-2 and Sentinel-1 satellite images for a given
    location and time period from an API.

    Args:
        location (str): Bounding box as "min_lon,min_lat,max_lon,max_lat"
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        output_dir (str): Directory to save downloaded images
        max_cloud_cover (float): Maximum cloud cover percentage (0-100)
        tile_size (int): Size of output tiles in pixels

    Returns:
        Dictionary with paths to downloaded images
    """
    return find_and_download_sentinel_images(location, start_date, end_date, output_dir, max_cloud_cover, tile_size)

def find_sentinel_tiff_filepath(images_dir: str):
    """
    Find Sentinel-2 and Sentinel-1 TIFF files in a directory.

    Args:
        images_dir (str): Directory where S2 and S1 images are located

    Returns:
        Separated S2 and S1 file paths
    """
    images_path = Path(images_dir)
    # Find all TIFF files
    tiff_files = list(images_path.glob("*.tif")) + list(images_path.glob("*.tiff"))
    if not tiff_files:
        print(f"No TIFF files found in: {images_path}")
        return
    print(f"\nFound {len(tiff_files)} image(s) in {images_path}")

    # Separate S2 and S1 files
    s2_files = [f for f in tiff_files if 'S2' in f.name.upper()]
    s1_files = [f for f in tiff_files if 'S1' in f.name.upper()]
    
    # Get S2 and S1 paths (we are sure there are only 2 pairs of S2 & S1 images in the images_dir)
    s2_path = str(s2_files[0])
    s1_path = str(s1_files[0])
    
    print(f"S2 path: {s2_path}")
    print(f"S1 path: {s1_path}")
    
    return s2_path, s1_path
    
@tool
def predict_pair_tool(checkpoint_path: str, sentinel_images_dir: str, out_dir: str):
    """
    Predict and plot water segmentation from example
    pair of downloaded Sentinel-2 and Sentinel-1 images
    when provided with model checkpoint and images paths.
    Predicted image will be plotted alongside the
    ground-truth Sentinel-2 RGB and saved to out_dir
    as well an overlay of the predicted mask on the 
    Sentinel-2 RGB.

    Args:
        checkpoint_path (str): Path to the model checkpoint file
        images_path (str): Path to Sentinel-2 & Sentinel-1 TIFF images
        out_dir (str): Output directory to save predictions

    Returns:
        Predicted images saved to out_dir
    """
    # Unpack to get S2 and S1 file paths
    s2_path, s1_path = find_sentinel_tiff_filepath(sentinel_images_dir)

    if s2_path is None or s1_path is None:
        return {"Error": "Could not find S2 or S1 images in directory"}

    return predict_pair(checkpoint_path, s2_path, s1_path, out_dir)

def main():
    """
    The main call to the language model which runs
    the model as well as provided tools to it. The
    get_satellite_data and predict_pair_tool are
    provided to the language model as tools it can
    access
    """
    # Define language model and provide tools to it
    model = ChatOllama(model="mistral:7b")
    model_with_tools = model.bind_tools([get_satellite_data, predict_pair_tool])

    # Specify paths for tool 1
    # location="-0.1,51.5,-0.05,51.52"  # For ~London
    location="-2.85,53.20,-2.25,53.65" # For ~Warrington area of UK
    start_date="2024-01-01" # "2024-06-01"
    end_date="2025-11-06" # "2024-06-30"
    output_dir="/Users/samuel.omole/Desktop/repos/geofm_datasets/test_satellite_data"
    max_cloud_cover=10.0 # 30.0
    tile_size=512
    
    # Specify paths for tool 2
    checkpoint_path = "/Users/samuel.omole/Desktop/repos/geofm/output/terramind_small_sen1floods11/checkpoints/best-mIoU.ckpt"
    sentinel_images_dir = output_dir
    out_dir = "/Users/samuel.omole/Desktop/repos/geofm_datasets/prediction_pair"
    
    # Validate checkpoint path exists
    if not Path(checkpoint_path).exists():
        print(f"Error: checkpoint path does not exist: {checkpoint_path}")
        return
    
    
    # Define the prompt
    prompt = f"""You have two tasks to complete in sequence:

    Task 1: Download satellite images using get_satellite_data tool with these parameters:
    - location: "{location}"
    - start_date: "{start_date}"
    - end_date: "{end_date}"
    - output_dir: "{output_dir}"
    - max_cloud_cover: {max_cloud_cover}
    - tile_size: {tile_size}

    Task 2: After downloading, predict water segmentation using predict_pair_tool with these parameters:
    - checkpoint_path: "{checkpoint_path}"
    - sentinel_images_dir: "{sentinel_images_dir}"
    - out_dir: "{out_dir}"

    Please call both tools in sequence."""

    print("="*60)
    print("Sending prompt to model...")
    print("="*60)

    # Get response from model
    response = model_with_tools.invoke(prompt)
    
    print(f"\nModel response content: {response.content if hasattr(response, 'content') else 'N/A'}")
    
    # Get the tool calls
    tool_calls = getattr(response, "tool_calls", None)

    # If no tool calls i.e., model did not call the tool for any reason, fallback to calling the tool directly
    if not tool_calls:
        print("\n" + "="*60)
        print("\nNo tool calls found in response — calling predict_pair_tool directly as fallback.")
        print("="*60)
        
        # Tool 1: Download satellite data
        print("\n[1/2] Downloading satellite images...")
        download_result = get_satellite_data.invoke({
            "location": location,
            "start_date": start_date,
            "end_date": end_date,
            "output_dir": output_dir,
            "max_cloud_cover": max_cloud_cover,
            "tile_size": tile_size
        })
        print(f"Download result: {download_result}")
        # Check if download was successful
        if not download_result.get('sentinel2_paths') or not download_result.get('sentinel1_paths'):
            print("\nError: Failed to download images. Cannot proceed with prediction.")
            return
        
        # Tool 2: Make prediction
        print("\n[2/2] Making prediction...")
        prediction_result = predict_pair_tool.invoke({
            "checkpoint_path": checkpoint_path,
            "sentinel_images_dir": sentinel_images_dir,
            "out_dir": out_dir
        })
        print(f"Prediction result: {prediction_result}")
        
        print("\n" + "="*60)
        print("Both tasks completed!")
        print("="*60)
        return
    
    # Process all tool calls
    print(f"\n" + "="*60)
    print(f"Found {len(tool_calls)} tool call(s)")
    print("="*60)
    
    download_completed = False
    
    # Loop through tool calls
    for i, call in enumerate(tool_calls):
        tool_name = call.get('name') if isinstance(call, dict) else None
        args = call.get('args') if isinstance(call, dict) else None
        
        print(f"\n[{i+1}/{len(tool_calls)}] Processing tool: {tool_name}")
        
        if not args:
            print("  No args found, skipping...")
            continue
        
        print(f"  Args: {args}")
        
        # Execute the appropriate tool
        if tool_name == 'get_satellite_data':
            print("  Downloading satellite images...")
            result = get_satellite_data.invoke(args)
            print(f"  Result: {result}")
            
            # Check if download was successful
            if result.get('sentinel2_paths') and result.get('sentinel1_paths'):
                download_completed = True
                print("  Download successful!")
            else:
                print("  Download failed!")
                
        elif tool_name == 'predict_pair_tool':
            if not download_completed:
                print("  Warning: Attempting prediction without confirmed download")
            
            print("  Making prediction...")
            result = predict_pair_tool.invoke(args)
            print(f"  Result: {result}")
            print("  Prediction completed!")
        else:
            print(f"  Unknown tool: {tool_name}")
    
    print("\n" + "="*60)
    print("All tasks completed!")
    print("="*60)
    print(f"Satellite images saved to: {output_dir}")
    print(f"Predictions saved to: {out_dir}")

if __name__ == "__main__":
    main()