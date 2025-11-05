from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from geofm.model_inference_pair import predict_pair

@tool
def predict_pair_tool(checkpoint_path: str, s2_path: str, s1_path: str, out_dir: str):
    """
    Predict and plot water segmentation of example
    pair of Sentinel-2 and Sentinel-1 images when
    provided with model checkpoint and images paths.
    Predicted image will be plotted alongside the
    ground-truth Sentinel-2 RGB and saved to out_dir
    as well an overlay of the predicted mask on the 
    Sentinel-2 RGB.

    Args:
        checkpoint_path (str): Path ti the model checkpoint file
        s2_path (str): Path to Sentinel-2 L1C TIFF image
        s1_path (str): Path to Sentinel-1 GRD TIFF image
        out_dir (str): Output directory to save predictions

    Returns:
        Predicted images saved to out_dir
    """
    return predict_pair(checkpoint_path, s2_path, s1_path, out_dir)
def main():
    # define model and provide tool(s) to it
    model = ChatOllama(model="mistral:7b")
    model_with_tools = model.bind_tools([predict_pair_tool])

    # define paths
    checkpoint_path = "/Users/samuel.omole/Desktop/repos/geofm/output/terramind_small_sen1floods11/checkpoints/best-mIoU.ckpt"
    s2_path = "/Users/samuel.omole/Desktop/repos/geofm_datasets/sen1floods11_v1.1/data/S2L1CHand/Bolivia_76104_S2Hand.tif"
    s1_path = "/Users/samuel.omole/Desktop/repos/geofm_datasets/sen1floods11_v1.1/data/S1GRDHand/Bolivia_76104_S1Hand.tif"
    out_dir = "/Users/samuel.omole/Desktop/repos/geofm_datasets/prediction_pair"
    
    # Validate paths exist
    for path, name in [(checkpoint_path, "checkpoint"), (s2_path, "S2"), (s1_path, "S1")]:
        if not Path(path).exists():
            print(f"Error: {name} path does not exist: {path}")
            return
    
    # define the prompt
    prompt = f"""Predict water segmentation using the following:
    - Sentinel-2 image: {s2_path}
    - Sentinel-1 image: {s1_path}
    - Model checkpoint: {checkpoint_path}
    - Save results to: {out_dir}
    
    Use the predict_pair_tool to perform this prediction."""

    # prompt =f"Predict water segmentation of Sentinel-2 {s2_path} and Sentinel-1 {s1_path} \
    #     images using model checkpoint path {checkpoint_path} and save to {out_dir}"
    print("Sending prompt to model...")
    # print the response
    response = model_with_tools.invoke(prompt)
    try: 
        print("response:", response)
    except Exception as e:
        print("Error printing response:", e)
    
    # get the tool calls
    tool_calls = getattr(response, "tool_calls", None)

    # if no tool calls i.e., model did not call the tool for any reason, fallback to calling the tool directly
    if not tool_calls:
        print("\nNo tool calls found in response — calling predict_pair_tool directly as fallback.")
        result = predict_pair_tool.invoke({
            "checkpoint_path": checkpoint_path,
            "s2_path": s2_path,
            "s1_path": s1_path,
            "out_dir": out_dir,
        })
        print("\nPrediction completed!")
        # predict_pair_tool(checkpoint_path=checkpoint_path,
        #                   s2_path=s2_path,
        #                   s1_path=s1_path,
        #                   out_dir=out_dir,
        # )
        return
    # if tool_calls exist, extract the args
    call = tool_calls[0]
    args = call.get("args") if isinstance(call, dict) else None
    
    # if no args within tool call, fallback to calling the tool
    if not args:
        print("Tool call present but could not parse args; calling tool directly as fallback.")
        result = predict_pair_tool.invoke({
                "checkpoint_path": checkpoint_path,
                "s2_path": s2_path,
                "s1_path": s1_path,
                "out_dir": out_dir,
            })
        # predict_pair_tool(checkpoint_path=checkpoint_path,
        #                   s2_path=s2_path,
        #                   s1_path=s1_path,
        #                   out_dir=out_dir,
        # )
        return
    else:
        print(f"\nArgs: {args}")
        predict_pair_tool.invoke(args)
        print("\nPrediction completed!")

if __name__ == "__main__":
    main()