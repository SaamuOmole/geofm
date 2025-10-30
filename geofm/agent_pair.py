from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from geofm.model_inference_pair import predict_pair

@tool
def predict_pair_tool(checkpoint_path, s2_path, s1_path, out_dir):
    """
    Predict and plot water segmentation of example
    pair of Sentinel-2 and Sentinel-1 images when
    provided with model checkpoint and images paths.
    Predicted image will be plotted alongside the
    ground-truth Sentinel-2 RGB and saved to out_dir
    as well an overlay of the predicted mask on the 
    Sentinel-2 RGB. 
    """
    return predict_pair(checkpoint_path, s2_path, s1_path, out_dir)
def main():
    model = ChatOllama(model="mistral:7b")
    model_with_tools = model.bind_tools([predict_pair_tool])

    checkpoint_path = "/Users/samuel.omole/Desktop/repos/geofm/output/terramind_small_sen1floods11/checkpoints/best-mIoU.ckpt"
    s2_path = "/Users/samuel.omole/Desktop/repos/geofm_datasets/sen1floods11_v1.1/data/S2L1CHand/Somalia_699062_S2Hand.tif"
    s1_path = "/Users/samuel.omole/Desktop/repos/geofm_datasets/sen1floods11_v1.1/data/S1GRDHand/Somalia_699062_S1Hand.tif"
    out_dir = "/Users/samuel.omole/Desktop/repos/geofm_datasets/prediction_pair"

    response = model_with_tools.invoke(
        f"Predict water segmentation of Sentinel-2 {s2_path} and Sentinel-1 {s1_path} \
            images using model checkpoint path {checkpoint_path} and save to {out_dir}"
    )
    # print(response.tool_calls)
    arguments = response.tool_calls[0]["args"]
    predict_pair_tool.invoke(arguments)

if __name__ == "__main__":
    main()