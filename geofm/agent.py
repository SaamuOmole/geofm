from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from geofm.model_inference import predict_image_batches

@tool
def predict_image_batches_tool(checkpoint_path, dataset_path):
    """
    Predict and plot water segmentation
    of example batch image using the model
    checkpoint and dataset paths. 
    """
    return predict_image_batches(checkpoint_path, dataset_path)
def main():
    model = ChatOllama(model="mistral:7b")
    model_with_tools = model.bind_tools([predict_image_batches_tool])

    checkpoint_path = "/Users/samuel.omole/Desktop/repos/geofm/output/terramind_small_sen1floods11/checkpoints/best-mIoU.ckpt"
    dataset_path = "/Users/samuel.omole/Desktop/repos/geofm_datasets/sen1floods11_v1.1"

    response = model_with_tools.invoke(
        f"Predict and plot an example batch image from model checkpoint path {checkpoint_path} and dataset path {dataset_path}"
    )

    # print(response.tool_calls)
    arguments = response.tool_calls[0]["args"]
    predict_image_batches_tool.invoke(arguments)

if __name__ == "__main__":
    main()