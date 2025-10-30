import os
import torch
import terratorch
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import shutil
import warnings
warnings.filterwarnings("ignore")

# define model checkpoint and dataset paths
# checkpoint_path = "/Users/samuel.omole/Desktop/repos/geofm/output/terramind_small_sen1floods11/checkpoints/best-mIoU.ckpt"
# let user supply first provide the dataset path for now (prompt the llm I have path to images / apply the tool)
# dataset_path = Path("/Users/samuel.omole/Desktop/repos/geofm_datasets/sen1floods11_v1.1")

s2_means=[2357.089, 2137.385, 2018.788, 2082.986, 2295.651, 2854.537, 3122.849, 3040.560, 3306.481, 1473.847, 506.070, 2472.825, 1838.929]
s1_means=[-12.599, -20.293]
s2_stds=[1624.683, 1675.806, 1557.708, 1833.702, 1823.738, 1733.977, 1732.131, 1679.732, 1727.26, 1024.687, 442.165, 1331.411, 1160.419]
s1_stds=[5.195, 5.890]

s2_path="/Users/samuel.omole/Desktop/repos/geofm_datasets/sen1floods11_v1.1/data/S2L1CHand/Somalia_699062_S2Hand.tif"
s1_path="/Users/samuel.omole/Desktop/repos/geofm_datasets/sen1floods11_v1.1/data/S1GRDHand/Somalia_699062_S1Hand.tif"
def load_model_from_checkpoint(checkpoint_path):
    model = terratorch.tasks.SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory",  # Combines a backbone with necks, the decoder, and a head
        model_args={
            # TerraMind backbone
            "backbone": "terramind_v1_small", # change to specific model e.g., for large version: terramind_v1_large 
            "backbone_pretrained": True,
            "backbone_modalities": ["S2L1C", "S1GRD"],
            # Optionally, define the input bands. This is only needed if you select a subset of the pre-training bands, as explained above.
            # "backbone_bands": {"S1GRD": ["VV"]},
            
            # Necks 
            "necks": [
                {
                    "name": "SelectIndices",
                    "indices": [2, 5, 8, 11] # indices for terramind_v1_base & small
                    # "indices": [5, 11, 17, 23] # indices for terramind_v1_large
                },
                {"name": "ReshapeTokensToImage",
                "remove_cls_token": False},  # TerraMind is trained without CLS token, which needs to be specified.
                {"name": "LearnedInterpolateToPyramidal"}  # Some decoders like UNet or UperNet expect hierarchical features.
            ],
            
            # Decoder
            "decoder": "UNetDecoder",
            "decoder_channels": [512, 256, 128, 64],
            
            # Head
            "head_dropout": 0.1,
            "num_classes": 2, # there are two classes in the mask label image
        },
        
        loss="dice",  # dice is recommended for binary tasks and ce for multi-class tasks. 
        optimizer="AdamW",
        lr=2e-5,  # We can perform hyperparameter optimization using terratorch-iterate but we have demonstrated that  
        ignore_index=-1,
        freeze_backbone=True, # Setting as True speeds up fine-tuning. It is recommended to fine-tune the backbone as well for the best performance. 
        freeze_decoder=False, # Should be false to update the decoder layer parameters
        plot_on_val=True,  # Plot predictions during validation steps  
        class_names=["Others", "Water"]  # optionally define class names
    )
    model = terratorch.tasks.SemanticSegmentationTask.load_from_checkpoint(
        checkpoint_path,
        model_factory=model.hparams.model_factory,
        model_args=model.hparams.model_args,
    )
    return model

def read_multiband_tif(path: str):
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    with rasterio.open(path) as src:
        arr = src.read() # shape (C, H, W)
        # convert to float32 for processing with Torch
        arr = arr.astype(np.float32)
    return arr  # shape (C, H, W)

def numpy_to_tensor(arr: np.ndarray):
    """
    arr_hwc: numpy array (H, W, C) in original units (e.g. raw DN, dB, etc.)
    Returns torch tensor (1, C, H, W) as float32 in the SAME units (no division).
    If target_size is provided, use PIL resize but keep scale (use interpolation then divide back as needed).
    """
    if arr.ndim == 3 and arr.shape[2] <= 20:
        arr_hwc = arr.copy()
    elif arr.ndim == 3 and arr.shape[0] <= 20:
        # C,H,W -> H,W,C
        arr_hwc = np.transpose(arr, (1,2,0))
    # Convert to float32 but preserve numeric range
    arr_float = arr_hwc.astype(np.float32)
    return torch.from_numpy(arr_float).permute(2,0,1).float().unsqueeze(0)  # (1,C,H,W) in original units

def normalize(tensor, means_orig, stds_orig):
    # tensor: (1,C,H,W)
    c = tensor.shape[1]
    if len(means_orig) != c or len(stds_orig) != c:
        raise ValueError(f"means & stds lengths of {len(means_orig)} & {len(stds_orig)} must match channels ({c})")
    means = torch.tensor(means_orig, dtype=torch.float32).view(1,c,1,1)
    stds  = torch.tensor(stds_orig,  dtype=torch.float32).view(1,c,1,1)
    return (tensor - means) / stds

def predict_pair(checkpoint_path: str, s2_path: str, s1_path: str, out_dir: str, show_plot: bool = True):
    """
    Predict using one S2L1C image and one S1GRD image.
    Returns: dict with keys: output_image_path, overlay_path, raw_prediction (numpy)
    """
    # delete out_dir everytime and recreate 
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    
    os.makedirs(out_dir, exist_ok=True)

    model = load_model_from_checkpoint(checkpoint_path)
    device = model.device

    s2 = read_multiband_tif(s2_path)   # (C_s2, H, W)
    s1 = read_multiband_tif(s1_path)   # (C_s1, H, W)

    # Apply normalisation based on the means and stds per channel
    s2_tensor = normalize(numpy_to_tensor(s2).to(device),
                          s2_means,
                          s2_stds,
    )   # (1, C_s2, H, W)
    s1_tensor = normalize(numpy_to_tensor(s1).to(device),
                          s1_means,
                          s1_stds,
    )   # (1, C_s1, H, W)

    # Build input images dict like datamodule provides to provide to model
    images = {
        "S2L1C": s2_tensor.to(device),   # shape (1, C, H, W)
        "S1GRD": s1_tensor.to(device),
    }

    model.eval()
    with torch.no_grad():
        outputs = model(images) 

    # img_preds is (B, num_classes, H, W)
    img_preds = outputs.output if hasattr(outputs, "output") else outputs
    if isinstance(img_preds, torch.Tensor):
        preds = torch.argmax(img_preds, dim=1).cpu().numpy()  # (B, H, W)
    else:
        # If model returns a dict/other, adapt accordingly
        raise ValueError("Unexpected model output format; inspect outputs to adapt postprocess")

    # Select the first and only prediction
    pred_mask = preds[0]
    # out_mask_path = os.path.join(out_dir, Path(s2_path).stem + "_pred_mask.png") # no need to save this

    # convert predicted binary mask to image and save (0/1 -> 0/255)
    pred_img = (pred_mask.astype(np.uint8) * 255)
    # Image.fromarray(pred_img).save(out_mask_path) # no need to save this

    # extract the S2 RGB channels for plotting purpose
    s2_rgb = None
    try:
        if s2.shape[0] <= 20:
            # indices chosen in your datamodule rgb_indices = [3,2,1] (1-based in your config). convert to 0-based
            rgb_idx = [3, 2, 1]  # user-specified earlier
            rgb_idx0 = [i - 1 for i in rgb_idx if i - 1 < s2.shape[0]]
            rgb_arr = np.transpose(s2[rgb_idx0, :, :], (1, 2, 0))
            # normalize for display
            rgb_arr = rgb_arr - rgb_arr.min()
            if rgb_arr.max() > 0:
                rgb_arr = rgb_arr / rgb_arr.max()
            rgb_img = (rgb_arr * 255).astype(np.uint8)
            s2_rgb = Image.fromarray(rgb_img)
    except Exception:
        s2_rgb = None

    # overlay predicted image on S2 RGB
    if s2_rgb is not None:
        overlay = s2_rgb.convert("RGBA")
        mask_color = Image.fromarray((pred_img).astype(np.uint8)).convert("L").resize(overlay.size)
        red_layer = Image.new("RGBA", overlay.size, (255, 0, 0, 50))  # semi-transparent red
        overlay.paste(red_layer, (0, 0), mask_color)
        overlay_path = os.path.join(out_dir, Path(s2_path).stem + "_overlay.png")
        overlay.save(overlay_path)

    # If show_plot is True
    if show_plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 7))
        if s2_rgb is not None:
            ax[0].imshow(s2_rgb)
            ax[0].set_title("S2 RGB")
        else:
            ax[0].imshow(np.transpose(s2[:3], (1, 2, 0)).astype(np.uint8))
            ax[0].set_title("S2 RGB")
        ax[1].imshow(pred_img, cmap="gray")
        ax[1].set_title("Predicted mask")
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        # to save the image
        fig_path = os.path.join(out_dir, Path(s2_path).stem + "_figure.png")
        fig.savefig(fig_path)
        plt.close(fig)

    return {
        "overlay_path": overlay_path,
        "figure_path": fig_path,
        "raw_pred": pred_mask,
    }

if __name__ == "__main__":
    predict_pair(checkpoint_path="/Users/samuel.omole/Desktop/repos/geofm/output/terramind_small_sen1floods11/checkpoints/best-mIoU.ckpt",
                 s2_path="/Users/samuel.omole/Desktop/repos/geofm_datasets/sen1floods11_v1.1/data/S2L1CHand/Somalia_699062_S2Hand.tif",
                 s1_path="/Users/samuel.omole/Desktop/repos/geofm_datasets/sen1floods11_v1.1/data/S1GRDHand/Somalia_699062_S1Hand.tif",
                 out_dir="/Users/samuel.omole/Desktop/repos/geofm_datasets/prediction_pair",
                 )

