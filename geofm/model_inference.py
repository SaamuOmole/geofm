import os
import torch
import gdown
import terratorch
import albumentations
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from pathlib import Path
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
import warnings
warnings.filterwarnings("ignore")

# define model checkpoint and dataset paths
# checkpoint_path = "/Users/samuel.omole/Desktop/repos/geofm/output/terramind_small_sen1floods11/checkpoints/best-mIoU.ckpt"
# let user supply first provide the dataset path for now (prompt the llm I have path to images / apply the tool)
# dataset_path = Path("/Users/samuel.omole/Desktop/repos/geofm_datasets/sen1floods11_v1.1")
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

def create_datamodule(dataset_path):
    dataset_path = Path(dataset_path)
    datamodule = terratorch.datamodules.GenericMultiModalDataModule(
        task="segmentation",
        batch_size=8,
        num_workers=2,
        num_classes=2,
        # Define input modalities. The names must match the keys in the dicts below and everywhere.
        modalities=["S2L1C", "S1GRD"],
        rgb_modality="S2L1C",  # Used for plotting. Defaults to the first modality if not provided.
        rgb_indices=[3,2,1],  # RGB channel positions in the rgb_modality.

        # Define data paths as dicts using the modality names as keys.
        train_data_root={
            "S2L1C": dataset_path / "data/S2L1CHand",
            "S1GRD": dataset_path / "data/S1GRDHand",
        },
        train_label_data_root=dataset_path / "data/LabelHand",
        val_data_root={
            "S2L1C": dataset_path / "data/S2L1CHand",
            "S1GRD": dataset_path / "data/S1GRDHand",
        },
        val_label_data_root=dataset_path / "data/LabelHand",
        test_data_root={
            "S2L1C": dataset_path / "data/S2L1CHand",
            "S1GRD": dataset_path / "data/S1GRDHand",
        },
        test_label_data_root=dataset_path / "data/LabelHand",

        # Define split files
        train_split=dataset_path / "splits/flood_train_data.txt",
        val_split=dataset_path / "splits/flood_valid_data.txt",
        test_split=dataset_path / "splits/flood_test_data.txt",
        
        # Define suffix
        image_grep={
            "S2L1C": "*_S2Hand.tif",
            "S1GRD": "*_S1Hand.tif",
        },
        label_grep="*_LabelHand.tif",

        dataset_bands={
            "S1GRD": ["VV", "VH"]
        },
        output_bands={
            "S1GRD": ["VV", "VH"]
        },

        means={
        "S2L1C": [2357.089, 2137.385, 2018.788, 2082.986, 2295.651, 2854.537, 3122.849, 3040.560, 3306.481, 1473.847, 506.070, 2472.825, 1838.929],
        "S1GRD": [-12.599, -20.293],
        },
        stds={
        "S2L1C": [1624.683, 1675.806, 1557.708, 1833.702, 1823.738, 1733.977, 1732.131, 1679.732, 1727.26, 1024.687, 442.165, 1331.411, 1160.419],
        "S1GRD": [5.195, 5.890],
        },
        
        # Apply albumentations to augment the dataset
        train_transform=[
            albumentations.D4(), # Performs random flips and rotation
            albumentations.pytorch.transforms.ToTensorV2(),
        ],
        val_transform=None,  # Applies ToTensorV2() by default if not provided
        test_transform=None,
        
        no_label_replace=-1,  # Replace NaN labels. defaults to -1 which is ignored in the loss and metrics.
        no_data_replace=0,  # Replace NaN data
    )
    return datamodule


def predict_image_batches(checkpoint_path, dataset_path):
    model = load_model_from_checkpoint(checkpoint_path)
    datamodule = create_datamodule(dataset_path)
    # set datamodule for test set
    datamodule.setup("test")
    test_dataset = datamodule.test_dataset
    print("Length of test dataset is: ", len(test_dataset))

    # pass the test set in batches (selecting only first batch) 
    test_loader = datamodule.test_dataloader()
    with torch.no_grad():
        batch = next(iter(test_loader)) # this only selects the first batch in the test_dataloader
        images = batch["image"]
        for mod, value in images.items():
            images[mod] = value.to(model.device)
        # masks = batch["mask"].numpy()

        with torch.no_grad():
            outputs = model(images)
        
        preds = torch.argmax(outputs.output, dim=1).cpu().numpy()

    for i in range(1): # 8 is the batch size so set this to <= 8 (plot only the first example from the batch)
        sample = {
            "image": batch["image"]["S2L1C"][i].cpu(),
            "mask": batch["mask"][i],
            "prediction": preds[i],
        }
        test_dataset.plot(sample)
        plt.show()
