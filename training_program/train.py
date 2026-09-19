from pathlib import Path
import yaml
import argparse
import cv2
import torch
import numpy as np
from torch import nn
import os
from utils.images_utils import Satellite_Tile, Drone_Image

class Residual_Block(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class Satellite_Vision_Model(nn.Module):
    def __init__(self, conv_layers: list[dict]):
        super().__init__()

        layers = []
        in_channels = 3

        for layer in conv_layers:
            out_channels = layer["filters"]
            kernel_size = layer["kernel_size"]

            layers.extend([nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                padding = kernel_size // 2
            ), nn.ReLU(),
            Residual_Block(out_channels)
            ])

            in_channels = out_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Drone_Vision_Model(nn.Module):
    def __init__(self, conv_layers: list[dict]):
        super().__init__()

        layers = []
        in_channels = 3

        for layer in conv_layers:
            out_channels = layer["filters"]
            kernel_size = layer["kernel_size"]

            layers.extend([nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                padding = kernel_size // 2
            ), nn.ReLU(),
            Residual_Block(out_channels)
            ])

            in_channels = out_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def cross_entropy_loss():
    pass

if __name__ == "__main__":

    # Read command line arguments for model name and config (optional)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    model_name = args.model_name
    config_name = args.config

    if config_name == None:
        config_name = "default.yaml"

    # Hyperparameters read from config file
    config_path = Path(__file__).resolve().parent.parent/"configs"/config_name
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    learning_rate = config.get("training").get("learning_rate")
    num_epochs = config.get("training").get("epochs")
    batch_size = config.get("training").get("batch_size")
    random_seed = config.get("training").get("seed")
    train_ratio = config.get("training").get("train_ratio")
    validation_ratio = config.get("training").get("validation_ratio")
    test_ratio = config.get("training").get("test_ratio")

    model_architecture = config.get("model").get("conv_layers")

    # Initialize models
    sv_model = Satellite_Vision_Model(model_architecture)
    dv_model = Drone_Vision_Model(model_architecture)

    # Training cycle + Validation testing
    for epoch in range(num_epochs):
        pass

    # Export model files for Satellite and Drone in seperate files
    # Format will be <name>_s.npy and <name>_d.npy respectively
    