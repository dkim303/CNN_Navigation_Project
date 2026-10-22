from pathlib import Path
import yaml
import argparse
import cv2
import torch
import numpy as np
from torch import nn
import os
import pandas as pd
from PIL import Image
from utils.models import Satellite_Vision_Model, Drone_Vision_Model

class Residual_Block(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
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
                padding = kernel_size // 2,
                bias=False
            ), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            Residual_Block(out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
            ])

            in_channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        final_channels = conv_layers[-1]["filters"]
        self.projection = nn.Linear(final_channels, final_channels)

    # Output should be a high dimensional vector
    def forward(self, x):
        x = self.layers(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.projection(x)
        return x


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
                padding = kernel_size // 2,
                bias=False
            ), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            Residual_Block(out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
            ])

            in_channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        final_channels = conv_layers[-1]["filters"]
        self.projection = nn.Linear(final_channels, final_channels)

    # Output should be a high dimensional vector
    def forward(self, x):
        x = self.layers(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.projection(x)
        return x


def load_cnn_models(model_name: str) -> tuple[Satellite_Vision_Model, Drone_Vision_Model]:
    Satellite_Model = Satellite_Vision_Model()
    Drone_Model = Drone_Vision_Model()

    # Create paths to find model weights files
    models_folder_path = Path(__file__).parent.parent/"models/"
    satellite_cnn_model_name = model_name + "_SATELLITE"
    drone_cnn_model_name = model_name + "_DRONE"
    satellite_model_path = models_folder_path/satellite_cnn_model_name
    drone_model_path = models_folder_path/drone_cnn_model_name

    # Load torch files into state dicts
    loaded_satellite_dict = torch.load(satellite_model_path, weights_only = True)
    loaded_drone_dict = torch.load(drone_model_path, weights_only = True)

    # Upload state dict weights into models
    Satellite_Model.load_state_dict(loaded_satellite_dict)
    Drone_Model.load_state_dict(loaded_drone_dict)

    return (Satellite_Model, Drone_Model)


class Geo_Locator:
    Satellite_Vision_Model
    Drone_Vision_Model

    def __init__(self, model_name: str):
        self.satellite_model, self.drone_model = load_cnn_models(model_name)
        self.satellite_metadata = pd.DataFrame()
        self.satellite_embeddings: torch.Tensor = None

    # Read satellite data and maintain dataframe of values along with tensor of embeddings
    def read_satellite(self, satellite_images_path: str):
        pass

    # Given an image, locate the satellite tile and estimate coordinates
    def locate(self, drone_image_path, top_k: int = 5):
        pass

    # Clear all memorized satellite data
    def clear(self):
        pass

    # Return number of tracked satellite tiles
    def __len__(self):
        return len(self.satellite_metadata)

    