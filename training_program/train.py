from pathlib import Path
import yaml
import argparse
import cv2
import torch
import numpy as np
from torch import nn
import os
from utils.images_utils import Satellite_Tile, Drone_Image
import pandas as pd
from dotenv import load_dotenv

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
            Residual_Block(out_channels)
            ])

            in_channels = out_channels

        self.layers = nn.Sequential(*layers)

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
            Residual_Block(out_channels)
            ])

            in_channels = out_channels

        self.layers = nn.Sequential(*layers)

    # Output should be a high dimensional vector
    def forward(self, x):
        x = self.layers(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.projection(x)
        return x

if __name__ == "__main__":

    # Read command line arguments for model name and config (optional)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    model_name = args.model_name
    config_name = args.config

    # use default.yaml architecture if no config is given
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

    train_ratio = config.get("data").get("train_ratio")
    validation_ratio = config.get("data").get("validation_ratio")
    test_ratio = config.get("data").get("test_ratio")

    tile_size_pixels = config.get("data").get("satellite_tiling").get("tile_size_pixels")
    stride_pixels = config.get("data").get("satellite_tiling").get("stride_pixels")
    model_input_size = config.get("data").get("satellite_tiling").get("model_input_size")

    # Environment information on data
    env_path = Path(__file__).resolve().parent.parent/"data.env"
    load_dotenv(env_path)
    DATASET_NAME = os.getenv("DATASET_NAME")
    NUM_DRONE_IMAGES = int(os.getenv("NUM_DRONE_IMAGES"))
    NUM_SATELLITE_MAPS = int(os.getenv("NUM_SATELLITE_MAPS"))

    # Set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Extract the model architecutre as list[dict]
    model_architecture = config.get("model").get("conv_layers")

    # Read info on dataset from data.env file


    # Construct initial models, weights are initially randomized using the random seed
    sv_model = Satellite_Vision_Model(model_architecture)
    dv_model = Drone_Vision_Model(model_architecture)

    # Define loss function as Cross-Entropy loss
    loss_fn = nn.CrossEntropyLoss()

    # Load data metadata into 2 Dataframes
    # Drone_DF: index=drone_id, cols=image_path,map_id,lattitude,longitude, split
    # Tile_DF: index=tile_id, x_min, y_min, x_max, y_max, north_lat, south_lat, west_long, east_lon, split
    
    # Step 1: Load sattelite maps CSV
    satellite_csv_path = Path(__file__).resolve().parent.parent/"data"/"satellite_ coordinates_range.csv"
    satellite_maps_csv_df = pd.read_csv(satellite_csv_path)
    satellite_maps_csv_df = satellite_maps_csv_df.rename(columns={
        "mapname": "map_filename",
        "LT_lat_map": "north_lat",
        "LT_lon_map": "west_lon",
        "RB_lat_map": "south_lat",
        "RB_lon_map": "east_lon",
    })

    satellite_maps_csv_df["map_id"] = (
        satellite_maps_csv_df["map_filename"]
        .str.extract(r"satellite(\d+)", expand=False)
    )

    satellite_maps_csv_df = satellite_maps_csv_df.set_index("map_id", verify_integrity=True)

    # Step 2: Create dataframe for satellite tiles metadata using csv dataframe to calculate corner coordinates
    Tiles_DF = 

    # Step 3: For each of the sattelites, read the CSVs of the drone photos
        # Cols should be: unique_photo_id, lat, lon, correct_tile_id

    drone_images_df = pd.DataFrame({
        "image_id": pd.Series(dtype="string"),
        "image_path": pd.Series(dtype="string"),
        "lat": pd.Series(dtype="float64"),
        "lon": pd.Series(dtype="float64"),
        "map_id": pd.Series(dtype="string"),
        "primary_tile_id": pd.Series(dtype="string"),
    })

    # Read each CSV and update the drone_images_df
    for map_id in range(NUM_SATELLITE_MAPS):


    # Train - Validation - Test split

    # Statistical diagonostics to ensure effective split

    # Training cycle + Validation testing
    for epoch in range(num_epochs):
        pass
        # Validation testing

    

    # Export model files for Satellite and Drone in seperate files
    # Format will be <name>_s.npy and <name>_d.npy respectively
    