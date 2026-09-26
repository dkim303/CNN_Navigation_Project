from pathlib import Path
import yaml
import argparse
import cv2
import torch
import numpy as np
from torch import nn
import os
from utils.images_utils import load_image_tensor
import pandas as pd
from dotenv import load_dotenv

from utils.data_etl import load_drone_metadata, load_satellite_tiles_metadata, map_images_to_tiles, data_partition_TVT, check_data_leakage, load_datasets_TVT
from utils.models import Satellite_Vision_Model, Drone_Vision_Model
from utils.statistical_diagnostics import create_random_sample

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
    weight_decay_rate = config.get("training").get("weight_decay")
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
    rng = np.random.default_rng(random_seed) # U(0,1)
    torch.manual_seed(random_seed)

    # Extract the model architecutre as list[dict]
    model_architecture = config.get("model").get("conv_layers")

    # Construct initial models, weights are initially randomized using the random seed
    sv_model = Satellite_Vision_Model(model_architecture)
    dv_model = Drone_Vision_Model(model_architecture)
    
    # Load sattelite maps metadata CSV
    # satellite_maps_csv_df expected format:
    #
    # Index:
    #   map_id        string   Unique map ID extracted from map_filename,
    #                        e.g. "01" from "satellite01.tif"
    #
    # Columns:
    #   map_filename  string   Satellite-map image filename
    #   north_lat     float64  Latitude of the map's northern boundary
    #   west_lon      float64  Longitude of the map's western boundary
    #   south_lat     float64  Latitude of the map's southern boundary
    #   east_lon      float64  Longitude of the map's eastern boundary
    #
    # Example:
    #
    # map_id | map_filename    | north_lat | west_lon | south_lat | east_lon
    # 01     | satellite01.tif | 29.750... | 115.980... | 29.720... | 116.020...
    # 02     | satellite02.tif | 30.110... | 116.200... | 30.080... | 116.240...
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
    project_root = Path(__file__).resolve().parents[1]

    # tiles_df format:
    #
    # Index:
    #   tile_id           string   Unique tile ID, e.g. "01_r0000_c0000"
    #
    # Columns:
    #   map_id           string   Source satellite-map ID, e.g. "01"
    #   satellite_path   string   Path to the original satellite image
    #
    #   x_min            int64    Left pixel boundary, inclusive
    #   y_min            int64    Top pixel boundary, inclusive
    #   x_max            int64    Right pixel boundary, exclusive
    #   y_max            int64    Bottom pixel boundary, exclusive
    #
    #   north_lat        float64  Northern geographic boundary
    #   south_lat        float64  Southern geographic boundary
    #   west_lon         float64  Western geographic boundary
    #   east_lon         float64  Eastern geographic boundary
    #
    #   center_lat       float64  Latitude of the tile center
    #   center_lon       float64  Longitude of the tile center
    tiles_df = load_satellite_tiles_metadata(satellite_maps_csv_df, 
                                             project_root / "data",
                                             tile_size_pixels, 
                                             stride_pixels)

    # Read each CSV and update the drone_images_df, tiles are not mapped yet
    # Index:
    #   image_id         string   Unique drone-image ID, e.g. "01_0001"
    #
    # Columns:
    #   image_path       string   Path to the image file
    #   lat              float64  Latitude of the image center
    #   lon              float64  Longitude of the image center
    #   map_id           string   Source satellite-map ID, e.g. "01"
    #   primary_tile_id  string   Best matching tile ID; initially missing
    drone_images_df = load_drone_metadata(NUM_SATELLITE_MAPS)

    if len(drone_images_df) != NUM_DRONE_IMAGES:
        raise ValueError(f"Exepected {NUM_DRONE_IMAGES} drone images, found {len(drone_images_df)}")

    # Perform mapping of correct tiles to drone images in the drone_images_df
    # drone_images_df is updated to have all cells in primary_tile_id col filled out
    map_images_to_tiles(drone_images_df, tiles_df)

    # Train - Validation - Test split
    # Create new column that designates which bucket it falls into: train - validation - test
    data_partition_TVT(drone_images_df,
                       tiles_df,
                       NUM_SATELLITE_MAPS,
                       train_ratio,
                       validation_ratio,
                       test_ratio,
                       rng)

    # Statistical diagonostics to ensure effective split and data leakage tests
    check_data_leakage(drone_images_df, tiles_df)

    # Unpack 3 data loader objects to be used in training process
    training_loader, validation_loader, test_loader = load_datasets_TVT(drone_images_df, tiles_df, batch_size, model_input_size)

    # Set up optimizer
    optimizer = torch.optim.AdamW(list(dv_model.parameters()) + list(sv_model.parameters()),
                                  lr=learning_rate, 
                                  weight_decay=weight_decay_rate)

    # Define loss function as Cross-Entropy loss
    loss_fn = nn.CrossEntropyLoss()
    
    # Metrics to track during training
    metrics = {
        "training_loss": [],
        "validation_loss": [],
        "validation_top1": [],
        "validation_top5": [],
        "validation_median_distance_m": [],
        "positive_similarity": [],
        "negative_similarity": [],
        "gradient_norm": [],
        "learning_rate": [],
    }

    # Models training step
    # Use training data to make gradients and check validation accuracy
    for epoch in range(num_epochs):
        # Train section
        sv_model.train()
        dv_model.train()

        for batch in training_loader:
            # drone_images: [B, 4, 3, S, S]
            # tile_images:  [B, 3, S, S]

            drone_images = batch["drone_tensor"]
            tile_images = batch["tile_tensor"]

            batch_size_actual, num_quadrants, channels, height, width = (drone_images.shape)

        # Validation test section
        sv_model.eval()
        dv_model.eval()
        with torch.no_grad():
            for batch in validation_loader:

                pass

    # Run test dataset
    sv_model.eval()
    dv_model.eval()
    with torch.no_grad():
        for batch in test_loader:

            pass

    # Export model files for Satellite and Drone in seperate files
    # Format will be <name>_s.tch and <name>_d.tch respectively

    model_dest_path = str(Path(__file__).parent.parent / "models")
    s_path = model_dest_path / f"{model_name}_s.pt"
    d_path = model_dest_path / f"{model_name}_d.pt"
 
    torch.save(x, s_path)
    torch.save(y, d_path)