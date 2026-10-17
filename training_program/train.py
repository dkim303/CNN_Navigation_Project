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
import logging

from utils.data_etl import load_drone_metadata, load_satellite_tiles_metadata, map_images_to_tiles, data_partition_TVT, check_data_leakage, load_datasets_TVT, create_similarity_logits
from utils.models import Satellite_Vision_Model, Drone_Vision_Model
from utils.statistical_diagnostics import create_random_sample
from utils.logging import setup_logging

if __name__ == "__main__":
    setup_logging("Model Training", Path(__file__).resolve().parent.parent / "logs")

    # Read command line arguments for model name and config (optional)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    model_name = args.model_name
    config_name = args.config
    logging.info(f"Model Name: {model_name}")

    # use default.yaml architecture if no config is given
    if config_name == None:
        config_name = "default.yaml"

    logging.info(f"Config Name: {config_name}")

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
    logging.info("Satellite and Drone Models Initialized")
    
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
    logging.info("Satellite maps CSV loaded")

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
    logging.info("Tiles DF loaded")

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
    logging.info("Drone Images DF loaded")

    if len(drone_images_df) != NUM_DRONE_IMAGES:
        raise ValueError(f"Exepected {NUM_DRONE_IMAGES} drone images, found {len(drone_images_df)}")

    # Perform mapping of correct tiles to drone images in the drone_images_df
    # drone_images_df is updated to have all cells in primary_tile_id col filled out
    map_images_to_tiles(drone_images_df, tiles_df)
    logging.info("Drone Images and Tiles mapping procedure done")

    # Train - Validation - Test split
    # Create new column that designates which bucket it falls into: train - validation - test
    data_partition_TVT(drone_images_df,
                       tiles_df,
                       NUM_SATELLITE_MAPS,
                       train_ratio,
                       validation_ratio,
                       test_ratio,
                       rng)
    logging.info("Data partition for training, validation, test datasets done")

    # Statistical diagonostics to ensure effective split and data leakage tests
    check_data_leakage(drone_images_df, tiles_df)
    logging.info("Data leakage check passed")

    # Unpack 3 data loader objects to be used in training process
    training_loader, validation_loader, test_loader = load_datasets_TVT(drone_images_df, tiles_df, batch_size, model_input_size)
    logging.info("Dataset loaders created")

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
        "satellite_gradient_norm": [],
        "drone_gradient_norm": [],
        "test_loss": []
    }

    # Models training step
    # Use training data to make gradients and check validation accuracy
    logging.info("Training process beginning")
    for epoch in range(num_epochs):
        # Model training section
        sv_model.train()
        dv_model.train()

        # Batch values for training loss
        training_loss_sum = 0.0
        training_samples = 0

        # Batch data for gradients
        satellite_grad_norm_sum = 0.0
        drone_grad_norm_sum = 0.0
        training_batches = 0

        for batch in training_loader:
            # drone_images: [B, 4, 3, S, S]
            # tile_images:  [B, 3, S, S]
            drone_images = batch["drone_tensor"]
            tile_images = batch["tile_tensor"]

            batch_size_actual = drone_images.shape[0]
            targets = torch.arange(batch_size_actual)

            # Clear current gradient and compute gradient for current batch
            optimizer.zero_grad(set_to_none=True)   

            similarity_logits = create_similarity_logits(drone_images,
                                                         tile_images,
                                                         dv_model,
                                                         sv_model)

            loss = loss_fn(similarity_logits, targets)
            loss.backward()

            # Update batch training loss sum
            training_loss_sum += loss.item() * batch_size_actual
            training_samples += batch_size_actual

            # Update gradient norm value
            s_grads = [p.grad for p in sv_model.parameters() if p.grad is not None]
            s_grad_norm = torch.nn.utils.get_total_norm(s_grads, norm_type=2)

            d_grads = [p.grad for p in dv_model.parameters() if p.grad is not None]
            d_grad_norm = torch.nn.utils.get_total_norm(d_grads, norm_type=2)

            # Update batch gradients
            satellite_grad_norm_sum += s_grad_norm.item()
            drone_grad_norm_sum += d_grad_norm.item()
            training_batches += 1
            optimizer.step()

        # Update all training metrics
        metrics["training_loss"].append(training_loss_sum / training_samples)
        metrics["satellite_gradient_norm"].append(satellite_grad_norm_sum / training_batches)
        metrics["drone_gradient_norm"].append(drone_grad_norm_sum / training_batches) 

        # Model validation section
        sv_model.eval()
        dv_model.eval()
        with torch.no_grad():
            validation_loss_sum = 0.0
            validation_samples = 0

            for batch in validation_loader:
                drone_images = batch["drone_tensor"]
                tile_images = batch["tile_tensor"]
                batch_size_actual = drone_images.shape[0]
                targets = torch.arange(batch_size_actual)
                
                similarity_logits = create_similarity_logits(drone_images,
                                                            tile_images,
                                                            dv_model,
                                                            sv_model)

                loss = loss_fn(similarity_logits, targets)
                validation_loss_sum += loss.item() * batch_size_actual
                validation_samples += batch_size_actual

            # Update validation loss metric
            metrics["validation_loss"].append(validation_loss_sum / validation_samples)

    # Model test section
    logging.info("Final evaluation on test dataset running")
    sv_model.eval()
    dv_model.eval()
    with torch.no_grad():
        test_loss_sum = 0.0
        test_samples = 0

        for batch in test_loader:
            drone_images = batch["drone_tensor"]
            tile_images = batch["tile_tensor"]
            batch_size_actual = drone_images.shape[0]

            targets = torch.arange(batch_size_actual)

            similarity_logits = create_similarity_logits(drone_images,
                                                         tile_images,
                                                         dv_model,
                                                         sv_model)

            loss = loss_fn(similarity_logits, targets)
            test_loss_sum += loss.item() * batch_size_actual
            test_samples += batch_size_actual

        metrics["test_loss"].append(test_loss_sum / test_samples)

    logging.info("Training Procedure Completed:")
    logging.info(f"Final Test Loss: {metrics["test_loss"][-1]}")

    logging.info("Beggining Model export process")

    # Export model files for Satellite and Drone in seperate files
    # Format will be <name>_s.pt and <name>_d.pt respectively
    model_dest_path = Path(__file__).parent.parent / "models"
    model_dest_path.mkdir(parents=True, exist_ok=True)

    s_path = model_dest_path / f"{model_name}_s.pt"
    d_path = model_dest_path / f"{model_name}_d.pt"
    torch.save(sv_model.state_dict(), s_path)
    torch.save(dv_model.state_dict(), d_path)

    logging.info("Torch model state dictionaries exported")
    logging.info("Program terminated successfully")