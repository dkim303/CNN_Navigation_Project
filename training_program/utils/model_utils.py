from pathlib import Path
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from utils.models import Satellite_Vision_Model, Drone_Vision_Model

def load_cnn_models(model_name: str):
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