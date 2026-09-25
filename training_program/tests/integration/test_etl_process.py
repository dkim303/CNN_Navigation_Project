from pathlib import Path
import yaml
import argparse
import cv2
import torch
import numpy as np
from torch import nn
import os
import pandas as pd
from utils.models import Satellite_Vision_Model, Drone_Vision_Model

def test_workflo():
    config_path = Path(__file__).resolve().parents[3] / "config" / "default.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_architecture = config.get("model").get("conv_layers")
    s_model = Satellite_Vision_Model(model_architecture)
    d_model = Drone_Vision_Model(model_architecture)

    