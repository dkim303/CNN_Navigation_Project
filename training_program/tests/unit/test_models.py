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

def test_load_empty_models_config():
    config_path = Path(__file__).resolve().parents[3] / "config" / "default.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_architecture = config.get("model").get("conv_layers")

    s_model = Satellite_Vision_Model(model_architecture)
    d_model = Drone_Vision_Model(model_architecture)

    s_state = s_model.state_dict()
    d_state = d_model.state_dict()

    conv_layers = [layer for layer in s_model.modules() if isinstance(layer, nn.Conv2d)]

    assert len(conv_layers) == 9

    expected_shapes = [
        (3, 32),
        (32, 32),
        (32, 32),
        (32, 64),
        (64, 64),
        (64, 64),
        (64, 128),
        (128, 128),
        (128, 128),
    ]

    actual_shapes = [(layer.in_channels, layer.out_channels) for layer in conv_layers]
    assert actual_shapes == expected_shapes

def test_model_load_file():
    pass