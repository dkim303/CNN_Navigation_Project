from pathlib import Path
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from utils.images_utils import (
    load_image_tensor,
    Satellite_Tile,
    Drone_Image,
    SatelliteTileEmbedding
)

def test_tensor_loading():
    img_path = Path(__file__).parent/"test_data"/"test_image_1.JPG"
    tensor = load_image_tensor(img_path)

    # Check correct color channels and dimensions: 3976 × 2652
    assert(tensor.shape == torch.Size([3, 2652, 3976]))
    assert(tensor.dtype == torch.float32)

def test_load_null():
    null_img_path = Path(__file__).resolve()

    with pytest.raises(ValueError):
        tensor = load_image_tensor(null_img_path)

def test_satellite_tile():
    tile = Satellite_Tile(tile_id=1, 
                          map_id=1, 
                          image_path=None, 
                          x_min=0, 
                          y_min=0, 
                          x_max=5, 
                          y_max=5,
                          north_lat=100,
                          south_lat=-100,
                          west_lon=10,
                          east_lon=20)

    assert(tile.center_lat == 0)
    assert(tile.center_lon == 15)
    assert(tile.contains_coordinate(0, 12) == True)

def test_drone_image():
    pass

