from pathlib import Path
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from utils.images_utils import load_image_tensor

def test_tensor_loading():
    img_path = Path(__file__).parent.parent/"test_data"/"test_image_1.JPG"
    tensor = load_image_tensor(img_path)

    # Check correct color channels and dimensions: 3976 × 2652
    assert(tensor.shape == torch.Size([3, 2652, 3976]))
    assert(tensor.dtype == torch.float32)

def test_load_null():
    null_img_path = Path(__file__).resolve()

    with pytest.raises(ValueError):
        tensor = load_image_tensor(null_img_path)
