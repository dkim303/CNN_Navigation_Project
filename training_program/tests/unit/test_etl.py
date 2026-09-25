from pathlib import Path
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from utils.data_etl import (
    load_drone_metadata,
    tile_positions,
    load_satellite_tiles_metadata,
    map_images_to_tiles
)