from pathlib import Path
import cv2
import torch
import numpy as np
from images_utils import Satellite_Tile, Drone_Image
import pandas as pd

def compare_channel_statistics(dataset_A: pd.DataFrame, dataset_B: pd.DataFrame) -> float:
    pass

def MMD(dataset_A: pd.DataFrame, dataset_B: pd.DataFrame) -> float:
    pass

def FID(dataset_A: pd.DataFrame, dataset_B: pd.DataFrame) -> float:
    pass

def Wasserstein_Distance(dataset_A: pd.DataFrame, dataset_B: pd.DataFrame) -> float:
    pass

def Hotelling_T2(dataset_A: pd.DataFrame, dataset_B: pd.DataFrame) -> float:
    pass
