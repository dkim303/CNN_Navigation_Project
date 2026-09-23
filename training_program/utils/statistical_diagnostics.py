from pathlib import Path
import cv2
import torch
import numpy as np
import pandas as pd

# Sample with replacement a sample size 10% of the population size
# dataset arg meant to specify samples are from train-validate-test sets 
def create_random_sample(df: pd.DataFrame,
                         dataset: str,
                         rng: np.random.Generator) -> pd.DataFrame:
    
    data_set = df.loc[df["dataset"] == dataset]
    sample_size = int(len(data_set) * 0.1)

    # if sample size is 0, just return a copy of the data set
    if sample_size == 0:
        return data_set.iloc[:0].copy()

    positions = rng.integers(0, len(data_set), size=sample_size)    
    return data_set.iloc[positions].copy()

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
