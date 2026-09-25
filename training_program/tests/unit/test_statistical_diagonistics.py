from pathlib import Path
import cv2
import torch
import numpy as np
import pandas as pd

from utils.statistical_diagnostics import create_random_sample

def test_create_random_sample():
    df = pd.DataFrame({
        "tile_id": range(30),
        "dataset": ["train"] * 20 + ["test"] * 10,
        "value": range(30),
    })

    rng = np.random.default_rng(42)
    result = create_random_sample(df, "train", rng)

    assert len(result) == 2
    assert (result["dataset"] == "train").all()
    assert result["tile_id"].isin(df.loc[df["dataset"] == "train", "tile_id"]).all()

