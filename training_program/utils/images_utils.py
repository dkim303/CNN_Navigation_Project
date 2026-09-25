from dataclasses import dataclass
from pathlib import Path
import cv2
import torch
import numpy as np

def load_image_tensor(image_path: Path) -> torch.Tensor:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # OpenCV uses BGR, manually switch to RBG
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # H × W × C -> C × H × W
    tensor = torch.from_numpy(image).permute(2, 0, 1)

    # uint8 [0, 255] -> float32 [0, 1]
    return tensor.float().div(255.0)

