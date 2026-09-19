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

# Structs for the Sattelite Tiles and for Drone Images
# Contains info on mapping drone images to the correct map and correct coords
@dataclass(frozen=True)
class Satellite_Tile:
    tile_id: str
    map_id: str
    image_path: Path

    # Position within the original satellite image
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    # Geographic bounds
    north_lat: float
    south_lat: float
    west_lon: float
    east_lon: float

    @property
    def center_lat(self) -> float:
        return (self.north_lat + self.south_lat) / 2

    @property
    def center_lon(self) -> float:
        return (self.west_lon + self.east_lon) / 2

    def contains_coordinate(self, latitude: float, longitude: float) -> bool:
        return (
            self.south_lat <= latitude <= self.north_lat
            and self.west_lon <= longitude <= self.east_lon
        )

    def load_tensor(self) -> torch.Tensor:
        return load_image_tensor(self.image_path)

@dataclass(frozen=True)
class Drone_Image:
    image_id: str
    map_id: str
    image_path: Path

    latitude: float
    longitude: float
    height: float

    pitch: float
    roll: float
    yaw: float

    def load_tensor(self) -> torch.Tensor:
        return load_image_tensor(self.image_path)

@dataclass(frozen=True)
class SatelliteTileEmbedding:
    tile: Satellite_Tile
    embedding: torch.Tensor