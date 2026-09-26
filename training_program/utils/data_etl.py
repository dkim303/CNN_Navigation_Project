from pathlib import Path
import cv2
import torch
import numpy as np
from utils.images_utils import load_image_tensor
import pandas as pd
from PIL import Image
import math
from collections.abc import Callable

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms import v2

# drone_images_df format:
#
# Index:
#   image_id         string   Unique drone-image ID, e.g. "01_0001"
#
# Columns:
#   image_path       string   Path to the image file
#   lat              float64  Latitude of the image center
#   lon              float64  Longitude of the image center
#   map_id           string   Source satellite-map ID, e.g. "01"
#   primary_tile_id  string   Best matching tile ID; initially missing
#
# Example:
#
# image_id | image_path                    | lat       | lon        | map_id | primary_tile_id
# 01_0001  | data/01/drone/01_0001.JPG     | 29.74...  | 115.98... | 01     | <NA>
# 01_0002  | data/01/drone/01_0002.JPG     | 29.74...  | 115.98... | 01     | <NA>
def load_drone_metadata(NUM_SATELLITE_MAPS: int) -> pd.DataFrame:
    dataframes_list: list[pd.DataFrame] = []
    project_root = Path(__file__).resolve().parents[1]
    data_directory = project_root / "data"

    # For each folder of satellite data
    for map_number in range(1, NUM_SATELLITE_MAPS + 1):
        map_id = f"{map_number:02d}"

        map_directory = data_directory / map_id
        csv_path = map_directory / f"{map_id}.csv"
        drone_directory = map_directory / "drone"

        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing drone metadata CSV file: {csv_path}")

        # Read the metadata csv and label drop unnecessary info
        curr_df = pd.read_csv(csv_path)
        curr_df.columns = curr_df.columns.str.strip()
        required_columns = {"filename", "lat", "lon"}
        missing_columns = required_columns - set(curr_df.columns)

        if missing_columns:
            raise ValueError(
                f"{csv_path} is missing columns: {missing_columns}"
            )

        curr_df = curr_df[
            ["filename", "lat", "lon"]
        ].copy()

        curr_df["image_id"] = (
            curr_df["filename"]
            .map(lambda filename: Path(str(filename)).stem)
            .astype("string")
        )

        curr_df["image_path"] = (
            curr_df["filename"]
            .map(
                lambda filename: str(
                    drone_directory / str(filename)
                )
            )
            .astype("string")
        )

        curr_df["map_id"] = map_id

        # Assigned later after satellite tiles are generated.
        curr_df["primary_tile_id"] = pd.NA

        curr_df = curr_df[
            [
                "image_id",
                "image_path",
                "lat",
                "lon",
                "map_id",
                "primary_tile_id",
            ]
        ]

        dataframes_list.append(curr_df)

    drone_images_df = pd.concat(
        dataframes_list,
        ignore_index=True,
    )

    drone_images_df = drone_images_df.astype({
        "image_id": "string",
        "image_path": "string",
        "lat": "float64",
        "lon": "float64",
        "map_id": "string",
        "primary_tile_id": "string",
    })

    drone_images_df = drone_images_df.set_index(
        "image_id",
        verify_integrity=True,
    )

    return drone_images_df


# Return list of the horizontal start points of tiles in the satellite image
def tile_positions(
    image_length: int,
    tile_size: int,
    stride: int,
) -> list[int]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    if stride <= 0:
        raise ValueError("stride must be positive")

    if image_length <= tile_size:
        return [0]

    positions = list(
        range(
            0,
            image_length - tile_size + 1,
            stride,
        )
    )

    # Ensure the final tile reaches the image boundary.
    final_position = image_length - tile_size

    if positions[-1] != final_position:
        positions.append(final_position)

    return positions

def load_satellite_tiles_metadata(satellite_csv_df: pd.DataFrame,
                                  dataset_directory: Path,
                                  tile_size_pixels: int,
                                  stride_pixels: int) -> pd.DataFrame:
    
    tile_rows: list[dict] = []

    # Expected satellite_csv_df format:
    #
    # Index:
    #   map_id
    #
    # Columns:
    #   map_filename
    #   north_lat
    #   west_lon
    #   south_lat
    #   east_lon
    for raw_map_id, row in satellite_csv_df.iterrows():
        map_id = str(raw_map_id).zfill(2)

        satellite_path = (
            dataset_directory
            / map_id
            / row["map_filename"]
        )

        if not satellite_path.is_file():
            raise FileNotFoundError(
                f"Missing satellite image: {satellite_path}"
            )

        # Pillow reads the image header to obtain dimensions.
        # It does not normally decode the complete image here.
        with Image.open(satellite_path) as image:
            image_width, image_height = image.size

        lat_per_pixel = (
            row["north_lat"] - row["south_lat"]
        ) / image_height

        lon_per_pixel = (
            row["east_lon"] - row["west_lon"]
        ) / image_width

        x_positions = tile_positions(
            image_width,
            tile_size_pixels,
            stride_pixels,
        )

        y_positions = tile_positions(
            image_height,
            tile_size_pixels,
            stride_pixels,
        )

        for tile_row_index, y_min in enumerate(y_positions):
            for tile_column_index, x_min in enumerate(x_positions):
                # x_max and y_max are exclusive bounds.
                x_max = min(
                    x_min + tile_size_pixels,
                    image_width,
                )
                y_max = min(
                    y_min + tile_size_pixels,
                    image_height,
                )

                tile_north_lat = (
                    row["north_lat"]
                    - y_min * lat_per_pixel
                )

                tile_south_lat = (
                    row["north_lat"]
                    - y_max * lat_per_pixel
                )

                tile_west_lon = (
                    row["west_lon"]
                    + x_min * lon_per_pixel
                )

                tile_east_lon = (
                    row["west_lon"]
                    + x_max * lon_per_pixel
                )

                tile_id = (
                    f"{map_id}_"
                    f"r{tile_row_index:04d}_"
                    f"c{tile_column_index:04d}"
                )

                tile_rows.append({
                    "tile_id": tile_id,
                    "map_id": map_id,
                    "satellite_path": str(satellite_path),
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "north_lat": tile_north_lat,
                    "south_lat": tile_south_lat,
                    "west_lon": tile_west_lon,
                    "east_lon": tile_east_lon,
                    "center_lat": (
                        tile_north_lat + tile_south_lat
                    ) / 2,
                    "center_lon": (
                        tile_west_lon + tile_east_lon
                    ) / 2,
                })

    tiles_df = pd.DataFrame(tile_rows)

    if tiles_df.empty:
        raise ValueError("No satellite tiles were generated")

    tiles_df = tiles_df.astype({
        "tile_id": "string",
        "map_id": "string",
        "satellite_path": "string",
        "x_min": "int64",
        "y_min": "int64",
        "x_max": "int64",
        "y_max": "int64",
        "north_lat": "float64",
        "south_lat": "float64",
        "west_lon": "float64",
        "east_lon": "float64",
        "center_lat": "float64",
        "center_lon": "float64",
    })

    return tiles_df.set_index("tile_id", verify_integrity=True)

# Update every entry in the drone_dataframe "primary_tile_id" column
# Test that no entries are NA in the primary_tile_id col at the end
def map_images_to_tiles(drone_dataframe: pd.DataFrame,
                        tiles_dataframe: pd.DataFrame) -> None:
    
    tiles_by_map = {map_id: group for map_id, group in tiles_dataframe.groupby("map_id")}

    for image_id, drone_row in drone_dataframe.iterrows():
        map_id = drone_row["map_id"]

        if map_id not in tiles_by_map:
            raise ValueError(f"No tiles found for map {map_id}")

        map_tiles = tiles_by_map[map_id]

        containing_tiles = map_tiles[
            (map_tiles["south_lat"] <= drone_row["lat"])
            & (drone_row["lat"] <= map_tiles["north_lat"])
            & (map_tiles["west_lon"] <= drone_row["lon"])
            & (drone_row["lon"] <= map_tiles["east_lon"])
        ]

        if containing_tiles.empty:
            raise ValueError(
                f"No tile contains drone image {image_id}"
            )

        lat_difference = (
            containing_tiles["center_lat"] - drone_row["lat"]
        )

        longitude_scale = np.cos(
            np.radians(drone_row["lat"])
        )

        lon_difference = (
            containing_tiles["center_lon"] - drone_row["lon"]
        ) * longitude_scale

        distance_squared = (
            lat_difference**2 + lon_difference**2
        )

        primary_tile_id = distance_squared.idxmin()

        # Modifies the original DataFrame.
        drone_dataframe.at[
            image_id,
            "primary_tile_id",
        ] = primary_tile_id

    if drone_dataframe["primary_tile_id"].isna().any():
        raise ValueError(
            "Some drone images were not assigned a tile"
        )

def data_partition_TVT(drone_dataframe: pd.DataFrame,
                       tiles_dataframe: pd.DataFrame,
                       NUM_SATELLITE_MAPS: int,
                       train_ratio: float,
                       validation_ratio: float,
                       test_ratio: float,
                       rng: np.random.Generator) -> None:

    ratio_sums = train_ratio + validation_ratio + test_ratio

    if not math.isclose(1, ratio_sums):
        raise ValueError("Data split ratios do not sum to 1")

    if NUM_SATELLITE_MAPS < 3:
        raise ValueError("Insufficient number of satellite maps for model training")

    if set(drone_dataframe["map_id"]) != set(tiles_dataframe["map_id"]):
        raise ValueError("Drone and tiles datasets contain unmatched map IDs")

    map_ids = drone_dataframe["map_id"].unique().tolist()

    if len(map_ids) != NUM_SATELLITE_MAPS:
        raise ValueError("NUM_SATELLITE_MAPS does not match the number of unique map IDs")
    
    train_maps = []
    validation_maps = []
    test_maps = []

    shuffled_maps = list(rng.permutation(map_ids))
    train_maps.append(shuffled_maps[0])
    validation_maps.append(shuffled_maps[1])
    test_maps.append(shuffled_maps[2])

    for map_id in shuffled_maps[3::]:
        u = rng.random()
        if u <= train_ratio:
            train_maps.append(map_id)
        elif u <= train_ratio + validation_ratio:
            validation_maps.append(map_id)
        else:
            test_maps.append(map_id)

    # Update "dataset" cols in drone and tiles dataframes to be based on this split
    dataset_lookup = {
        **{map_id: "train" for map_id in train_maps},
        **{map_id: "validation" for map_id in validation_maps},
        **{map_id: "test" for map_id in test_maps},
    }

    drone_dataframe["dataset"] = (drone_dataframe["map_id"].map(dataset_lookup).astype("string"))
    tiles_dataframe["dataset"] = (tiles_dataframe["map_id"].map(dataset_lookup).astype("string"))

def check_data_leakage(drone_df: pd.DataFrame,
                       tiles_df: pd.DataFrame) -> None:
    
    # Get map_ids from drone training dataset
    train_maps = set(drone_df.loc[drone_df["dataset"] == "train", "map_id"])

    # Get map_ids from drone validation dataset
    validation_maps = set(drone_df.loc[drone_df["dataset"] == "validation", "map_id"])

    # Get map_ids from drone test dataset
    test_maps = set(drone_df.loc[drone_df["dataset"] == "test", "map_id"])

    # Ensure no map appears in multiple datasets
    if not (train_maps.isdisjoint(validation_maps) and train_maps.isdisjoint(test_maps) and validation_maps.isdisjoint(test_maps)):
        raise ValueError("Error: dataset partition failed, duplicate data across multiple datasets present")

    # Ensure no missing values in any of the datasets
    if any ([drone_df["dataset"].isna().any(), drone_df["primary_tile_id"].isna().any(), tiles_df["dataset"].isna().any()]):
        raise ValueError("Error: not all cells in drone_df dataset and primary_tile_id cols or tiles_df dataset col were filled in")

    # For each drone datapoint's primary_tile_id, look it up to ensure it is mapped correctly
    tile_splits = tiles_df["dataset"]
    drone_tile_splits = drone_df["primary_tile_id"].map(tile_splits)

    if not (drone_df["dataset"] == drone_tile_splits).all():
        raise ValueError("Error: drone tile mapping was invalid")


class DTCombinedDataset(Dataset):
    def __init__(self, drone_df: pd.DataFrame, 
                 tile_df: pd.DataFrame, 
                 load_image_tensor: Callable[[Path], torch.Tensor], 
                 model_input_size: int,
                 transform: Callable[[torch.Tensor], torch.Tensor] | None = None,):

        self.drone_df = drone_df.reset_index()
        self.tile_df = tile_df
        self.load_image_tensor = load_image_tensor
        self.model_input_size = model_input_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.drone_df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        drone_row = self.drone_df.iloc[index]
        tile_id = str(drone_row["primary_tile_id"])
        tile_row = self.tile_df.loc[tile_id]

        drone_tensor = self.load_image_tensor(Path(drone_row["image_path"]))

        satellite_tensor = self.load_image_tensor(Path(tile_row["satellite_path"]))

        x_min = int(tile_row["x_min"])
        x_max = int(tile_row["x_max"])
        y_min = int(tile_row["y_min"])
        y_max = int(tile_row["y_max"])

        output_size = (self.model_input_size, self.model_input_size)

        # [channels, height, width]
        tile_tensor = satellite_tensor[:, y_min:y_max, x_min:x_max]

        # Split drone images into 4 quadrants
        _, drone_height, drone_width = drone_tensor.shape

        middle_y = drone_height // 2
        middle_x = drone_width // 2

        quadrants = [drone_tensor[:, :middle_y, :middle_x],
                                       drone_tensor[:, :middle_y, middle_x:],
                                       drone_tensor[:, middle_y:, :middle_x],
                                       drone_tensor[:, middle_y:, middle_x:]]

        # F.interpolate expects a batch dimension:
        # [C, H, W] -> [1, C, H, W]
        # Standardize tensor sizes for model based in input param from config file

        # Splitting drone images into quadrants mitigates issue of info being lost
        # while balancing issue of high computational cost
        drone_quadrants = torch.stack([
            F.interpolate(
                quadrant.unsqueeze(0),
                size=output_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
            for quadrant in quadrants
        ])

        
        tile_tensor = F.interpolate(tile_tensor.unsqueeze(0), 
                                    size=output_size, 
                                    mode="bilinear", 
                                    align_corners=False,
                                    antialias=True).squeeze(0)

        # Optoinal transform feature to apply slight randomized changes to drone input images to reduce overfitting
        if self.transform is not None:
            # Apply independently to every quadrant.
            drone_quadrants = torch.stack([
                self.transform(quadrant)
                for quadrant in drone_quadrants
            ])

        return {"drone_tensor": drone_quadrants,
                "tile_tensor": tile_tensor,
                "image_id": str(drone_row["image_id"]),
                "tile_id": tile_id}

def load_datasets_TVT(drone_images_df: pd.DataFrame, tiles_df: pd.DataFrame, batch_size: int, model_input_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    # Create seperate dataframes for drone and tiles based on [train - validation - test]
    training_drone_df = drone_images_df[drone_images_df["dataset"] == "train"]
    validation_drone_df = drone_images_df[drone_images_df["dataset"] == "validation"]
    test_drone_df = drone_images_df[drone_images_df["dataset"] == "test"]

    training_tiles_df = tiles_df[tiles_df["dataset"] == "train"]
    validation_tiles_df = tiles_df[tiles_df["dataset"] == "validation"]
    test_tiles_df = tiles_df[tiles_df["dataset"] == "test"]

    # Set up transform function to apply random small transformations to images
    training_transform = v2.Compose([v2.RandomHorizontalFlip(p=0.5), 
                                     v2.RandomRotation(degrees=10),
                                     v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)])

    # Create overall datasets for train - validate - test
    training_dataset = DTCombinedDataset(training_drone_df, 
                                         training_tiles_df, 
                                         load_image_tensor, 
                                         model_input_size,
                                         transform = training_transform)

    validation_dataset = DTCombinedDataset(validation_drone_df, 
                                           validation_tiles_df, 
                                           load_image_tensor, 
                                           model_input_size,
                                           transform = None)
    
    test_dataset = DTCombinedDataset(test_drone_df, 
                                     test_tiles_df, 
                                     load_image_tensor, 
                                     model_input_size,
                                     transform = None)

    # Automate batching process using Pytorch DataLoader
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return training_loader, validation_loader, test_loader