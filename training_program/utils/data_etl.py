from pathlib import Path
import cv2
import torch
import numpy as np
from utils.images_utils import Satellite_Tile, Drone_Image
import pandas as pd
from PIL import Image

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
                        tiles_dataFrame: pd.DataFrame) -> None:
    pass
    