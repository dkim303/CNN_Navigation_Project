from pathlib import Path
import yaml
import argparse
import cv2
import torch
import numpy as np
from torch import nn
import os
import pandas as pd

class Residual_Block(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class Satellite_Vision_Model(nn.Module):
    def __init__(self, conv_layers: list[dict]):
        super().__init__()

        layers = []
        in_channels = 3

        for layer in conv_layers:
            out_channels = layer["filters"]
            kernel_size = layer["kernel_size"]

            layers.extend([nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                padding = kernel_size // 2,
                bias=False
            ), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            Residual_Block(out_channels)
            ])

            in_channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        final_channels = conv_layers[-1]["filters"]
        self.projection = nn.Linear(final_channels, final_channels)

    # Output should be a high dimensional vector
    def forward(self, x):
        x = self.layers(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.projection(x)
        return x


class Drone_Vision_Model(nn.Module):
    def __init__(self, conv_layers: list[dict]):
        super().__init__()

        layers = []
        in_channels = 3

        for layer in conv_layers:
            out_channels = layer["filters"]
            kernel_size = layer["kernel_size"]

            layers.extend([nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                padding = kernel_size // 2,
                bias=False
            ), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            Residual_Block(out_channels)
            ])

            in_channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        final_channels = conv_layers[-1]["filters"]
        self.projection = nn.Linear(final_channels, final_channels)

    # Output should be a high dimensional vector
    def forward(self, x):
        x = self.layers(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.projection(x)
        return x