# 0001 — Tile and resize images efore encoding

**Status:** Implemented

## Context

We want to reduce the computational cost of encoding images during the training process and standardize sizes
for PyTorch DataLoader.

## Decision

For satellite images, break up the satellite image into many smaller tiles and shrink the size using bilinear interpolation.
For drone images, break up the image into 4 quadrants, and use bilinear interpolation to further shrink the image, and treat
the 4 resized quadrants as one unit.

## Consequences

Resizing reduces the number of input pixels and gives each input type fixed dimensions for batching.
Downsampling can blur or erase small landmarks that help distinguish nearby locations.