# Sharpness Map

This code calculates a sharpness map of an input image using the Discrete Cosine Transform (DCT) and edge-aware recursive filtering. The sharpness map highlights areas of the image with high-frequency components, which correspond to edges and fine details.

## Overview

The process involves the following steps:
1. **Preprocessing**: The input image is smoothed using a Gaussian blur to reduce noise.
2. **Gradient Calculation**: The Sobel operator is applied to calculate the gradient magnitude of the smoothed image.
3. **Downsampling**: The gradient image is downsampled to reduce computational load.
4. **DCT Calculation**: The DCT is computed for patches around each pixel in the downsampled image.
5. **High-Frequency Component Analysis**: High-frequency components of the DCT are analyzed to estimate local sharpness.
6. **Edge-Aware Filtering**: An edge-aware recursive filter is applied to smooth the sharpness map while preserving edges.

## Code Explanation

### Functions

#### `sharpness_map(img)`

The main function that generates the sharpness map of the input image.

```python
def sharpness_map(img):
dctmtx(n)
Generates an n x n DCT matrix.
    def dctmtx(n):
    [mesh_cols, mesh_rows] = np.meshgrid(np.linspace(0, n-1, n), np.linspace(0, n-1, n))
    dct_matrix = np.sqrt(2/n) * np.cos(np.pi * np.multiply((2 * mesh_cols + 1), mesh_rows) / (2*n))
    dct_matrix[0, :] = dct_matrix[0, :] / np.sqrt(2)
    return dct_matrix ```

hgvfj

