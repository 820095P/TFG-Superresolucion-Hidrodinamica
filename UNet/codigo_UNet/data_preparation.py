import numpy as np
import os
import glob
import re
from scipy.interpolate import RectBivariateSpline
from sklearn.model_selection import train_test_split
from traitlets import Any
import torch.distributed as dist
import torch
import matplotlib.pyplot as plt
import ast
import numpy as np
import psutil
from dataclasses import dataclass
from typing import Optional, Any, Dict

#! def print_info_of_binary_image(file_path):


@dataclass(frozen=True)
class ModelConfig:
    DO_TRAINING: bool = False
    PATCH_BASED_MAPPING: bool = False
    NUMBER_OF_SUBDOMAINS: Optional[int] = None
    NUMBER_OF_COLUMNS_IN_SUBDOMAINS: Optional[int] = None

    TOPO_DATA_PATHS: Any = None
    Z_FINE_PATH: Optional[str] = None
    H_COARSE_DATA_PATH: Optional[str] = None
    H_FINE_DATA_PATH: Optional[str] = None
    WSE_COARSE_DATA_PATH: Optional[str] = None
    WSE_FINE_DATA_PATH: Optional[str] = None
    H_COARSE_REAL_CASE_DATA_PATH: Optional[str] = None
    H_FINE_REAL_CASE_DATA_PATH: Optional[str] = None

    FIRST_LAYER: str = "SG"
    HOTSTART: bool = False

    MODEL_SAVE_PATH: str = "models/Model.pth"
    NORM_PARAMS_SAVE_PATH: str = "models/norm_params.pt"
    VISUALIZATION_SAVE_PATH: str = "results/Model/Model"
    HISTORY_SAVE_PATH: str = "results/Model/Model"

    THRESHOLD: float = 0.001
    TOTAL_PIXELS_TO_CROP_ON_EACH_DIM: Optional[int] = None
    CELLSIZE_FINE: Optional[float] = None
    CELLSIZE_RATIO: Optional[int] = None
    CELLSIZE_RATIO_TRAIN: Optional[int] = None
    CELLSIZE_RATIO_REAL_CASE: Optional[int] = None
    GROUPING_PARAMETER_FOR_SUBGRID_ROWS: int = 1
    GROUPING_PARAMETER_FOR_SUBGRID_COLS: int = 1

    SEED: int = 795236
    CLIP_VALUE_MIN: Optional[float] = None
    CLIP_VALUE_MAX: Optional[float] = None

    LAST_ACTIVATION_FUNCTION: str = "nn.ReLU"
    ALPHA_FOR_LEAKY_RELU: float = 0.01

    BATCH_SIZE: int = 64
    EPOCHS: int = 1000
    INITIAL_LEARNING_RATE: float = 0.0001
    EARLY_STOPPING_PATIENCE: int = 25

    MASK_FINE_MAP_TO_THRESHOLD: bool = False
    REPRODUCIBILITY: bool = False
    LOG1P_EXP1M_TRANSFORM: bool = False

    SAVE_PREDICTIONS: bool = False
    PREDICTIONS_OF_REAL_CASE: bool = False
    VISUALIZE_PREDICTIONS: bool = False
    EVALUATION_DOMAIN_DEM_PATH: Optional[str] = None
    EVALUATION_DOMAIN_NODATA: Optional[float] = None

    WHICH_LOSS: str = "MAEMassConservationAndWetDryLoss"
    ALPHA_CSI: float = 0.0
    BETA_MASS: float = 0.1
    LAMBDA_DRY: float = 5.0
    ALPHA_CSI_MSE: float = 0.0
    OMEGA_MSE: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        fields = cls.__dataclass_fields__.keys()
        filtered = {k: d.get(k, getattr(cls, k)) for k in fields}
        return cls(**filtered)

def add_padding(x: torch.Tensor,
                     rows_top: int, rows_bottom: int,
                     cols_left: int, cols_right: int,
                     value: float = 0.0) -> torch.Tensor:
    """
    Añade padding constante a un tensor PyTorch en formato (B, H, W, C).

    Args:
        x: Tensor de entrada (B, H, W, C)
        rows_top, rows_bottom: padding en H
        cols_left, cols_right: padding en W
        value: valor del padding (por defecto 0.0)

    Returns:
        Tensor con padding, también en formato (B, H+top+bottom, W+left+right, C)
    """
    if x.ndim != 4:
        raise ValueError(f"Se esperaba un tensor 4D (B,H,W,C). Recibido shape: {x.shape}")
    if min(rows_top, rows_bottom, cols_left, cols_right) < 0:
        raise ValueError("Los paddings deben ser >= 0")

    x_bchw = x.permute(0, 3, 1, 2) # BHWC -> BCHW

    padding = (cols_left, cols_right, rows_top, rows_bottom)
    x_bchw_padded = torch.nn.functional.pad(x_bchw, padding, mode="constant", value=value)

    return x_bchw_padded.permute(0, 2, 3, 1) # BCHW -> BHWC

def modify_cfg(cfg_filepath, old_rows, old_cols, old_runoff, old_cellsize, new_rows, new_cols, new_runoff, new_cell_size, new_cfg_filepath):
    """
    Modify a configuration file to update dimensions and runoff values.

    Args:
        cfg_filepath (str): Path to the input configuration file.
        old_rows (int): Original number of rows in the DEM.
        old_cols (int): Original number of columns in the DEM.
        old_runoff (int): Original runoff value.
        old_cellsize (float): Original cell size.
        new_rows (int): New number of rows for the DEM.
        new_cols (int): New number of columns for the DEM.
        new_runoff (int): New runoff value.
        new_cell_size (float): New cell size.
        new_cfg_filepath (str): Path where the modified configuration file will be saved.

    Returns:
        None
    """

    if len({old_rows, old_cols, old_runoff, old_cellsize, new_rows, new_cols, new_runoff, new_cell_size}) < 8:
        raise ValueError("Warning: If you pass two or more identical numbers to the function, it will break. Please edit the configuration file manually in that case.")

    with open(cfg_filepath, 'r') as f:
        lines = f.readlines()

    new_lines = []
    count_rows, count_cols, count_runoff, count_cellsize = 0, 0, 0, 0
    for line in lines:
        new_line = line
        if f"{old_rows}" in new_line:
            new_line = new_line.replace(str(old_rows), str(new_rows), 1)
            count_rows += 1
            if count_rows > 3:
                raise ValueError(f"Warning: There are more than three lines with the number of rows {old_rows}. Please check the configuration file.")
        if f"{old_cols}" in new_line :
            new_line = new_line.replace(str(old_cols), str(new_cols), 1)
            count_cols += 1
            if count_cols > 3:
                raise ValueError(f"Warning: There are more than three lines with the number of columns {old_cols}. Please check the configuration file.")
        if f"{old_runoff}" in new_line:
            new_line = new_line.replace(str(old_runoff), str(new_runoff), 1)
            count_runoff += 1
            if count_runoff > 1:
                raise ValueError(f"Warning: There is more than one line with the number of runoff {old_runoff}. Please check the configuration file.")
        if f"{old_cellsize}" in new_line:
            new_line = new_line.replace(str(old_cellsize), str(new_cell_size), 1)
            count_cellsize += 1
            if count_cellsize > 3:
                raise ValueError(f"Warning: There are more than three lines with the cell size {old_cellsize}. Please check the configuration file.")
        new_lines.append(new_line)

    with open(new_cfg_filepath, 'w') as f:
        f.writelines(new_lines)
    
def grow_dem(dem_filepath, new_rows, new_cols, new_cell_size, new_dem_filepath):
    """
    Grow (expand) a DEM (Digital Elevation Model) file by adding rows and/or columns with bilinear interpolation.

    Args:
        dem_filepath (str): Path to the input DEM file.
        new_rows (int): Target number of rows after expansion.
        new_cols (int): Target number of columns after expansion.
        new_cell_size (float): New cell size for the output DEM.
        new_dem_filepath (str): Path where the new DEM file will be saved.

    Returns:
        final_rows (int): Final number of rows after rescaling.
        final_cols (int): Final number of columns after rescaling.
    """
    
    # Read DEM file and parse header
    with open(dem_filepath, 'r') as f:
        lines = f.readlines()
    
    header = {}
    header_lines = 6
    
    for i in range(header_lines):
        line = lines[i].strip().split()
        key = line[0].lower()
        value = float(line[1]) if key in ['xllcorner', 'yllcorner', 'cellsize', 'nodata_value'] else int(line[1])
        header[key] = value
    
    # Extract original parameters
    orig_ncols = header['ncols']
    orig_nrows = header['nrows']
    orig_cellsize = header['cellsize']
    xllcorner = header['xllcorner']
    yllcorner = header['yllcorner']
    nodata_value = header.get('nodata_value', -9999)
    
    # Read DEM data
    data_lines = lines[header_lines:]
    dem_data = np.array([list(map(float, line.strip().split())) for line in data_lines])
    
    # Ensure we're expanding, not shrinking
    if new_rows < orig_nrows or new_cols < orig_ncols:
        print(f"Warning: new dimensions ({new_rows}x{new_cols}) are smaller than original ({orig_nrows}x{orig_ncols})")
        print("This function is designed for expansion. Consider using crop_and_rescale_dem instead.")
    
    # Calculate how many rows/cols to add
    rows_to_add = max(0, new_rows - orig_nrows)
    cols_to_add = max(0, new_cols - orig_ncols)
    
    # Expand the DEM by padding with edge values ONLY at bottom and right
    if rows_to_add > 0 or cols_to_add > 0:
        # Padding only at bottom and right (no top, no left)
        pad_top = 0
        pad_bottom = rows_to_add
        pad_left = 0
        pad_right = cols_to_add
        
        # Pad the DEM using 'edge' mode:
        # 'edge' repeats the border values.
        # It extends the last column/row values into the new padding area
        expanded_dem = np.pad(dem_data, 
                             ((pad_top, pad_bottom), (pad_left, pad_right)), 
                             mode='edge')
        
        # Coordinate system adjustment:
        # Since we only pad bottom and right, coordinates don't change
        # (we're not moving the top-left origin)
        new_yllcorner = yllcorner - rows_to_add*orig_cellsize  # No change since no top padding
        new_xllcorner = xllcorner  # No change since no left padding
    else:
        expanded_dem = dem_data
        new_yllcorner = yllcorner
        new_xllcorner = xllcorner
    
    expanded_rows, expanded_cols = expanded_dem.shape
    
    # Calculate scaling factor based on cell sizes
    scale_factor = orig_cellsize / new_cell_size
    
    # Calculate final dimensions after resampling
    final_rows = int(expanded_rows * scale_factor)
    final_cols = int(expanded_cols * scale_factor)
    
    # Bilinear interpolation using scipy if rescaling is needed
    if scale_factor != 1.0:
        # Create coordinate arrays for original data
        x_orig = np.arange(expanded_cols)
        y_orig = np.arange(expanded_rows)
        
        # Create coordinate arrays for new data
        x_new = np.linspace(0, expanded_cols - 1, final_cols)
        y_new = np.linspace(0, expanded_rows - 1, final_rows)
        
        # Create interpolation function
        interpolator = RectBivariateSpline(y_orig, x_orig, expanded_dem, kx=1, ky=1)
        
        # Apply interpolation
        rescaled_dem = interpolator(y_new, x_new)
        
        # Adjust cell size and coordinates for the final rescaling
        final_xllcorner = new_xllcorner
        final_yllcorner = new_yllcorner
    else:
        rescaled_dem = expanded_dem
        final_xllcorner = new_xllcorner
        final_yllcorner = new_yllcorner
    
    # Update header for the new DEM
    new_header = {
        'ncols': final_cols,
        'nrows': final_rows,
        'xllcorner': final_xllcorner,
        'yllcorner': final_yllcorner,
        'cellsize': new_cell_size,
        'nodata_value': int(nodata_value)
    }
    
    # Write the new DEM file
    with open(new_dem_filepath, 'w') as f:
        # Write header
        f.write(f"ncols         {new_header['ncols']}\n")
        f.write(f"nrows         {new_header['nrows']}\n")
        f.write(f"xllcorner     {new_header['xllcorner']}\n")
        f.write(f"yllcorner     {new_header['yllcorner']}\n")
        f.write(f"cellsize      {new_header['cellsize']}\n")
        f.write(f"NODATA_value  {new_header['nodata_value']}\n")
        
        # Write data
        for row in rescaled_dem:
            f.write(' '.join([f"{val:.6f}" if val != nodata_value else str(int(nodata_value)) 
                             for val in row]) + '\n')
            
    print(f"New expanded and rescaled DEM file saved to {new_dem_filepath}")
    print(f"Original dimensions: {orig_nrows}x{orig_ncols}")
    print(f"Expanded dimensions: {expanded_rows}x{expanded_cols}")
    print(f"Final dimensions: {final_rows}x{final_cols}")

    return final_rows, final_cols

def crop_and_rescale_dem(dem_filepath, new_cropped_rows, new_cropped_cols, new_cell_size, new_dem_filepath):
    """
    Crop and rescale a DEM (Digital Elevation Model) file with bilinear interpolation.

    Args:
        dem_filepath (str): Path to the input DEM file.
        new_cropped_rows (int): Number of rows to keep after cropping.
        new_cropped_cols (int): Number of columns to keep after cropping.
        new_cell_size (float): New cell size for the output DEM.
        new_dem_filepath (str): Path where the new DEM file will be saved.

    Returns:
        final_rows (int): Final number of rows after rescaling.
        final_cols (int): Final number of columns after rescaling.
    """
    
    # Read DEM file and parse header
    with open(dem_filepath, 'r') as f:
        lines = f.readlines()
    
    header = {}
    header_lines = 6
    
    for i in range(header_lines):
        line = lines[i].strip().split()
        key = line[0].lower()
        value = float(line[1]) if key in ['xllcorner', 'yllcorner', 'cellsize', 'nodata_value'] else int(line[1])
        header[key] = value
    
    # Extract original parameters
    orig_ncols = header['ncols']
    orig_nrows = header['nrows']
    orig_cellsize = header['cellsize']
    xllcorner = header['xllcorner']
    yllcorner = header['yllcorner']
    nodata_value = header.get('nodata_value', -9999)
    
    # Read DEM data
    data_lines = lines[header_lines:]
    dem_data = np.array([list(map(float, line.strip().split())) for line in data_lines])
    
    # Crop the DEM data first
    cropped_dem = dem_data[:min(new_cropped_rows, orig_nrows), :min(new_cropped_cols, orig_ncols)]
    crop_rows, crop_cols = cropped_dem.shape

    # Calculate how many rows were removed from the bottom (for yllcorner adjustment)
    rows_removed_from_bottom = orig_nrows - crop_rows
    # Adjust yllcorner: if we removed rows from bottom, move yllcorner up
    new_yllcorner = yllcorner + (rows_removed_from_bottom * orig_cellsize)
    
    # Calculate scaling factor based on cell sizes
    scale_factor = orig_cellsize / new_cell_size
    
    # Calculate new dimensions after resampling
    final_rows = int(crop_rows * scale_factor)
    final_cols = int(crop_cols * scale_factor)
    
    # Bilinear interpolation using scipy
    if scale_factor != 1.0:
        # Create coordinate arrays for original data
        x_orig = np.arange(crop_cols)
        y_orig = np.arange(crop_rows)
        
        # Create coordinate arrays for new data
        x_new = np.linspace(0, crop_cols - 1, final_cols)
        y_new = np.linspace(0, crop_rows - 1, final_rows)
        
        # Create interpolation function
        interpolator = RectBivariateSpline(y_orig, x_orig, cropped_dem, kx=1, ky=1) # RectBivariateSpline expects (y, x) order
        
        # Apply interpolation
        rescaled_dem = interpolator(y_new, x_new)
    else:
        rescaled_dem = cropped_dem
    
    # Update header for the new DEM
    new_header = {
        'ncols': final_cols,
        'nrows': final_rows,
        'xllcorner': xllcorner,
        'yllcorner': new_yllcorner,
        'cellsize': new_cell_size,
        'nodata_value': int(nodata_value)
    }
    
    # Write the new DEM file
    with open(new_dem_filepath, 'w') as f:
        # Write header
        f.write(f"ncols         {new_header['ncols']}\n")
        f.write(f"nrows         {new_header['nrows']}\n")
        f.write(f"xllcorner     {new_header['xllcorner']}\n")
        f.write(f"yllcorner     {new_header['yllcorner']}\n")
        f.write(f"cellsize      {new_header['cellsize']}\n")
        f.write(f"NODATA_value  {new_header['nodata_value']}\n")
        
        # Write data
        for row in rescaled_dem:
            f.write(' '.join([f"{val:.6f}" if val != nodata_value else str(int(nodata_value)) 
                             for val in row]) + '\n')
            
    print(f"New cropped and rescaled DEM file saved to {new_dem_filepath}")

    return final_rows, final_cols

def grow_mann(mann_filepath, new_rows, new_cols, final_rows, final_cols, new_mann_filepath):
    """
    Grow (expand) and then rescale a Manning file with bilinear interpolation.
    
    This function performs two steps:
    1. Grow: Expands the original file by adding rows/cols (padding only at bottom and right)
    2. Rescale: Resamples the expanded data to final_rows x final_cols using bilinear interpolation

    Args:
        mann_filepath (str): Path to the input Manning file.
        new_rows (int): Target number of rows after expansion.
        new_cols (int): Target number of columns after expansion.
        final_rows (int): Final number of rows after rescaling.
        final_cols (int): Final number of columns after rescaling.
        new_mann_filepath (str): Path where the new Manning file will be saved.

    Returns:
        None
    """
    
    # Read Manning file
    with open(mann_filepath, 'r') as f:
        lines = f.readlines()

    # Read Manning data (all of it)
    mann_data = np.array([list(map(float, line.strip().split())) for line in lines])
    orig_nrows, orig_ncols = mann_data.shape

    # Ensure we're expanding, not shrinking
    if new_rows < orig_nrows or new_cols < orig_ncols:
        print(f"Warning: new dimensions ({new_rows}x{new_cols}) are smaller than original ({orig_nrows}x{orig_ncols})")
        print("This function is designed for expansion. Consider using crop_and_rescale_mann instead.")

    # STEP 1: Expand the Manning data by padding ONLY at bottom and right
    rows_to_add = max(0, new_rows - orig_nrows)
    cols_to_add = max(0, new_cols - orig_ncols)
    
    if rows_to_add > 0 or cols_to_add > 0:
        # Padding only at bottom and right (no top, no left)
        pad_top = 0
        pad_bottom = rows_to_add
        pad_left = 0
        pad_right = cols_to_add
        
        # Pad the Manning data using 'edge' mode:
        # 'edge' repeats the border values. For Manning coefficients, this makes sense
        # as roughness values tend to have spatial continuity
        expanded_mann = np.pad(mann_data, 
                              ((pad_top, pad_bottom), (pad_left, pad_right)), 
                              mode='edge')
    else:
        expanded_mann = mann_data
    
    expanded_rows, expanded_cols = expanded_mann.shape
    
    # STEP 2: Rescale the expanded data to final dimensions
    if expanded_rows != final_rows or expanded_cols != final_cols:
        # Create coordinate arrays for expanded data
        x_orig = np.arange(expanded_cols)
        y_orig = np.arange(expanded_rows)
        
        # Create coordinate arrays for final dimensions
        x_new = np.linspace(0, expanded_cols - 1, final_cols)
        y_new = np.linspace(0, expanded_rows - 1, final_rows)

        # Create interpolation function
        # RectBivariateSpline expects (y, x) order
        interpolator = RectBivariateSpline(y_orig, x_orig, expanded_mann, kx=1, ky=1)

        # Apply interpolation to get final dimensions
        final_mann = interpolator(y_new, x_new)
        
    else:
        # If expanded dimensions equal final dimensions, no need to interpolate
        final_mann = expanded_mann

    # Write the new Manning file
    with open(new_mann_filepath, 'w') as f:
        for row in final_mann:
            f.write(' '.join([f"{val:.6f}" for val in row]) + '\n')

    print(f"New expanded and rescaled Manning file saved to {new_mann_filepath}")
    print(f"Original dimensions: {orig_nrows}x{orig_ncols}")
    print(f"Expanded dimensions: {expanded_rows}x{expanded_cols}")
    print(f"Final dimensions: {final_rows}x{final_cols}")

def crop_and_rescale_mann(mann_filepath, new_cropped_rows, new_cropped_cols, final_rows, final_cols, new_mann_filepath):
    """
    Crop and then rescale a Manning file with bilinear interpolation.
    
    This function performs two steps:
    1. Crop: Takes only the first new_cropped_rows and new_cropped_cols from the original file
    2. Rescale: Resamples the cropped data to final_rows x final_cols using bilinear interpolation

    Args:
        mann_filepath (str): Path to the input Manning file.
        new_cropped_rows (int): Number of rows to keep after cropping.
        new_cropped_cols (int): Number of columns to keep after cropping.
        final_rows (int): Final number of rows after rescaling.
        final_cols (int): Final number of columns after rescaling.
        new_mann_filepath (str): Path where the new Manning file will be saved.

    Returns:
        None
    """
    
    # Read Manning file
    with open(mann_filepath, 'r') as f:
        lines = f.readlines()

    # Read Manning data (all of it)
    mann_data = np.array([list(map(float, line.strip().split())) for line in lines])
    orig_nrows, orig_ncols = mann_data.shape

    # STEP 1: Crop the Manning data
    cropped_mann = mann_data[:min(new_cropped_rows, orig_nrows), :min(new_cropped_cols, orig_ncols)]
    crop_rows, crop_cols = cropped_mann.shape
    
    # STEP 2: Rescale the cropped data to final dimensions
    if crop_rows != final_rows or crop_cols != final_cols:
        # Create coordinate arrays for cropped data
        x_orig = np.arange(crop_cols)
        y_orig = np.arange(crop_rows)
        
        # Create coordinate arrays for final dimensions
        x_new = np.linspace(0, crop_cols - 1, final_cols)
        y_new = np.linspace(0, crop_rows - 1, final_rows)

        # Create interpolation function
        # RectBivariateSpline expects (y, x) order
        interpolator = RectBivariateSpline(y_orig, x_orig, cropped_mann, kx=1, ky=1)

        # Apply interpolation to get final dimensions
        final_mann = interpolator(y_new, x_new)
        
    else:
        # If cropped dimensions equal final dimensions, no need to interpolate
        final_mann = cropped_mann

    # Write the new Manning file
    with open(new_mann_filepath, 'w') as f:
        for row in final_mann:
            f.write(' '.join([f"{val:.6f}" for val in row]) + '\n')

    print(f"New cropped and rescaled Manning file saved to {new_mann_filepath}")

def crop_runoff_map(rmap_filepath, new_cropped_rows, new_cropped_cols, new_rmap_filepath):
    """
    Crop a runoff map file to specified dimensions.

    Args:
        rmap_filepath (str): Path to the input runoff map file.
        new_cropped_rows (int): Number of rows to keep after cropping.
        new_cropped_cols (int): Number of columns to keep after cropping.
        new_rmap_filepath (str): Path where the new cropped runoff map file will be saved.

    Returns:
        None
    """
    
    # Read runoff map file
    with open(rmap_filepath, 'r') as f:
        lines = f.readlines()
    
    # Crop the runoff map data
    rmap_data = np.array([list(map(int, line.strip().split())) for line in lines])
    orig_nrows, orig_ncols = rmap_data.shape
    
    cropped_rmap = rmap_data[:min(new_cropped_rows, orig_nrows), :min(new_cropped_cols, orig_ncols)]

    # Write the new cropped runoff map file
    with open(new_rmap_filepath, 'w') as f:
        for row in cropped_rmap:
            f.write(' '.join([str(val) for val in row]) + '\n')

    print(f"New cropped runoff map saved to {new_rmap_filepath}")

def create_runoff_map(rows, cols, n_runoff, new_rmap_filepath):
    """
    Create a uniform runoff map file with specified dimensions and runoff value.
    
    This function creates a text file with 'rows' number of rows and 'cols' number of columns,
    where every cell contains the same runoff value 'n_runoff'.

    Args:
        rows (int): Number of rows in the runoff map.
        cols (int): Number of columns in the runoff map.
        n_runoff (int): Uniform runoff value to fill the entire map.
        new_rmap_filepath (str): Path where the new runoff map file will be saved.

    Returns:
        None
    """

    row_string = ' '.join([str(int(n_runoff))] * cols) + '\n'

    with open(new_rmap_filepath, 'w') as f:
        for _ in range(rows):
            f.write(row_string)
    
    print(f"Runoff map created and saved to: {new_rmap_filepath}")    

def make_patches(full_filepath, desired_H, desired_W, stride_cells, extra_cells, chunk_filepath):
    """
    Divide un archivo de datos en patches con overlap y los guarda como archivos separados.
    
    Args:
        full_filepath (str): Ruta al archivo que se quiere dividir
        desired_H (int): Altura deseada de los patches
        desired_W (int): Anchura deseada de los patches
        stride_cells (int): Stride (paso) para mover la ventana de extracción (mismo en ambos ejes)
        extra_cells (int): Celdas extra a añadir al tamaño del patch
        chunk_filepath (str): Ruta base para guardar los patches (sin extensión)
    """
    
    # Leer el archivo
    with open(full_filepath, 'r') as f:
        lines = f.readlines()
    
    # Si es un archivo DEM o slope o catchment, saltarse las 6 primeras líneas (header)
    if 'dem' in full_filepath.lower() or 'slope' in full_filepath.lower() or 'catch' in full_filepath.lower():
        lines = lines[6:]
    
    data = []
    for line in lines:
        row = [float(x) for x in line.strip().split()]
        data.append(row)
    
    data = np.array(data)
    original_H, original_W = data.shape
    
    chunk_counter = 1
    
    # Empezamos desde (stride_cells, stride_cells) para tener margen por arriba y por la izquierda
    row = stride_cells
    while row + desired_H + extra_cells <= original_H:
        col = stride_cells
        while col + desired_W + extra_cells <= original_W:
            # Extraer el chunk (incluyendo el margen)
            chunk_start_row = row - extra_cells
            chunk_end_row = row + desired_H + extra_cells
            chunk_start_col = col - extra_cells
            chunk_end_col = col + desired_W + extra_cells
            
            chunk = data[chunk_start_row:chunk_end_row, chunk_start_col:chunk_end_col]
            
            # Guardar el chunk
            chunk_filename = f"{chunk_filepath}{chunk_counter}.txt"
            
            # Crear el directorio si no existe
            os.makedirs(os.path.dirname(chunk_filename), exist_ok=True)
            
            # Guardar como archivo de texto
            np.savetxt(chunk_filename, chunk, fmt='%.6f', delimiter=' ')
            
            print(f"Guardado chunk {chunk_counter}: {chunk_filename} (shape: {chunk.shape})")
            chunk_counter += 1
            
            # Moverse hacia la derecha con stride desired_W
            col += desired_W
        
        # Moverse hacia abajo con stride desired_H
        row += desired_H
    
    print(f"Total de chunks creados: {chunk_counter - 1}")

def create_wse_patch_from_h_and_dem_patches(h_patch_filepath, dem_patch_filepath, wse_patch_filepath):
    """
    Crea patches de WSE (Water Surface Elevation) a partir de patches de altura (H) y DEM.
    
    Args:
        h_patch_filepath (str): Ruta al archivo del patch de altura (H).
        dem_patch_filepath (str): Ruta al archivo del patch de DEM (Z).
        wse_patch_filepath (str): Ruta base para guardar el patch de WSE (H+Z).
    """
    
    h_patches = np.loadtxt(h_patch_filepath)
    dem_patches = np.loadtxt(dem_patch_filepath)
    
    if h_patches.shape != dem_patches.shape:
        raise ValueError(f"Los parches de H y DEM deben tener las mismas dimensiones, pero tienen {h_patches.shape} y {dem_patches.shape} respectivamente.")
    
    wse_patches = h_patches + dem_patches
    
    np.savetxt(wse_patch_filepath, wse_patches, fmt='%.6f', delimiter=' ')

    print(f"Parches de WSE guardados en: {wse_patch_filepath}")

def natural_key(s):
    """Convierte una cadena en lista de números y texto para orden natural."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', os.path.basename(s))]

def load_info_model(config_path):
    config = {}
    with open(config_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # Pasamos de líneas vacías o comentarios
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            try:
                config[key] = ast.literal_eval(value)
            except Exception:
                config[key] = value  # Si no se puede evaluar (ej. string sin comillas), se guarda tal cual
    return config

def load_evaluation_domain_mask(dem_path=None, z_fine_path=None, nodata_override=None):
    """Construye la máscara fina desde DEM ASCII o reutiliza Z_FINE_PATH."""
    mask = None
    used_path = dem_path

    if dem_path is not None:
        if not os.path.isfile(dem_path):
                raise ValueError(f"Evaluation-domain DEM not found: {dem_path}")

        with open(dem_path, "r") as f:
                lines = f.readlines()
        if len(lines) < 6:
                raise ValueError(f"Evaluation-domain DEM has no valid six-line header: {dem_path}")

        header = {}
        for line in lines[:6]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    header[parts[0].lower()] = parts[1]

        try:
                ncols = int(header["ncols"])
                nrows = int(header["nrows"])
        except KeyError as exc:
                raise ValueError(f"Evaluation-domain DEM header is missing ncols/nrows: {dem_path}") from exc

        nodata_key = next((k for k in header if k == "nodata_value"), None)
        nodata = float(nodata_override) if nodata_override is not None else (
                float(header[nodata_key]) if nodata_key is not None else None
        )
        if nodata is None:
                raise ValueError(
                    f"NODATA desconocido para la máscara geográfica; falta NODATA_value "
                    f"en la cabecera y EVALUATION_DOMAIN_NODATA no está configurado: {dem_path}"
                )

        dem = np.loadtxt(lines[6:], dtype=np.float64)
        if dem.shape != (nrows, ncols):
                raise ValueError(
                    f"Tamaño incompatible en la máscara geográfica: matriz {dem.shape}, "
                    f"cabecera {nrows} filas x {ncols} columnas ({dem_path})"
                )
        mask = np.isfinite(dem) & (dem != nodata)
    else:
        if not z_fine_path:
                raise ValueError(
                    "No se puede activar la máscara geográfica: configura "
                    "EVALUATION_DOMAIN_DEM_PATH o Z_FINE_PATH."
                )
        used_path = z_fine_path
        files = []
        if os.path.isdir(z_fine_path):
                files = glob.glob(os.path.join(z_fine_path, "*.txt"))
                files.sort(key=natural_key)
        elif os.path.isfile(z_fine_path):
                files = [z_fine_path]

        if not files:
                raise ValueError(
                    f"Z_FINE_PATH no contiene archivos TXT utilizables para inferir la máscara: {z_fine_path}"
                )

        nodata = float(nodata_override) if nodata_override is not None else -9999.0
        found_nodata = False
        masks = []
        dimensions = None
        for file in files:
                dem = np.loadtxt(file, dtype=np.float64)
                if dem.shape == dimensions and dimensions is not None:
                    pass
                elif dimensions is None:
                    dimensions = dem.shape
                elif dem.shape != dimensions:
                    raise ValueError(
                        f"Tamaño incompatible dentro de Z_FINE_PATH: {file} tiene "
                        f"{dem.shape}, pero se esperaba {dimensions}"
                    )
                if np.any(dem == nodata):
                    found_nodata = True
                masks.append(np.isfinite(dem) & (dem != nodata))

        if not found_nodata:
                raise ValueError(
                    "Z_FINE_PATH no contiene ningún -9999: no se puede confirmar automáticamente "
                    "el contorno. Configura EVALUATION_DOMAIN_DEM_PATH con la ruta al DEM original."
                )

        reference = masks[0]
        for i, candidate in enumerate(masks[1:], start=1):
                if not np.array_equal(candidate, reference):
                    raise ValueError(
                        f"Las máscaras de Z_FINE_PATH no coinciden: el primer DEM difiere "
                        f"del archivo {files[i]}"
                    )
        mask = reference
        nrows, ncols = dimensions

    if mask is None:
        return None
    if mask.ndim != 2:
        raise ValueError(f"La máscara geográfica debe ser 2D; se recibió {mask.shape}")
    valid_count = int(mask.sum())
    if valid_count == 0:
        raise ValueError(f"La máscara geográfica está vacía (0 celdas válidas): {used_path}")
    nrows, ncols = mask.shape
    print(
        f"Dominio de evaluación: máscara geográfica | ruta usada: {used_path} | "
        f"dimensiones: {nrows} filas x {ncols} columnas | "
        f"celdas válidas por campo: {valid_count}"
    )
    return mask

def load_coarse_simulation_data(filepath):
    """
    Loads and stacks all coarse simulation data files.
    
    Args:
        filepath (str): Path to the coarse data files
        
    Returns:
        torch.Tensor: Tensor of shape (N_coarse, H_coarse, W_coarse, 1)
    """
    files = glob.glob(os.path.join(filepath, "*.txt"))
    files.sort(key=natural_key) # orden natural

    data_list = []
    
    for file in files:
        data = np.loadtxt(file, dtype=np.float32) # Ensure float32 for PyTorch compatibility
        data = np.expand_dims(data, axis=-1) # Add channel dimension (H, W) -> (H, W, 1)
        data_list.append(data)
    
    data_array = np.stack(data_list, axis=0) # (N_coarse, H, W, 1)
    
    print(f"Coarse simulation data shape: {data_array.shape}")
    return torch.from_numpy(data_array).float() # Convert to PyTorch Tensor

def load_fine_simulation_data(filepath):
    """
    Loads and stacks all fine simulation data files.
    
    Args:
        filepath (str): Path to the fine data files
        
    Returns:
        torch.Tensor: Tensor of shape (N_fine, H_fine, W_fine, 1)
    """
    files = glob.glob(os.path.join(filepath, "*.txt"))
    files.sort(key=natural_key) # orden natural
    
    data_list = []
    
    for file in files:
        data = np.loadtxt(file, dtype=np.float32)
        data = np.expand_dims(data, axis=-1) # Add channel dimension (H_fine, W_fine) -> (H_fine, W_fine, 1)
        data_list.append(data)
    
    data_array = np.stack(data_list, axis=0) # (N_fine, H_fine, W_fine, 1)
    
    print(f"Fine simulation data shape: {data_array.shape}")
    return torch.from_numpy(data_array).float() # Convert to PyTorch Tensor

def load_z_fine_data(filepath):
    """
    Loads and stacks all z_fine simulation data files.
    
    Args:
        filepath (str): Path to the z_fine data files
        
    Returns:
        torch.Tensor: Tensor of shape (N_fine, H_fine, W_fine, 1)
    """
    files = glob.glob(os.path.join(filepath, "*.txt"))
    files.sort(key=natural_key) # orden natural

    data_list = []
    
    for file in files:
        data = np.loadtxt(file, dtype=np.float32)
        data = np.expand_dims(data, axis=-1) # Add channel dimension (H_fine, W_fine) -> (H_fine, W_fine, 1)
        data_list.append(data)
    
    data_array = np.stack(data_list, axis=0) # (N_fine, H_fine, W_fine, 1)
    
    print(f"z_fine simulation data shape: {data_array.shape}")
    return torch.from_numpy(data_array).float() # Convert to PyTorch Tensor
    
def load_fine_topographic_features(filepaths_list):
    """
    Loads and stacks all fine topographic feature files.
    
    Args:
        filepaths_list (list): List of paths, one for each topographic channel
        
    Returns:
        torch.Tensor: Tensor of shape (N_topo, H_topo, W_topo, N_topo_channels)
    """
    all_channels_data = []
    
    for channel_idx, filepath in enumerate(filepaths_list):
        files = glob.glob(os.path.join(filepath, "*.txt"))
        files.sort(key=natural_key) # orden natural

        channel_data = []
        
        for file in files:
            data = np.loadtxt(file, dtype=np.float32)
            channel_data.append(data)
        
        channel_array = np.stack(channel_data, axis=0) # (N_topo, H_topo, W_topo)
        all_channels_data.append(channel_array)
    
    # Stack all channels: (N_topo_channels, N_topo, H_topo, W_topo)
    all_data = np.stack(all_channels_data, axis=0)
    
    # Reorganize to (N_topo, H_topo, W_topo, N_topo_channels)
    all_data = np.transpose(all_data, (1, 2, 3, 0))
    
    print(f"Fine topographic features shape: {all_data.shape}")
    return torch.from_numpy(all_data).float() # Convert to PyTorch Tensor

def split_dataset(coarse_data, fine_data, topo_data, z_fine_data, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_state=42):
    """
    Splits datasets into train, validation, and test while maintaining consistency.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum 1"
    
    n_coarse = coarse_data.shape[0]
    n_fine = fine_data.shape[0]
    n_topo = topo_data.shape[0]
    n_z_fine = z_fine_data.shape[0]

    assert n_coarse == n_fine, f"Coarse data({n_coarse} samples) and Fine data({n_fine} samples) must have the same number of samples"
    assert n_coarse % n_topo == 0, f"N_coarse ({n_coarse}) must be a multiple of N_topo ({n_topo})"
    assert n_topo == n_z_fine, f"Topographic features data({n_topo} samples) and z_fine data({n_z_fine} samples) must have the same number of samples"
    
    indices = np.arange(n_coarse)
    
    train_indices, temp_indices = train_test_split(
        indices, 
        train_size=train_ratio, 
        random_state=random_state
    )
    
    val_size_adjusted = val_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices, 
        train_size=val_size_adjusted, 
        random_state=random_state
    )
    
    train_topo_indices = train_indices % n_topo
    val_topo_indices = val_indices % n_topo
    test_topo_indices = test_indices % n_topo

    train_z_fine_indices = train_indices % n_z_fine
    val_z_fine_indices = val_indices % n_z_fine
    test_z_fine_indices = test_indices % n_z_fine

    splits = {
        'coarse': {
            'train': coarse_data[train_indices],
            'val': coarse_data[val_indices],
            'test': coarse_data[test_indices]
        },
        'fine': {
            'train': fine_data[train_indices],
            'val': fine_data[val_indices],
            'test': fine_data[test_indices]
        },
        'topo': {
            'train': topo_data[train_topo_indices],
            'val': topo_data[val_topo_indices],
            'test': topo_data[test_topo_indices]
        },
        'z_fine': {
            'train': z_fine_data[train_z_fine_indices],
            'val': z_fine_data[val_z_fine_indices],
            'test': z_fine_data[test_z_fine_indices]
        }
    }
    
    return splits

def get_files_sorted(folder):
    files = glob.glob(os.path.join(folder, "*.txt"))
    files.sort(key=natural_key)
    return files

def _index_synthetic_paths(folder, label):
    paths_by_key = {}
    filename_pattern = re.compile(r"^Sint(\d+)_H_(\d+)_(\d+)\.txt$")

    for path in get_files_sorted(folder):
        basename = os.path.basename(path)
        match = filename_pattern.fullmatch(basename)
        if match is None:
            raise ValueError(f"Unexpected synthetic filename in {label}: {path}")

        scenario, output, timestep = map(int, match.groups())
        if scenario not in {1, 2, 3, 4, 5}:
            raise ValueError(
                f"Unexpected scenario Sint{scenario} in {label}: {path}"
            )

        key = (scenario, output, timestep)
        if key in paths_by_key:
            raise ValueError(
                f"Duplicate synthetic key Sint{scenario}_H_{output}_{timestep} "
                f"in {label}: {paths_by_key[key]} and {path}"
            )
        paths_by_key[key] = path

    return paths_by_key


#! Pass only data paths, not loaded data, to avoid overloading GPUs.
def load_and_split_all_data_paths(coarse_folder, fine_folder, topo_folders):
    coarse_by_key = _index_synthetic_paths(coarse_folder, "LR")
    fine_by_key = _index_synthetic_paths(fine_folder, "HR")

    missing_fine_keys = sorted(coarse_by_key.keys() - fine_by_key.keys())
    if missing_fine_keys:
        scenario, output, timestep = missing_fine_keys[0]
        raise ValueError(
            f"No HR file for key Sint{scenario}_H_{output}_{timestep}; "
            f"LR file: {coarse_by_key[missing_fine_keys[0]]}"
        )

    missing_coarse_keys = sorted(fine_by_key.keys() - coarse_by_key.keys())
    if missing_coarse_keys:
        scenario, output, timestep = missing_coarse_keys[0]
        raise ValueError(
            f"No LR file for key Sint{scenario}_H_{output}_{timestep}; "
            f"HR file: {fine_by_key[missing_coarse_keys[0]]}"
        )

    expected_scenarios = {1, 2, 3, 4, 5}
    for label, folder, paths_by_key in (
        ("LR", coarse_folder, coarse_by_key),
        ("HR", fine_folder, fine_by_key),
    ):
        present_scenarios = {key[0] for key in paths_by_key}
        missing_scenarios = sorted(expected_scenarios - present_scenarios)
        if missing_scenarios:
            raise ValueError(
                f"Missing scenarios in {label} directory {folder}: "
                f"{', '.join(f'Sint{scenario}' for scenario in missing_scenarios)}"
            )

    ordered_keys = sorted(coarse_by_key)
    paired_paths = [
        (coarse_by_key[key], fine_by_key[key])
        for key in ordered_keys
    ]
    coarse_paths = [coarse_path for coarse_path, _ in paired_paths]
    fine_paths = [fine_path for _, fine_path in paired_paths]

    scenario_splits = {
        "train": {1, 2, 5},
        "val": {4},
        "test": {3},
    }
    splits = {
        split_name: np.asarray(
            [index for index, key in enumerate(ordered_keys) if key[0] in scenarios],
            dtype=np.int64,
        )
        for split_name, scenarios in scenario_splits.items()
    }

    print("Validated LR/HR pairs by split and scenario:")
    for split_name, scenarios in scenario_splits.items():
        split_keys = {ordered_keys[index] for index in splits[split_name]}
        for scenario in sorted(scenarios):
            scenario_keys = [
                key for key in ordered_keys if key[0] == scenario and key in split_keys
            ]
            print(f"  {split_name} / Sint{scenario}: {len(scenario_keys)} pairs")
            for key in scenario_keys:
                print(
                    f"    {os.path.basename(coarse_by_key[key])} -> "
                    f"{os.path.basename(fine_by_key[key])}"
                )

    paths = {
        'coarse_paths': coarse_paths,
        'fine_paths': fine_paths,
        'topo_folders': topo_folders,
    }

    return splits, paths

def load_split(split_indices, paths, world_size, rank, patch_based_mapping):
    local_idx = np.array_split(split_indices, world_size)[rank]
    coarse, topo, y = load_local_data(paths, local_idx, patch_based_mapping=patch_based_mapping)
    return (coarse, topo), y

def load_local_data(paths, indices, dtype=torch.float32, device="cpu", patch_based_mapping=False):
    """
    Función unificada para cargar datos dinámicos y estáticos locales a cada rank.
    
    Args:
        paths (dict): Diccionario con rutas ('coarse_paths', 'fine_paths', 'topo_folders').
        indices (list): Lista de índices a cargar.
        dtype (torch.dtype): Tipo de dato.
        device (str): Dispositivo ('cpu' o 'cuda').
        patch_based_mapping (bool): 
            - True: Usa la lógica compleja de extraer IDs del nombre del archivo coarse
              para seleccionar el mapa topográfico correspondiente de una lista ordenada. (Para Patch-based UNet).
            - False: Usa la lógica simple de cargar todo y repetir los mapas estáticos para que coincidan 
              con N_dyn (Para Full-scale o Partitioned donde el dominio es fijo).
    """

    def natural_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', os.path.basename(s))]

    # Función interna para extraer ID del nombre de archivo (Solo usada si patch_based_mapping=True)
    def get_chunk_id_from_name(filepath):
        basename = os.path.basename(filepath)
        # Buscar _DIGITO.txt al final
        match = re.search(r'_(\d+)\.[a-zA-Z]+$', basename)
        if match: return int(match.group(1))
        # Buscar red_DIGITO
        match_red = re.search(r'red_(\d+)', filepath)
        if match_red: return int(match_red.group(1))
        # Fallback: último número
        digits = re.findall(r'\d+', basename)
        if digits: return int(digits[-1])
        raise ValueError(f"No se pudo extraer ID del archivo: {basename}")

    # CARGA DE DATOS DINÁMICOS (Común para ambos métodos)
    local_coarse_paths = [paths['coarse_paths'][i] for i in indices]
    local_fine_paths   = [paths['fine_paths'][i]   for i in indices]

    coarse_np = [np.loadtxt(p) for p in local_coarse_paths]
    fine_np   = [np.loadtxt(p) for p in local_fine_paths]

    # Añadir canal y stackear
    coarse_np = np.stack(coarse_np, axis=0)[..., np.newaxis] 
    fine_np   = np.stack(fine_np,   axis=0)[..., np.newaxis] 

    coarse = torch.tensor(coarse_np, dtype=dtype, device=device)
    fine   = torch.tensor(fine_np,   dtype=dtype, device=device)

    N_local, H, W, _ = fine.shape
    
    # CARGA DE DATOS ESTÁTICOS (Lógica Bifurcada)
    
    # Topo (DEM, Slope, Catchment)
    # Asumimos 3 canales por defecto, se ajustará si se cargan datos
    topo_tensor = torch.zeros((N_local, H, W, 3), dtype=dtype, device=device)

    if 'topo_folders' in paths and paths['topo_folders'] is not None:
        n_features = len(paths['topo_folders'])
        
        # Cargar todas las rutas de topo
        paths_by_feature = []
        for folder in paths['topo_folders']:
            files = glob.glob(os.path.join(folder, "*.txt"))
            if not files: raise ValueError(f"No se encontraron archivos en {folder}")
            paths_by_feature.append(sorted(files, key=natural_key))
            
        if patch_based_mapping:
            # === MODO PATCH-BASED (Mapeo por ID) ===
            n_topo_chunks = len(paths_by_feature[0])
            
            # Cargar maestros (42, H, W, 3)
            all_topo_chunks_list = []
            for i_chunk in range(n_topo_chunks):
                chunk_features = []
                for i_feat in range(n_features):
                    f_path = paths_by_feature[i_feat][i_chunk]
                    chunk_features.append(np.loadtxt(f_path))
                all_topo_chunks_list.append(np.stack(chunk_features, axis=-1))
            
            topo_np_all_chunks = np.stack(all_topo_chunks_list, axis=0)

            # Asignar
            final_topo_list = []
            for c_path in local_coarse_paths:
                chunk_id = get_chunk_id_from_name(c_path)
                chunk_idx = chunk_id - 1
                if chunk_idx < 0 or chunk_idx >= n_topo_chunks:
                     chunk_idx = max(0, min(chunk_idx, n_topo_chunks - 1))
                final_topo_list.append(topo_np_all_chunks[chunk_idx])
                
            final_topo_np = np.stack(final_topo_list, axis=0)
            topo_tensor = torch.tensor(final_topo_np, dtype=dtype, device=device)
            
        else:
            # MODO FULL-RANGE/PARTITIONED (Repetición simple)
            # Aquí asumimos que cada carpeta tiene 1 solo archivo (el mapa completo) 
            
            topo_list = []
            # Aplanamos la lista: cargamos todo lo que hay en las carpetas
            for feat_paths in paths_by_feature:
                for tp in feat_paths:
                    topo_list.append(np.loadtxt(tp))
            
            # Apilamos todo en la dimensión de canales (H, W, n_total_archivos)
            topo_np = np.stack(topo_list, axis=-1) 
            
            # Repetimos para coincidir con N_local
            topo_tensor = torch.tensor(np.stack([topo_np]*N_local, axis=0), dtype=dtype, device=device)

    return coarse, topo_tensor, fine


def load_and_split_all_data(coarse_filepath, fine_filepath, topo_filepaths_list, z_fine_filepath, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_state=42):
    """
    Wrapper function that loads all data and splits it.
    """
    print("Loading coarse, fine, and topographic data...")
    
    coarse_data = load_coarse_simulation_data(coarse_filepath)
    fine_data = load_fine_simulation_data(fine_filepath)
    topo_data = load_fine_topographic_features(topo_filepaths_list)
    z_fine_data = load_z_fine_data(z_fine_filepath)
    
    splits = split_dataset(coarse_data, fine_data, topo_data, z_fine_data, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, random_state=random_state)

    return splits

def create_train_validation_test_datasets_per_rank(splits, world_size, global_rank, deterministic=True, seed=42):
    """
    Creates deterministic rank-wise splits for train/val/test. This does not mean that if we train twice the same model, 
    we will get the same results, since shuffling inside each epoch is still random, and also the algorithms themselves may be non-deterministic.
    But at least the data each rank sees will be the same between runs.

    Args:
        splits (dict): Dictionary with data divided by split_dataset()
        world_size (int): Total number of ranks (processes)
        global_rank (int): Current process rank
        deterministic (bool): If True, uses a fixed seed for shuffling
        seed (int): Seed for random number generator if deterministic is True

    Returns:
        tuple: (X_train, y_train, z_fine_train,
                X_val, y_val, z_fine_val,
                X_test, y_test, z_fine_test)
               where X_* are tuples of (coarse_data, topo_data),
               y_* are fine_data and z_fine_* are fine z (DEM values)
    """
    def split_for_rank(data):
        n = data.shape[0]
        if deterministic:
            # permutación determinista idéntica entre ejecuciones
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n)
            data = data[perm]
        per_rank = n // world_size
        start = global_rank * per_rank
        end = (global_rank + 1) * per_rank if global_rank < world_size - 1 else n
        return data[start:end]

    # sanity checks: all splits must have same length across modalities
    for split_name in ['train', 'val', 'test']:
        n_coarse = splits['coarse'][split_name].shape[0]
        n_fine = splits['fine'][split_name].shape[0]
        n_topo = splits['topo'][split_name].shape[0]
        n_z_fine = splits['z_fine'][split_name].shape[0]

        assert n_coarse == n_fine == n_topo == n_z_fine, \
            f"Inconsistent sample counts in {split_name}: coarse={n_coarse}, fine={n_fine}, topo={n_topo}, z_fine={n_z_fine}"

    # TRAIN
    X_train = (
        split_for_rank(splits['coarse']['train']),
        split_for_rank(splits['topo']['train'])
    )
    y_train = split_for_rank(splits['fine']['train'])
    z_fine_train = split_for_rank(splits['z_fine']['train'])

    # VAL
    X_val = (
        split_for_rank(splits['coarse']['val']),
        split_for_rank(splits['topo']['val'])
    )
    y_val = split_for_rank(splits['fine']['val'])
    z_fine_val = split_for_rank(splits['z_fine']['val'])

    # TEST
    X_test = (
        split_for_rank(splits['coarse']['test']),
        split_for_rank(splits['topo']['test'])
    )
    y_test = split_for_rank(splits['fine']['test'])
    z_fine_test = split_for_rank(splits['z_fine']['test'])

    return (
        X_train, y_train, z_fine_train,
        X_val, y_val, z_fine_val,
        X_test, y_test, z_fine_test
    )

def print_memory_usage(tag, global_rank, device):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 3)  # GB
    print(f"[Rank {global_rank}] {tag}: CPU RAM = {mem:.2f} GB", flush=True)
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        print(f"[Rank {global_rank}] {tag}: GPU mem allocated = {allocated:.2f} GB, reserved = {reserved:.2f} GB", flush=True)

def diagnose_all_data(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Diagnoses common data problems for PyTorch Tensors in distributed training.
    Each rank computes its stats, and only rank 0 prints them all.
    X_* are tuples: (X_coarse, X_topo), where X_topo has shape [N, H, W, C].
    y_* are Tensors of shape [N, H, W, 1].
    """

    def compute_stats(Xc, Xt, y):
        stats = {}

        # NaN / Inf
        stats['nan'] = torch.isnan(Xc).any() or torch.isnan(Xt).any() or torch.isnan(y).any()
        stats['inf'] = torch.isinf(Xc).any() or torch.isinf(Xt).any() or torch.isinf(y).any()

        # Ranges
        stats['Xc_range'] = (Xc.min().item(), Xc.max().item())
        stats['Xt_range'] = [(Xt[:, :, :, c].min().item(), Xt[:, :, :, c].max().item()) for c in range(Xt.shape[3])]
        stats['y_range']  = (y.min().item(), y.max().item())

        # Variances
        stats['Xc_var'] = torch.var(Xc).item()
        stats['Xt_var'] = [torch.var(Xt[:, :, :, c]).item() for c in range(Xt.shape[3])]
        stats['y_var'] = torch.var(y).item()

        # Means
        stats['Xc_mean'] = torch.mean(Xc).item()
        stats['Xt_mean'] = [torch.mean(Xt[:, :, :, c]).item() for c in range(Xt.shape[3])]
        stats['y_mean'] = torch.mean(y).item()

        # Sizes
        stats['N'] = Xc.shape[0]

        return stats

    # Compute stats locally per rank
    local_stats = {
        'train': compute_stats(X_train[0], X_train[1], y_train),
        'val':   compute_stats(X_val[0], X_val[1], y_val),
        'test':  compute_stats(X_test[0], X_test[1], y_test)
    }

    # Gather all stats to rank 0
    if dist.is_initialized():
        all_stats = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(all_stats, local_stats)
    else:
        all_stats = [local_stats]

    # Only rank 0 prints
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("\n🔬 DATA DIAGNOSIS (GLOBAL VIEW)")
        print("=" * 60)
        for r, stats in enumerate(all_stats):
            print(f"\n--- Rank {r} ---")
            if stats is None:
                # In some environments all_gather_object may leave entries as None if not received;
                # skip these to avoid "None is not subscriptable" errors.
                print(f"No stats available for rank {r} (None). Skipping.")
                continue
            for split in ['train', 'val', 'test']:
                s = stats.get(split)
                if s is None:
                    print(f"No '{split}' stats for rank {r}.")
                    continue
                print(f"\n📊 {split.upper()} STATISTICS:")
                print(f"Samples: {s['N']}")
                print(f"NaN present: {s['nan']}, Inf present: {s['inf']}")
                print(f"X_coarse range: {s['Xc_range']}, mean: {s['Xc_mean']:.6f}, var: {s['Xc_var']:.6f}")
                for c, (rmin, rmax) in enumerate(s['Xt_range']):
                    print(f"X_topo channel {c} range: [{rmin:.6f}, {rmax:.6f}], mean: {s['Xt_mean'][c]:.6f}, var: {s['Xt_var'][c]:.6f}")
                print(f"y range: {s['y_range']}, mean: {s['y_mean']:.6f}, var: {s['y_var']:.6f}")
        print("=" * 60)

def coerce_normalization_parameters(parameters):
    if parameters is None:
        return None
    if len(parameters) in (6, 8):
        return tuple(parameters[:6])
    raise ValueError(
        f"Expected 6 normalization parameters (or 8 from a legacy file), got {len(parameters)}"
    )


def normalize_all_datasets_minmax(X_train_coarse, X_train_topo, y_train_fine,
                                   X_val_coarse, X_val_topo, y_val_fine,
                                   X_test_coarse, X_test_topo, y_test_fine,
                                   device,
                                   log1p_exp1m_transform=False,
                                   normalization_parameters=None):
    """
    Performs Min-Max normalization on all datasets.
    
    Note: Normalization parameters (min/max) are computed ONLY from the training set
          to avoid data leakage.
    """
    normalization_parameters = coerce_normalization_parameters(normalization_parameters)

    if log1p_exp1m_transform: # Apply log1p to handle skewed distributions, which are h_fine and h_coarse
        X_train_coarse = torch.log1p(X_train_coarse)
        X_val_coarse   = torch.log1p(X_val_coarse)
        X_test_coarse  = torch.log1p(X_test_coarse)
        y_train_fine = torch.log1p(y_train_fine)
        y_val_fine   = torch.log1p(y_val_fine)  
        y_test_fine  = torch.log1p(y_test_fine)

    # Compute min/max from training data (from all ranks if DDP. We use all_reduce to get global min/max)
    if normalization_parameters is None:
        Xc_min, Xc_max = compute_global_minmax_distributed(X_train_coarse, device=device)
        Xt_min, Xt_max = compute_global_minmax_distributed(
            X_train_topo,
            dim=(0, 1, 2),
            device=device,
            nodata_value=-9999
        )
        y_min,  y_max  = compute_global_minmax_distributed(y_train_fine, device=device)
    else:
        Xc_min, Xc_max, Xt_min, Xt_max, y_min, y_max = normalization_parameters

    # Place each pair on the device of the tensor it will normalize.
    Xc_min, Xc_max = Xc_min.to(X_train_coarse.device), Xc_max.to(X_train_coarse.device)
    Xt_min, Xt_max = Xt_min.to(X_train_topo.device), Xt_max.to(X_train_topo.device)
    y_min, y_max = y_min.to(y_train_fine.device), y_max.to(y_train_fine.device)

    # Coarse
    X_train_coarse = (X_train_coarse - Xc_min) / (Xc_max - Xc_min + 1e-8)
    X_val_coarse   = (X_val_coarse - Xc_min.to(X_val_coarse.device)) / (Xc_max.to(X_val_coarse.device) - Xc_min.to(X_val_coarse.device) + 1e-8)
    X_test_coarse  = (X_test_coarse - Xc_min.to(X_test_coarse.device)) / (Xc_max.to(X_test_coarse.device) - Xc_min.to(X_test_coarse.device) + 1e-8)

    # Topographic features (channel-wise)
    X_train_topo_mask = X_train_topo == -9999
    X_train_topo = (X_train_topo - Xt_min) / (Xt_max - Xt_min + 1e-8)
    X_train_topo = torch.where(X_train_topo_mask, torch.zeros_like(X_train_topo), X_train_topo)

    X_val_topo_mask = X_val_topo == -9999
    X_val_topo = (X_val_topo - Xt_min.to(X_val_topo.device)) / (Xt_max.to(X_val_topo.device) - Xt_min.to(X_val_topo.device) + 1e-8)
    X_val_topo = torch.where(X_val_topo_mask, torch.zeros_like(X_val_topo), X_val_topo)

    X_test_topo_mask = X_test_topo == -9999
    X_test_topo = (X_test_topo - Xt_min.to(X_test_topo.device)) / (Xt_max.to(X_test_topo.device) - Xt_min.to(X_test_topo.device) + 1e-8)
    X_test_topo = torch.where(X_test_topo_mask, torch.zeros_like(X_test_topo), X_test_topo)

    # Fine (output)
    y_train_fine = (y_train_fine - y_min) / (y_max - y_min + 1e-8)
    y_val_fine   = (y_val_fine - y_min.to(y_val_fine.device)) / (y_max.to(y_val_fine.device) - y_min.to(y_val_fine.device) + 1e-8)
    y_test_fine  = (y_test_fine - y_min.to(y_test_fine.device)) / (y_max.to(y_test_fine.device) - y_min.to(y_test_fine.device) + 1e-8)

    if dist.get_rank() == 0:  # Solo rank 0 imprime
        print("Normalization parameters:")
        print(f"Xc_min: {Xc_min.item():.6f}, Xc_max: {Xc_max.item():.6f}")
        print("Xt_min por canal:", Xt_min.flatten().tolist())
        print("Xt_max por canal:", Xt_max.flatten().tolist())
        print(f"y_min:  {y_min.item():.6f}, y_max:  {y_max.item():.6f}")

    return (
        X_train_coarse, X_train_topo, y_train_fine,
        X_val_coarse, X_val_topo, y_val_fine,
        X_test_coarse, X_test_topo, y_test_fine,
        # Return min/max values for inverse transformation later if needed for denormalization
        (Xc_min, Xc_max, Xt_min, Xt_max, y_min, y_max)
    )

def compute_global_minmax_distributed(tensor, dim=None, device=None, nodata_value=None):
    """
    Calcula min y max globales de un tensor entre todos los ranks (DDP).
    
    Args:
        tensor (torch.Tensor): Tensor local del rank actual.
        dim (tuple or None): Dimensiones sobre las que calcular min/max local antes de sincronizar.
                             Ej: (0,1,2) para normalizar canal por canal (siendo canal la última dim).
                             Si es None, calcula el min/max global del tensor completo.
        device (torch.device or None): Dispositivo donde mover los tensores locales antes de all_reduce.

    Returns:
        global_min, global_max (torch.Tensor): Tensores con los min/max globales.
    """
    # Min y max locales
    if nodata_value is None:
        local_min = tensor.amin(dim=dim, keepdim=True) if dim is not None else torch.min(tensor)
        local_max = tensor.amax(dim=dim, keepdim=True) if dim is not None else torch.max(tensor)
    else:
        valid_mask = tensor != nodata_value
        tensor_for_min = torch.where(
            valid_mask,
            tensor,
            torch.full_like(tensor, float("inf"))
        )
        tensor_for_max = torch.where(
            valid_mask,
            tensor,
            torch.full_like(tensor, float("-inf"))
        )
        local_min = tensor_for_min.amin(dim=dim, keepdim=True) if dim is not None else torch.min(tensor_for_min)
        local_max = tensor_for_max.amax(dim=dim, keepdim=True) if dim is not None else torch.max(tensor_for_max)

    # Mover a la GPU del rank actual
    local_min = local_min.to(device)
    local_max = local_max.to(device)

    if dist.is_available() and dist.is_initialized():
        global_min = local_min.clone()
        global_max = local_max.clone()
        dist.all_reduce(global_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
        if nodata_value is not None and (torch.isinf(global_min).any() or torch.isinf(global_max).any()):
            raise ValueError("No hay valores válidos para al menos un canal tras la reducción distribuida.")
        # Devolver al host (CPU) para guardarlos o imprimir
        return global_min.cpu(), global_max.cpu()
    else:
        if nodata_value is not None and (torch.isinf(local_min).any() or torch.isinf(local_max).any()):
            raise ValueError("No hay valores válidos para al menos un canal.")
        return local_min.cpu(), local_max.cpu()

def prepare_real_case_inputs(
    H_COARSE_REAL_CASE_DATA_PATH,
    TOPO_DATA_PATHS,
    norm_params,
    device,
    H_FINE_REAL_CASE_DATA_PATH=None,
    LOG1P_EXP1M_TRANSFORM=False
):
    """
    Carga, normaliza, transforma y prepara los datos de un caso real para evaluación.
    Devuelve (X_real_case, y_real_case)
    donde X_real_case = (Xc_real_case, Xt_real_case).
    """
    norm_params = coerce_normalization_parameters(norm_params)

    # --- Load h_coarse ---
    Xc_min, Xc_max = norm_params[0], norm_params[1]
    Xc_real_case = load_coarse_simulation_data(H_COARSE_REAL_CASE_DATA_PATH)
    Xc_min, Xc_max = Xc_min.to(Xc_real_case.device), Xc_max.to(Xc_real_case.device)
    if LOG1P_EXP1M_TRANSFORM:
        Xc_real_case = torch.log1p(Xc_real_case)
    Xc_real_case = (Xc_real_case - Xc_min) / (Xc_max - Xc_min + 1e-8)
    Xc_real_case = Xc_real_case.to(device)

    # --- Load h_fine (optional) ---
    y_real_case = None
    if H_FINE_REAL_CASE_DATA_PATH is not None:
        y_min, y_max = norm_params[4], norm_params[5]
        y_real_case = load_fine_simulation_data(H_FINE_REAL_CASE_DATA_PATH)
        y_min, y_max = y_min.to(y_real_case.device), y_max.to(y_real_case.device)
        if LOG1P_EXP1M_TRANSFORM:
            y_real_case = torch.log1p(y_real_case)
        y_real_case = (y_real_case - y_min) / (y_max - y_min + 1e-8)
        y_real_case = y_real_case.to(device)

    # --- Load topo ---
    Xt_min, Xt_max = norm_params[2], norm_params[3]
    Xt_real_case = load_fine_topographic_features(TOPO_DATA_PATHS)
    Xt_min, Xt_max = Xt_min.to(Xt_real_case.device), Xt_max.to(Xt_real_case.device)
    Xt_real_case_mask = Xt_real_case == -9999
    Xt_real_case = (Xt_real_case - Xt_min) / (Xt_max - Xt_min + 1e-8)
    Xt_real_case = torch.where(
        Xt_real_case_mask,
        torch.zeros_like(Xt_real_case),
        Xt_real_case
    )
    Xt_real_case = Xt_real_case.to(device)

    print(
        "Real case normalized ranges: "
        f"DEM=[{Xt_real_case[..., 0].min().item():.6f}, {Xt_real_case[..., 0].max().item():.6f}], "
        f"MANN=[{Xt_real_case[..., 1].min().item():.6f}, {Xt_real_case[..., 1].max().item():.6f}]"
    )

    # --- Expand topo to match coarse length ---
    num_groups = Xc_real_case.shape[0] // Xt_real_case.shape[0]
    Xt_real_case = Xt_real_case.repeat(num_groups, 1, 1, 1)

    # --- Package inputs X---
    X_real_case = (Xc_real_case, Xt_real_case)

    return X_real_case, y_real_case

def plot_pixel_value_distribution(y_tensor, save_path, bins=100, log_scale=True, title="Distribución de profundidades", x_range=None):
    """
    Dibuja un histograma con la distribución de valores de los píxeles
    en un tensor (N, H, W, 1) o (N, H, W), y lo guarda en archivo.

    Args:
        y_tensor (torch.Tensor or np.ndarray): Tensor con los valores de profundidad.
        save_path (str): Ruta para guardar la figura.
        bins (int): Número de bins para el histograma.
        log_scale (bool): Si True, usa escala logarítmica en el eje Y.
        title (str): Título del gráfico.
        x_range (tuple): Rango (min, max) de valores a mostrar en el eje X.
    """
    if isinstance(y_tensor, torch.Tensor):
        y_np = y_tensor.detach().cpu().numpy()
    else:
        y_np = y_tensor

    # Aplanar todos los valores
    pixel_values = y_np.reshape(-1)

    # Filtrar si se especifica un rango en X
    if x_range is not None:
        pixel_values = pixel_values[(pixel_values >= x_range[0]) & (pixel_values <= x_range[1])]

    plt.figure(figsize=(8, 5))
    plt.hist(pixel_values, bins=bins, color='steelblue', edgecolor='black')
    
    if log_scale:
        plt.yscale('log')
    
    if x_range:
        plt.xlim(x_range)

    plt.xlabel("Valor de profundidad")
    plt.ylabel("Frecuencia")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Histograma guardado en: {save_path}")
