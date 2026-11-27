import torch
import numpy as np
import re
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr
from data_preparation import coerce_normalization_parameters
from unet_model import ResizeWithBicubic, ResizeWithClassicalSubGrid
import pickle
from itertools import zip_longest
from collections import Counter
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset



def save_tensor_as_ascii_grids(tensor, xllcorner, yllcorner, cell_size, filename, extension_archivo, NODATA_value=-9999):
    """
    Saves each batch of a tensor of size (batch, H, W, 1) as an ASCII grid file.
    """
    # Ensure tensor is on CPU and convert to NumPy
    tensor_np = tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
    batch_size, nrows, ncols, channels = tensor_np.shape
    assert channels == 1, "Only one channel is supported for saving in ASCII grid format."
    
    for i in range(batch_size):
        data = tensor_np[i, :, :, 0]
        full_name = f"{filename}_{i}.{extension_archivo}"
        with open(full_name, "w") as f:
            f.write(f"ncols         {ncols}\n")
            f.write(f"nrows         {nrows}\n")
            f.write(f"xllcorner     {xllcorner}\n")
            f.write(f"yllcorner     {yllcorner}\n")
            f.write(f"cellsize      {cell_size}\n")
            f.write(f"NODATA_value  {NODATA_value}\n")
            for row in data:
                f.write(" ".join(str(x) if not np.isnan(x) else str(NODATA_value) for x in row) + "\n")

def save_file_as_ascii_grids(initial_filepath, xllcorner, yllcorner, cell_size, filename, extension_archivo, NODATA_value=-9999):
    """
    Reads a text file with data (batch, H, W, 1) and saves each batch as an ASCII grid file.
    The text file must contain data compatible with np.loadtxt.
    """
    data = np.loadtxt(initial_filepath, dtype=np.float32) # Ensure float32
    
    if data.ndim == 2:
        # Assume a single batch and a single channel
        data = data[np.newaxis, :, :, np.newaxis]
    elif data.ndim == 3:
        # Assume (batch, H, W) -> add channel dimension
        data = data[:, :, :, np.newaxis]
    elif data.ndim == 4:
        pass # Already in correct shape
    else:
        raise ValueError("Text file does not have a compatible format (2D, 3D, or 4D).")

    batch_size, nrows, ncols, channels = data.shape
    assert channels == 1, "Only one channel is supported for saving in ASCII grid format."

    for i in range(batch_size):
        grid = data[i, :, :, 0]
        full_name = f"{filename}_{i}.{extension_archivo}"
        with open(full_name, "w") as f:
            f.write(f"ncols         {ncols}\n")
            f.write(f"nrows         {nrows}\n")
            f.write(f"xllcorner     {xllcorner}\n")
            f.write(f"yllcorner     {yllcorner}\n")
            f.write(f"cellsize      {cell_size}\n")
            f.write(f"NODATA_value  {NODATA_value}\n")
            for row in grid:
                f.write(" ".join(str(x) if not np.isnan(x) else str(NODATA_value) for x in row) + "\n")

def load_and_stitch_and_plot_tensors_from_files(folders_to_search_list, file_pattern_list, 
                                                n_samples, n_subs, n_cols, 
                                                title_list, cmap_list,
                                                save_path='./stitched_predictions_plots'):
    
    # --- Helper: Orden Natural ---
    def natural_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', os.path.basename(s))]

    # --- Helper: Stitching Robusto ---
    def stitch_tensor(tensor_flat, n_samples, n_subs, n_cols):
        if n_cols is None: raise ValueError("Falta n_cols")
        n_rows = n_subs // n_cols
        
        stitched_list = []
        # Ahora tensor_flat está ordenado correctamente: 
        # [Sample1_Chunk1...Sample1_Chunk42, Sample2_Chunk1...]
        for i in range(n_samples):
            # Extraemos todos los chunks de UNA sola sample
            patches = tensor_flat[i*n_subs : (i+1)*n_subs]
            
            rows_images = []
            for r in range(n_rows):
                row_patches = patches[r*n_cols : (r+1)*n_cols]
                # Concatenar horizontalmente
                if len(row_patches) > 0:
                    row_img = torch.cat(list(row_patches), dim=1) 
                    rows_images.append(row_img)
            
            if rows_images:
                full_img = torch.cat(rows_images, dim=0)
                stitched_list.append(full_img)
            
        if not stitched_list:
            raise ValueError("No se pudieron crear imágenes cosidas (lista vacía).")

        return torch.stack(stitched_list, dim=0)

    # --- Procesamiento ---
    stitched_tensors_per_variable = []
    
    # Verificamos que el número de carpetas coincida con los sub-chunks (subs)
    # Asumimos que folders_to_search_list está ordenada: red_1, red_2... red_42
    if len(folders_to_search_list) != n_subs:
        print(f"⚠️ Advertencia: Tienes {len(folders_to_search_list)} carpetas pero n_subs={n_subs}. Asegúrate de que coincidan espacialmente.")

    for pattern in file_pattern_list:
        print(f"Buscando archivos para patrón: {pattern}...")
        
        # Estructura temporal: files_matrix[sample_index][chunk_index]
        files_matrix = [[None for _ in range(n_subs)] for _ in range(n_samples)]
        
        # Iteramos sobre las carpetas (que representan los CHUNKS espaciales)
        for chunk_idx, folder in enumerate(folders_to_search_list):
            if chunk_idx >= n_subs: break # Seguridad

            found_in_folder = glob.glob(os.path.join(folder, pattern))
            # Ordenamos por nombre para asegurar que Sample 1 va antes que Sample 2
            found_in_folder.sort(key=natural_key)
            
            # Cogemos solo los necesarios (por si hay basura en la carpeta)
            found_in_folder = found_in_folder[:n_samples]
            
            if len(found_in_folder) < n_samples:
                print(f"❌ Error Crítico: En la carpeta '{folder}' solo encontré {len(found_in_folder)} archivos para '{pattern}', necesito {n_samples}.")
                return

            # Asignamos cada archivo a su Sample correspondiente
            for sample_idx, f_path in enumerate(found_in_folder):
                files_matrix[sample_idx][chunk_idx] = f_path

        # APLANAMOS LA LISTA EN EL ORDEN CORRECTO
        # Orden deseado: [Sample1_Chunk1, Sample1_Chunk2... Sample1_Chunk42, Sample2_Chunk1...]
        files_to_load = []
        for s in range(n_samples):
            for c in range(n_subs):
                file_path = files_matrix[s][c]
                if file_path is None:
                    print(f"❌ Error: Falta el archivo para Sample {s}, Chunk {c}")
                    return
                files_to_load.append(file_path)

        print(f"  -> Cargando {len(files_to_load)} archivos ordenados...")

        data_list = []
        for f in files_to_load:
            try:
                if f.endswith('.txt'):
                    arr = np.loadtxt(f, dtype=np.float32)
                else:
                    arr = torch.load(f, map_location='cpu')
                    if isinstance(arr, torch.Tensor): arr = arr.numpy()
                
                if arr.ndim == 2: arr = arr[..., np.newaxis]
                data_list.append(arr)
            except Exception as e:
                print(f"Error leyendo {f}: {e}")
                return

        # Crear tensor y Unir
        tensor_flat = torch.tensor(np.stack(data_list, axis=0)) # (Total, H, W, C)
        
        try:
            stitched_img = stitch_tensor(tensor_flat, n_samples, n_subs, n_cols)
            stitched_tensors_per_variable.append(stitched_img)
        except Exception as e:
            print(f"Error haciendo stitch para {pattern}: {e}")
            return

    # --- Plotting ---
    if not stitched_tensors_per_variable:
        print("No hay tensores para plotear.")
        return

    n_vars = len(stitched_tensors_per_variable)
    fig, axes = plt.subplots(n_samples, n_vars, figsize=(5 * n_vars, 5 * n_samples))
    
    if n_samples == 1 and n_vars == 1: axes = np.array([[axes]])
    elif n_samples == 1: axes = axes.reshape(1, -1)
    elif n_vars == 1: axes = axes.reshape(-1, 1)

    for i in range(n_samples):
        for j in range(n_vars):
            ax = axes[i, j]
            # Accedemos: Variable J, Sample I
            img = stitched_tensors_per_variable[j][i, :, :, 0].cpu().numpy()
            
            cmap = cmap_list[j] if j < len(cmap_list) else 'viridis'
            title = title_list[j] if j < len(title_list) else f"Var {j}"
            
            vmin, vmax = None, None
            if 'Binary' in title or 'Bin' in title: vmin, vmax = -1, 1
            
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{title}\nSample {i+1}")
            ax.axis('off')
            
            # Colorbar solo en la última columna para no saturar, o en todas si prefieres
            if j == n_vars - 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                 plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Guardado correctamente en: {save_path}")
    
def visualize_predictions(model, X_test, y_test,
                                  save_path, total_pixels_to_crop_on_each_dim, cell_size_ratio,
                                  log1p_exp1m_transform=False,
                                  normalization_parameters=None,
                                  n_samples=5, cmap='terrain', seed=42,
                                  device='cpu', clip_value_min=None, clip_value_max=None,
                                  error_threshold=0.01,
                                  grouping_parameter_for_subgrid_rows=1,
                                  grouping_parameter_for_subgrid_cols=1,
                                  specific_indices=None,
                                  PATCH_BASED_MAPPING=False,
                                  NUMBER_OF_SUBDOMAINS=None,
                                  NUMBER_OF_COLUMNS_IN_SUBDOMAINS=None,
                                  evaluation_mask=None,
                                  evaluation_mask_path=None
                                  ):
    """
    Visualiza y compara UNet vs Bicubic (con crop central) vs Ground Truth.
    Añade un mapa de error absoluto entre UNet y Ground Truth,
    mostrando sólo errores mayores que `error_threshold`.
    """
    cmap_reds = plt.cm.viridis.copy()
    cmap_reds.set_bad(color='white')  # para fijar el color con el que marcar los píxeles < error_threshold

    np.random.seed(seed)
    model.eval()
    model.to(device)

    X_coarse, X_topo = X_test
    N_total_entries = X_coarse.shape[0]

    # --------------------------------------------------------------
    # 1. SELECCIÓN DE ÍNDICES (Time Steps vs Imágenes)
    # --------------------------------------------------------------
    if PATCH_BASED_MAPPING and NUMBER_OF_SUBDOMAINS is not None:
        if N_total_entries % NUMBER_OF_SUBDOMAINS != 0:
            raise ValueError(f"Total datos ({N_total_entries}) no divisible por subdominios ({NUMBER_OF_SUBDOMAINS})")
        N_time_steps = N_total_entries // NUMBER_OF_SUBDOMAINS
        max_index = N_time_steps
        print(f"Modo Patch-Based: {N_total_entries} parches representan {N_time_steps} mapas completos.")
    else:
        max_index = N_total_entries

    if specific_indices is not None:
        indices = np.array(specific_indices)
        if np.any(indices >= max_index) or np.any(indices < 0):
            raise ValueError(f"Índices fuera de rango [0, {max_index-1}]")
        n_samples = len(indices)
        print(f"Visualizando {n_samples} muestras específicas: {indices}")
    else:
        n_samples = min(n_samples, max_index)
        indices = np.random.choice(max_index, n_samples, replace=False)
        print(f"Visualizando {n_samples} muestras aleatorias: {indices}")

    # --------------------------------------------------------------
    # 2. PREPARACIÓN DEL BATCH DE PARCHES (Expanded Indices)
    # --------------------------------------------------------------
    # Si es Patch-Based, expandimos los índices (ej: idx 0 -> parches 0 a 41)
    # Si es Normal, los índices son directos.
    
    expanded_indices = []
    if PATCH_BASED_MAPPING and NUMBER_OF_SUBDOMAINS is not None:
        for idx in indices:
            start = idx * NUMBER_OF_SUBDOMAINS
            end = start + NUMBER_OF_SUBDOMAINS
            expanded_indices.extend(range(start, end))
    else:
        expanded_indices = indices
    
    # Cargar datos (Parches o Imágenes completas según el caso)
    # Nota: Aquí cargamos TODOS los parches necesarios de golpe. 
    Xc_sub = X_coarse[expanded_indices].to(device)
    Xt_sub = X_topo[expanded_indices].to(device)
    y_sub = y_test[expanded_indices].to(device)

    # --------------------------------------------------------------
    # 3. PREDICCIÓN Y CROPEO (Sobre parches individuales)
    # --------------------------------------------------------------
    with torch.no_grad():
        pred = model(Xc_sub, Xt_sub)

    # Denormalize
    if normalization_parameters is not None:
        normalization_parameters = coerce_normalization_parameters(normalization_parameters)
        Xc_min, Xc_max, Xt_min, Xt_max, y_min, y_max = normalization_parameters
        Xc_min, Xc_max = Xc_min.to(Xc_sub.device), Xc_max.to(Xc_sub.device)
        Xt_min, Xt_max = Xt_min.to(Xt_sub.device), Xt_max.to(Xt_sub.device)
        y_min, y_max = y_min.to(pred.device), y_max.to(pred.device)

        Xc_sub = Xc_sub * (Xc_max - Xc_min + 1e-8) + Xc_min
        Xt_sub = Xt_sub * (Xt_max - Xt_min + 1e-8) + Xt_min
        pred = pred * (y_max - y_min + 1e-8) + y_min
        y_sub = y_sub * (y_max - y_min + 1e-8) + y_min

    if log1p_exp1m_transform: # inverse transform
        pred = torch.expm1(pred)
        y_sub = torch.expm1(y_sub)
        Xc_sub = torch.expm1(Xc_sub)

    # Crop the target to match output dimensions when the size difference is odd.
    H_pred, W_pred = pred.shape[1:3]
    H_true, W_true = y_sub.shape[1:3]
    crop_h = (H_true - H_pred) // 2
    crop_w = (W_true - W_pred) // 2
    y_sub_cropped = y_sub[:, crop_h:crop_h + H_pred, crop_w:crop_w + W_pred, :]

    # Crop central de la entrada Xc_sub y Xt_sub para que coincida con dimensiones de pred
    crop_coarse = (total_pixels_to_crop_on_each_dim // cell_size_ratio) // 2
    if crop_coarse == 0:
        Xc_sub_cropped = Xc_sub
    else:
        Xc_sub_cropped = Xc_sub[
            :, crop_coarse:-crop_coarse, crop_coarse:-crop_coarse, :
        ]
    crop_topo = total_pixels_to_crop_on_each_dim // 2
    if crop_topo == 0:
        Xt_sub_cropped = Xt_sub
    else:
        Xt_sub_cropped = Xt_sub[
            :, crop_topo:-crop_topo, crop_topo:-crop_topo, :
        ]

    # Bicubic resize
    target_h_resize, target_w_resize = y_sub_cropped.shape[1], y_sub_cropped.shape[2] 
    resize_layer_bicubic = ResizeWithBicubic(target_h_resize, target_w_resize).to(device)
    with torch.no_grad():
        bicubic = resize_layer_bicubic(Xc_sub_cropped[:, :, :, :1])  # Only the first channel, already cropped (coarse input)

    # Classical Subgrid resize
    target_h_resize, target_w_resize = y_sub.shape[1], y_sub.shape[2] #! Usamos dimensiones ANTES del cropeo para que se haga correctamente el algoritmo de subgrid
    resize_layer_subgrid = ResizeWithClassicalSubGrid(target_h_resize, target_w_resize).to(device)
    with torch.no_grad():
        classical_subgrid = resize_layer_subgrid(
            coarse_depth=Xc_sub[:, :, :, :1],   # Coarse depth map (sin crop)
            fine_dem=Xt_sub[:, :, :, :1],       # Fine DEM (sin crop)
            group_size_rows=grouping_parameter_for_subgrid_rows, 
            group_size_cols=grouping_parameter_for_subgrid_cols
        )
    classical_subgrid = classical_subgrid[:, crop_h:crop_h + H_pred, crop_w:crop_w + W_pred, :]

    # --------------------------------------------------------------
    # 4. UNIÓN (STITCHING) - Solo si es Patch-Based
    # --------------------------------------------------------------
    # Aplicar Stitching o Pasar Directo
    n_samples = len(indices) if specific_indices is not None else n_samples
    if PATCH_BASED_MAPPING and NUMBER_OF_SUBDOMAINS is not None: # En caso de que sea Patch-Based
        # Estamos uniendo parches YA RECORTADOS Y PROCESADOS
        pred    = stitch_tensor(pred, n_samples, NUMBER_OF_SUBDOMAINS, NUMBER_OF_COLUMNS_IN_SUBDOMAINS)
        y_sub_cropped       = stitch_tensor(y_sub_cropped, n_samples, NUMBER_OF_SUBDOMAINS, NUMBER_OF_COLUMNS_IN_SUBDOMAINS)
        
        bicubic = stitch_tensor(bicubic, n_samples, NUMBER_OF_SUBDOMAINS, NUMBER_OF_COLUMNS_IN_SUBDOMAINS)
        classical_subgrid = stitch_tensor(classical_subgrid, n_samples, NUMBER_OF_SUBDOMAINS, NUMBER_OF_COLUMNS_IN_SUBDOMAINS)
        
        Xc_sub_cropped      = stitch_tensor(Xc_sub_cropped, n_samples, NUMBER_OF_SUBDOMAINS, NUMBER_OF_COLUMNS_IN_SUBDOMAINS)
        Xt_sub_cropped      = stitch_tensor(Xt_sub_cropped, n_samples, NUMBER_OF_SUBDOMAINS, NUMBER_OF_COLUMNS_IN_SUBDOMAINS)


    # Clip values if specified
    if clip_value_min is not None:
        pred = torch.clamp(pred, min=clip_value_min)
        bicubic = torch.clamp(bicubic, min=clip_value_min)
        classical_subgrid = torch.clamp(classical_subgrid, min=clip_value_min)
    if clip_value_max is not None:
        pred = torch.clamp(pred, max=clip_value_max)
        bicubic = torch.clamp(bicubic, max=clip_value_max)
        classical_subgrid = torch.clamp(classical_subgrid, max=clip_value_max)

    # # --- Estudio de Comparación de Volúmenes por Muestra (Pasar de él) ---
    # sum_depth_coarse = torch.sum(Xc_sub, dim=(1, 2, 3))
    # sum_depth_subgrid = torch.sum(classical_subgrid, dim=(1, 2, 3))
    # depth_diff_per_sample = sum_depth_coarse * (cell_size_ratio **2) - sum_depth_subgrid
    # print("\n--- Comparación de Alturas (Coarse vs. Subgrid) por Muestra ---")
    # for i in range(n_samples):
    #     vol_coarse = sum_depth_coarse[i].item() * (cell_size_ratio **2)
    #     vol_subgrid = sum_depth_subgrid[i].item()
    #     diff = depth_diff_per_sample[i].item()
    #     relative_error = (diff / vol_coarse) * 100 if vol_coarse > 1e-9 else float('inf')
    #     print(f"Muestra {i+1}:")
    #     print(f"  ├─ Altura GT      : {vol_coarse:.4f}")
    #     print(f"  ├─ Altura Subgrid : {vol_subgrid:.4f}")
    #     print(f"  └─ Diferencia      : {diff:.4f} ({relative_error:.4f} %)")

    # avg_diff = depth_diff_per_sample.mean().item()
    # std_diff = depth_diff_per_sample.std().item()
    # avg_rel_error = (torch.abs(depth_diff_per_sample) / sum_depth_coarse).mean().item() * 100
    # print("\nResumen Estadístico:")
    # print(f"  ├─ Diferencia media de alturas: {avg_diff:.4f} (std: {std_diff:.4f})")
    # print(f"  └─ Error relativo absoluto medio: {avg_rel_error:.4f} %")

    pred_np = pred.cpu().numpy()
    bicubic_np = bicubic.cpu().numpy()
    classical_subgrid_np = classical_subgrid.cpu().numpy()
    coarse_cropped_np = Xc_sub_cropped.cpu().numpy()
    topo_cropped_np = Xt_sub_cropped.cpu().numpy()
    y_np = y_sub_cropped.cpu().numpy()

    if evaluation_mask is not None:
        if evaluation_mask.ndim != 2:
            raise ValueError(
                f"La máscara geográfica debe ser 2D; se recibió shape {tuple(evaluation_mask.shape)}"
            )
        if PATCH_BASED_MAPPING:
            raise ValueError(
                "PATCH_BASED_MAPPING=True no admite visualización con máscara geográfica completa."
            )
        H_pred, W_pred = y_np.shape[1:3]
        full_reference_shape = y_sub.shape[1:3]
        if tuple(evaluation_mask.shape) != full_reference_shape:
            raise ValueError(
                f"Tamaño incompatible de máscara en visualización: máscara "
                f"{evaluation_mask.shape}, referencia completa {full_reference_shape}"
            )
        plot_domain = np.asarray(
            evaluation_mask[
                crop_h:crop_h + H_pred,
                crop_w:crop_w + W_pred,
            ],
            dtype=bool,
        )
        if plot_domain.shape != (H_pred, W_pred):
            raise ValueError(
                f"Tamaño incompatible tras recortar la máscara: máscara "
                f"{plot_domain.shape}, arrays exportados {(H_pred, W_pred)}"
            )
    else:
        plot_domain = None

    def fine_visible(array2d):
        if plot_domain is None:
            return array2d
        masked = np.array(array2d, copy=True)
        masked[~plot_domain] = np.nan
        return masked

    if save_path.endswith("_RealCase"):
        for i, timestep in enumerate(indices):
            np.savez_compressed(
                f"{save_path}_comparacion_t{int(timestep):02d}.npz",
                coarse=coarse_cropped_np[i, :, :, 0].astype(np.float32),
                gt=y_np[i, :, :, 0].astype(np.float32),
                unet=pred_np[i, :, :, 0].astype(np.float32),
                bicubic=bicubic_np[i, :, :, 0].astype(np.float32),
                subgrid=classical_subgrid_np[i, :, :, 0].astype(np.float32),
                domain_mask=plot_domain if plot_domain is not None else np.ones(
                    y_np.shape[1:3], dtype=bool
                ),
            )

    # Error relativo medio sobre todos los píxeles (dry & wet)
    if plot_domain is not None:
        relative_domain = np.broadcast_to(
            plot_domain[None, :, :],
            y_np.shape[:3],
        )
        y_relative = y_np[..., 0][relative_domain]
        unet_relative = pred_np[..., 0][relative_domain]
        bicubic_relative = bicubic_np[..., 0][relative_domain]
        subgrid_relative = classical_subgrid_np[..., 0][relative_domain]
    else:
        y_relative = y_np[..., 0].ravel()
        unet_relative = pred_np[..., 0].ravel()
        bicubic_relative = bicubic_np[..., 0].ravel()
        subgrid_relative = classical_subgrid_np[..., 0].ravel()

    rel_error_unet = np.mean(np.abs((y_relative - unet_relative) / (y_relative + 1e-3))) * 100
    rel_error_bicubic = np.mean(np.abs((y_relative - bicubic_relative) / (y_relative + 1e-3))) * 100
    rel_error_classical_subgrid = np.mean(np.abs((y_relative - subgrid_relative) / (y_relative + 1e-3))) * 100
    print(f"ERROR RELATIVO PROMEDIO (MAE denormalized) DE TODOS LOS PÍXELES (DRY & WET) SOBRE LAS {n_samples} samples:")
    print(f"UNet:         : {rel_error_unet:.2f}%")
    print(f"Bicubic:     : {rel_error_bicubic:.2f}%")
    print(f"Classical Subgrid: {rel_error_classical_subgrid:.2f}%\n")

    # Error relativo medio sólo sobre píxeles húmedos (donde h_true > threshold)
    mask = (np.abs(y_relative) > error_threshold)
    rel_error_unet = np.mean(np.abs((y_relative - unet_relative)[mask] / (y_relative[mask] + 1e-3))) * 100
    rel_error_bicubic = np.mean(np.abs((y_relative - bicubic_relative)[mask] / (y_relative[mask] + 1e-3))) * 100
    rel_error_classical_subgrid = np.mean(np.abs((y_relative - subgrid_relative)[mask] / (y_relative[mask] + 1e-3))) * 100

    print(f"ERROR RELATIVO PROMEDIO (MAE denormalized) DE SOLAMENTE LOS PÍXELES WET (EN LA REALIDAD) SOBRE LAS {n_samples} samples:")
    print(f"UNet:         : {rel_error_unet:.2f}%")
    print(f"Bicubic:     : {rel_error_bicubic:.2f}%")
    print(f"Classical Subgrid: {rel_error_classical_subgrid:.2f}%")

    # For plotting
    err_abs_signed = y_np - pred_np
    err_abs_signed_bicubic = y_np - bicubic_np
    err_abs_signed_classical_subgrid = y_np - classical_subgrid_np
    err_rel_signed_np = ((y_np - pred_np) / (np.abs(y_np) + 1e-8)) * 100
    gt_binary = (np.abs(y_np) > error_threshold).astype(int)
    pred_binary = (np.abs(pred_np) > error_threshold).astype(int)
    diff_binary = pred_binary - gt_binary # 0 = Acierto (TN/TP); 1 = FP; -1 = FN

    #! para la imagen grande con las n_samples filas que muestra todas las columnas

    output_dir = os.path.dirname(save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")


    fig, axes = plt.subplots(n_samples, 8, figsize=(25, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        # --- 1. Definir TODAS las imágenes a escalar juntas ---
        
        # Ground Truth (para la col 2)
        gt_img = y_np[i, :, :, 0]
        gt_masked = np.where(np.abs(gt_img) > error_threshold, gt_img, np.nan)
        gt_masked = fine_visible(gt_masked)

        # Bicubic Resize (para la col 3)
        bicubic_img = bicubic_np[i, :, :, 0]
        bicubic_masked = np.where(np.abs(bicubic_img) > error_threshold, bicubic_img, np.nan)
        bicubic_masked = fine_visible(bicubic_masked)

        # Classical Subgrid (para la col 4)
        classical_subgrid_img = classical_subgrid_np[i, :, :, 0]
        classical_subgrid_masked = np.where(np.abs(classical_subgrid_img) > error_threshold, classical_subgrid_img, np.nan)
        classical_subgrid_masked = fine_visible(classical_subgrid_masked)

        # Predicción UNet (para la col 5)
        pred_np_im = pred_np[i, :, :, 0]
        pred_np_masked = np.where(np.abs(pred_np_im) > error_threshold, pred_np_im, np.nan)
        pred_np_masked = fine_visible(pred_np_masked)
        # --- 2. Calcular límites comunes para esas 4 imágenes ---
        images_to_scale = [gt_masked, bicubic_masked, classical_subgrid_masked, pred_np_masked]
        
        try:
            # Usamos nanmin/nanmax para ignorar los NaN
            vmin = np.nanmin([np.nanmin(img) for img in images_to_scale])
            vmax = np.nanmax([np.nanmax(img) for img in images_to_scale])
        except ValueError: # Ocurre si todo es NaN
            vmin = 0
            vmax = 1

        # --- 3. Pintar todas las columnas ---

        # Col 0: Coarse Input (escala propia) (!! USA LÍMITES COMUNES !!)
        coarse_img = coarse_cropped_np[i, :, :, 0]
        coarse_masked = np.where(np.abs(coarse_img) > error_threshold, coarse_img, np.nan)
        axes[i, 0].imshow(coarse_masked, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"Coarse Input Cropped {indices[i]}")
        
        # Col 1: Topographic (escala propia)
        axes[i, 1].imshow(topo_cropped_np[i, :, :, 0], cmap=cmap)
        axes[i, 1].set_title("Topographic Input Cropped")
        
        # Col 2: Ground Truth (!! USA LÍMITES COMUNES !!)
        axes[i, 2].imshow(gt_masked, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i, 2].set_title("Ground Truth")

        # Col 3: Bicubic Resize (!! USA LÍMITES COMUNES !!)
        axes[i, 3].imshow(bicubic_masked, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i, 3].set_title("Bicubic Resize")

        # Col 4: Classical Subgrid Resize (!! USA LÍMITES COMUNES !!)
        axes[i, 4].imshow(classical_subgrid_masked, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i, 4].set_title("Classical Subgrid Resize")

        # Col 5: UNet Prediction (!! USA LÍMITES COMUNES !!)
        axes[i, 5].imshow(pred_np_masked, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i, 5].set_title("UNet Prediction")

        # --- Columnas de Error (escalas propias) ---

        # err_bicubic = err_abs_signed_bicubic[i, :, :, 0]
        # err_bicubic_masked = np.where(np.abs(err_bicubic) > error_threshold, err_bicubic, np.nan)
        # im_bicubic = axes[i, 6].imshow(err_bicubic_masked, cmap=cmap_reds)
        # axes[i, 6].set_title(f"Bicubic err abs signed (>{error_threshold})")

        # err_classical_subgrid = err_abs_signed_classical_subgrid[i, :, :, 0]
        # err_classical_subgrid_masked = np.where(np.abs(err_classical_subgrid) > error_threshold, err_classical_subgrid, np.nan)
        # im_classical_subgrid = axes[i, 7].imshow(err_classical_subgrid_masked, cmap=cmap_reds)
        # axes[i, 7].set_title(f"Classical Subgrid err abs signed (>{error_threshold})")
        
        diff_binary_img = diff_binary[i, :, :, 0]
        diff_binary_img = np.where(plot_domain, diff_binary_img, np.nan) if plot_domain is not None else diff_binary_img
        cmap_binary = mcolors.ListedColormap(['blue', 'white', 'red']) # Orden de valores: -1 (Azul), 0 (Blanco), 1 (Rojo)       
        im_unet_binary = axes[i, 6].imshow(diff_binary_img, cmap=cmap_binary, vmin=-1, vmax=1)
        axes[i, 6].set_title(f"Bin Err: Blue=FN, Red=FP")

        err_unet = err_abs_signed[i, :, :, 0]
        err_unet_masked = np.where(np.abs(err_unet) > error_threshold, err_unet, np.nan)
        err_unet_masked = fine_visible(err_unet_masked)
        im_unet = axes[i, 7].imshow(err_unet_masked, cmap=cmap_reds)
        axes[i, 7].set_title(f"UNet Abs Err (>{error_threshold})")

        # err_rel_signed = err_rel_signed_np[i, :, :, 0]
        # err_rel_signed_masked = np.where(np.abs(gt_img) > error_threshold, err_rel_signed, np.nan)
        # im_rel = axes[i, 9].imshow(err_rel_signed_masked, cmap='seismic')
        # axes[i, 9].set_title(f"GT err rel signed (>{error_threshold})")

        for j in range(8):
            axes[i, j].axis("off")

    # plt.tight_layout()
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # output_path = os.path.splitext(save_path)[0] + f"_predictions.png"
    # plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    # print(f"Saved all predictions comparison to: {output_path}")
    # plt.close()

    # Mapas individuales
    for i in range(n_samples):
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(topo_cropped_np[i, :, :, 0], cmap='YlOrRd')
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("DEM", rotation=270, labelpad=15)
        output_path = os.path.splitext(save_path)[0] + f"_DEM_{i+1}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved individual DEM map to: {output_path}")
        txt_output_path = os.path.splitext(save_path)[0] + f"_DEM_{i+1}.txt"
        np.savetxt(txt_output_path, topo_cropped_np[i, :, :, 0], fmt="%.6f")
        print(f"Saved individual DEM array to: {txt_output_path}")

        # fig, ax = plt.subplots(figsize=(4, 4))
        # im = ax.imshow(topo_cropped_np[i, :, :, 1], cmap='autumn')
        # ax.axis('off')
        # cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # cbar.ax.set_ylabel("Slope", rotation=270, labelpad=15)
        # output_path = os.path.splitext(save_path)[0] + f"_slope_{i+1}.png"
        # plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        # plt.close()
        # print(f"Saved individual slope map to: {output_path}")

        # fig, ax = plt.subplots(figsize=(4, 4))
        # im = ax.imshow(topo_cropped_np[i, :, :, 2], cmap='YlOrRd')
        # ax.axis('off')
        # cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # cbar.ax.set_ylabel("Catchment area", rotation=270, labelpad=15)
        # output_path = os.path.splitext(save_path)[0] + f"_catch_area_{i+1}.png"
        # plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        # plt.close()
        # print(f"Saved individual catchment area map to: {output_path}")

        fig, ax = plt.subplots(figsize=(4, 4))
        err_img = err_abs_signed[i, :, :, 0]
        err_img_masked = np.where(np.abs(err_img) > error_threshold, err_img, np.nan)
        err_img_masked = fine_visible(err_img_masked)
        im = ax.imshow(err_img_masked, cmap=cmap_reds)
        ax.set_title(f"GT err abs signed (>{error_threshold}) — S. {indices[i]}")
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Error Abs", rotation=270, labelpad=15)
        output_path = os.path.splitext(save_path)[0] + f"_abs_err_{i+1}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved individual signed abs error map to: {output_path}")
        txt_output_path = os.path.splitext(save_path)[0] + f"_abs_err_{i+1}.txt"
        np.savetxt(txt_output_path, err_img_masked, fmt="%.6f")
        print(f"Saved individual signed abs error array to: {txt_output_path}")

        fig, ax = plt.subplots(figsize=(4, 4))
        GT_img = y_np[i, :, :, 0]
        GT_img_masked = np.where(np.abs(GT_img) > error_threshold, GT_img, np.nan)
        GT_img_masked = fine_visible(GT_img_masked)
        im = ax.imshow(GT_img_masked, cmap='winter')
        ax.set_title(f"GT — S. {indices[i]}")
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("GT", rotation=270, labelpad=15)
        output_path = os.path.splitext(save_path)[0] + f"_GT_{i+1}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved GT map to: {output_path}")
        txt_output_path = os.path.splitext(save_path)[0] + f"_GT_{i+1}.txt"
        np.savetxt(txt_output_path, GT_img_masked, fmt="%.6f")
        print(f"Saved individual GT array to: {txt_output_path}")
        fig, ax = plt.subplots(figsize=(4, 4))
        diff_binary_img = diff_binary[i, :, :, 0]
        diff_binary_img = np.where(plot_domain, diff_binary_img, np.nan) if plot_domain is not None else diff_binary_img
        cmap_binary = mcolors.ListedColormap(['blue', 'white', 'red']) # Orden de valores: -1 (Azul), 0 (Blanco), 1 (Rojo)  
        im = ax.imshow(diff_binary_img, cmap=cmap_binary, vmin=-1, vmax=1)
        ax.set_title(f"Bin Err: Blue=FN, Red=FP — S. {indices[i]}")
        ax.axis('off')
        output_path = os.path.splitext(save_path)[0] + f"_bin_err_{i+1}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved Bin Err map to: {output_path}")
        txt_output_path = os.path.splitext(save_path)[0] + f"_bin_err_{i+1}.txt"
        np.savetxt(txt_output_path, diff_binary_img, fmt="%.6f")
        print(f"Saved individual Bin Err array to: {txt_output_path}")
        

        fig, ax = plt.subplots(figsize=(4, 4))
        X_coarse_img = coarse_cropped_np[i, :, :, 0]
        X_Coarse_img_masked = np.where(np.abs(X_coarse_img) > error_threshold, X_coarse_img, np.nan)
        im = ax.imshow(X_Coarse_img_masked, cmap='winter')
        ax.set_title(f"X_Coarse_cropped — S. {indices[i]}")
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("X_Coarse_cropped", rotation=270, labelpad=15)
        output_path = os.path.splitext(save_path)[0] + f"_coarse_cropped_{i+1}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved X_Coarse_cropped map to: {output_path}")
        txt_output_path = os.path.splitext(save_path)[0] + f"_coarse_cropped_{i+1}.txt"
        np.savetxt(txt_output_path, X_Coarse_img_masked, fmt="%.6f")
        print(f"Saved individual X_Coarse_cropped array to: {txt_output_path}")

        # fig, ax = plt.subplots(figsize=(4, 4))
        # err_rel_signed_img = err_rel_signed_np[i, :, :, 0]
        # err_rel_signed_masked = np.where(np.abs(GT_img) > error_threshold, err_rel_signed_img, np.nan)
        # im = ax.imshow(err_rel_signed_masked, cmap='seismic')
        # ax.set_title(f"GT err rel signed (>{error_threshold}) — S. {indices[i]}")
        # ax.axis('off')
        # cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # cbar.ax.set_ylabel("% Error Rel", rotation=270, labelpad=15)
        # output_path = os.path.splitext(save_path)[0] + f"_rel_err_{i+1}.png"
        # plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        # plt.close()
        # print(f"Saved individual signed relative error map to: {output_path}")
        # txt_output_path = os.path.splitext(save_path)[0] + f"_rel_err_{i+1}.txt"
        # np.savetxt(txt_output_path, err_rel_signed_masked, fmt="%.6f")
        # print(f"Saved individual signed relative error array to: {txt_output_path}")

        fig, ax = plt.subplots(figsize=(4, 4))
        pred_np_im = pred_np[i, :, :, 0]
        pred_np_masked = np.where(np.abs(pred_np_im) > error_threshold, pred_np_im, np.nan)
        pred_np_masked = fine_visible(pred_np_masked)
        im = ax.imshow(pred_np_masked, cmap='winter')
        ax.set_title(f"UNet Prediction — S. {indices[i]}")
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Prediction", rotation=270, labelpad=15)
        output_path = os.path.splitext(save_path)[0] + f"_pred_{i+1}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved UNet prediction map to: {output_path}")
        txt_output_path = os.path.splitext(save_path)[0] + f"_pred_{i+1}.txt"
        np.savetxt(txt_output_path, pred_np_masked, fmt="%.6f")
        print(f"Saved individual UNet prediction array to: {txt_output_path}")
    return pred_np

# def load_true_and_pred_as_numpy(predictions_root):
#     """
#     Loads and concatenates prediction and ground truth data from multiple batch files stored in rank-specific directories.
#     Args:
#         predictions_root (str): Path to the root directory containing rank_* subdirectories with prediction files.
#     Returns:
#         tuple:
#             - all_trues (np.ndarray): Flattened array of all ground truth values loaded from the batch files.
#             - all_preds (np.ndarray): Flattened array of all predicted values loaded from the batch files.
#     Note:
#         - The function searches for subdirectories matching 'rank_*' within the given root directory.
#         - Within each rank directory, it loads all files matching 'batch*.pt', expecting each to contain a dictionary with 'pred' and 'true' tensors.
#         - All predictions and ground truths are flattened and concatenated into large NumPy arrays.
#         - Useful for aggregating results from distributed or batched inference runs.
#     """

#     all_preds = []
#     all_trues = []

#     # Buscar todas las carpetas rank_*
#     rank_dirs = sorted(glob.glob(os.path.join(predictions_root, "rank_*")))

#     for rank_dir in rank_dirs:
#         print(f"Cargando predicciones de {rank_dir}...")
#         pt_files = sorted(glob.glob(os.path.join(rank_dir, "batch*.pt")))

#         for f in pt_files:
#             data = torch.load(f, map_location="cpu")
#             all_preds.append(data["pred"].numpy().flatten())
#             all_trues.append(data["true"].numpy().flatten())

#     # Concatenar en arrays gigantes
#     all_preds = np.concatenate(all_preds)
#     all_trues = np.concatenate(all_trues)

#     print(f"Total puntos cargados: {all_preds.shape[0]}")
#     return all_trues, all_preds

def scatter_plot_sim_vs_pred(sim_points: np.ndarray, pred_points: np.ndarray, variable='depth', units='m', savepath='scatter_plot_sim_vs_pred.png'):
    """
    Genera un gráfico comparando puntos simulados y predichos, ignorando NaNs.
    Args:
        sim_points (np.ndarray): Array de valores simulados.
        pred_points (np.ndarray): Array de valores predichos.
        variable (str): Nombre de la variable.
        units (str): Unidades.
        savepath (str): Ruta de guardado.
    """
    
    # 1. Asegurar que son numpy arrays y aplanarlos (1D)
    sim_points = np.array(sim_points).flatten()
    pred_points = np.array(pred_points).flatten()

    # 2. REEMPLAZO DE NaNs POR CERO
    # np.nan_to_num reemplaza nan con 0.0 (y también maneja infinitos si los hubiera)
    sim_points = np.nan_to_num(sim_points, nan=0.0)
    pred_points = np.nan_to_num(pred_points, nan=0.0)

    # 3. Comprobación de seguridad: Si no quedan datos tras limpiar, salir
    if len(sim_points) == 0:
        print("ADVERTENCIA: Los arrays de entrada están vacíos.")
        return


    min_val = min(sim_points.min(), pred_points.min())
    max_val = max(sim_points.max(), pred_points.max())

    plt.figure(figsize=(8, 8))
    
    # Hexbin maneja bien grandes cantidades de datos
    hb = plt.hexbin(pred_points, sim_points, gridsize=200, cmap='viridis', mincnt=1, bins='log')
    
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--') 
    
    plt.xlabel(f'Predicted {variable} ({units})', fontsize=18)
    plt.ylabel(f'Simulated {variable} ({units})', fontsize=18)
    plt.grid(False)

    # Cálculos para ejes (redondeos)
    x_min = round(pred_points.min(), 1)
    x_max = round(pred_points.max(), 1)
    y_min = round(sim_points.min(), 1)
    y_max = round(sim_points.max(), 1)

    # Generación de Ticks
    x_min_tick = int(np.floor(x_min))
    x_max_tick = int(np.ceil(x_max))
    if x_min_tick == x_max_tick:
        x_ticks = np.array([x_min_tick])
    else:
        x_ticks = np.linspace(x_min_tick, x_max_tick, num=5)
    
    x_ticks = np.unique(np.concatenate(([x_min_tick, x_max_tick], x_ticks))).astype(int)

    y_min_tick = int(np.floor(y_min))
    y_max_tick = int(np.ceil(y_max))
    if y_min_tick == y_max_tick:
        y_ticks = np.array([y_min_tick])
    else:
        y_ticks = np.linspace(y_min_tick, y_max_tick, num=5)
        
    y_ticks = np.unique(np.concatenate(([y_min_tick, y_max_tick], y_ticks))).astype(int)

    # Asegurar extremos
    x_ticks_full = np.unique(np.append(x_ticks, [x_min, x_max]))
    y_ticks_full = np.unique(np.append(y_ticks, [y_min, y_max]))
    
    plt.xticks(x_ticks_full, fontsize=18)
    plt.yticks(y_ticks_full, fontsize=18)

    # Estadísticas
    rho, _ = pearsonr(sim_points, pred_points)
    rmse = np.sqrt(np.mean((sim_points - pred_points) ** 2))

    # Texto de estadísticas
    plt.text(
        x_min + 0.05 * (x_max - x_min),
        y_max - 0.05 * (y_max - y_min),
        f'$\\rho$ = {rho:.3f}\nRMSE = {rmse:.3f}',
        fontsize=18,
        verticalalignment='top',
        horizontalalignment='left'
    )

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Barra de color
    ax = plt.gca()
    cb_ax = inset_axes(ax, width="4%", height="20%", loc='lower right', borderpad=4)
    cbar = plt.colorbar(hb, cax=cb_ax)
    cbar.ax.set_title('Density', fontsize=12, fontweight='bold', pad=10)
    cbar.ax.text(-0.9, 0, 'Low', va='bottom', ha='right', fontsize=10, transform=cbar.ax.transAxes)
    cbar.ax.text(-0.9, 1, 'High', va='top', ha='right', fontsize=10, transform=cbar.ax.transAxes)
    cbar.ax.tick_params(labelsize=8) 
    # cbar.formatter.set_powerlimits((0, 0)) # Si queremos formatear los números (ej. notación científica o enteros)
    # cbar.update_ticks()
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en: {savepath}")
    plt.close()

def histogram_errors_plot_sim_vs_pred(sim_points: np.ndarray, pred_points: np.ndarray, variable='depth', units='m', savepath='histogram_errors_plot_sim_vs_pred.png'):
    """
    Genera dos histogramas de errores (Sim - Pred): uno lineal y uno logarítmico.
    """
    
    # 1. OPTIMIZACIÓN DE DATOS (Ravel + isfinite)
    sim_points = np.ravel(sim_points)
    pred_points = np.ravel(pred_points)

    # REEMPLAZO DE NaNs POR CERO
    # np.nan_to_num reemplaza nan con 0.0 (y también maneja infinitos si los hubiera)
    sim_points = np.nan_to_num(sim_points, nan=0.0)
    pred_points = np.nan_to_num(pred_points, nan=0.0)

    if len(sim_points) == 0:
        print("ADVERTENCIA: No hay puntos válidos para graficar.")
        return

    # 2. CÁLCULO DE ESTADÍSTICAS (Una sola vez)
    errors = sim_points - pred_points
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    rmse = np.sqrt(np.mean(np.square(errors)))

    # Calculamos la desviación máxima absoluta para crear un rango simétrico. Si el error es 0 perfecto, damos un pequeño margen para evitar error de ploteo
    max_abs_err = np.max(np.abs(errors))
    if max_abs_err == 0: 
        max_abs_err = 1.0

    # Definimos los bordes de los bins manualmente.
    # Usamos 51 bins (IMPAR) para que haya una barra central exacta.
    # linspace pide el número de bordes, así que pedimos 52 bordes para obtener 51 barras.
    bins_edges = np.linspace(-max_abs_err, max_abs_err, 52)

    # Preparamos el texto de estadísticas
    textstr = '\n'.join((
        f'RMSE = {rmse:.3f}',
        f'Mean = {mean_error:.3f}',
        f'Std  = {std_error:.3f}',
        f'N    = {len(errors)}'))
    
    # Preparamos los nombres de archivo
    # Separa 'carpeta/archivo' de '.png'
    base_name, ext = os.path.splitext(savepath)
    
    # Lista de configuraciones: (Es_Logaritmica, Ruta_Salida)
    configs = [
        (False, savepath),
        (True, f"{base_name}_logscale{ext}")
    ]

    # 3. BUCLE DE GRAFICADO (Genera las 2 gráficas)
    for is_log, current_savepath in configs:
        
        plt.figure(figsize=(10, 6))
        
        # plt.hist tiene un parámetro 'log' nativo que maneja las barras mejor que cambiar el eje después
        plt.hist(errors, bins=bins_edges, color='skyblue', edgecolor='black', alpha=0.7, zorder=3, log=is_log)
        
        # Líneas de referencia
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error', zorder=4)
        plt.axvline(x=mean_error, color='green', linestyle=':', linewidth=2, label='Mean Error', zorder=4)

        # Etiquetas y Títulos
        scale_label = " (Log Scale)" if is_log else ""
        plt.xlabel(f'Error ({variable} in {units}) [Sim - Pred]', fontsize=14)
        plt.ylabel(f'Frequency{scale_label}', fontsize=14)
        # plt.title(f'Error Distribution: {variable}{scale_label}', fontsize=14)
        
        # Ticks
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        # Grid (el grid menor ayuda mucho en log scale)
        if is_log:
            plt.grid(True, which="both", ls="-", alpha=0.2, zorder=0)
        else:
            plt.grid(axis='y', alpha=0.5, linestyle='--', zorder=0)

        plt.legend(fontsize=12)

        # Caja de texto (Stats)
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax = plt.gca()
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', horizontalalignment='right', bbox=props)

        # Guardado
        os.makedirs(os.path.dirname(current_savepath), exist_ok=True)
        plt.savefig(current_savepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico guardado: {current_savepath}")

def confusion_matrix_sim_vs_pred(sim_points: np.ndarray, pred_points: np.ndarray, variable='depth', units='m', limits=[0.0, 0.1, 0.5, 1.0], savepath='confusion_matrix_sim_vs_pred.png'):
    """
    Genera una matriz de confusión visual entre valores simulados y predichos, mostrando la frecuencia relativa de coincidencias en rangos definidos.
    Args:
        sim_points (np.ndarray): Array de valores simulados.
        pred_points (np.ndarray): Array de valores predichos.
        variable (str, opcional): Nombre de la variable a mostrar en los ejes. Por defecto 'depth'.
        units (str, opcional): Unidades de la variable. Por defecto 'm'.
        limits (list, opcional): Límites para los rangos de la matriz de confusión. Por defecto [0.0, 0.1, 0.5, 1.0].
        savepath (str, opcional): Ruta donde guardar la imagen generada. Por defecto 'confusion_matrix_sim_vs_pred.png'.
    El gráfico incluye:
        - Matriz de confusión normalizada (frecuencia relativa).
        - Etiquetas de rangos en los ejes.
        - Frecuencia numérica en cada celda, con color adaptativo para legibilidad.
        - Ticks ajustados y layout optimizado para presentación.
    """
    extended_limits = limits + [max(sim_points.max(), pred_points.max()) + 1]
    bins = len(extended_limits) - 1

    conf_matrix, x_edges, y_edges = np.histogram2d(sim_points, pred_points, bins=[extended_limits, extended_limits])
    conf_matrix_freq = conf_matrix / conf_matrix.sum()

    plt.figure(figsize=(8, 8))

    # Crear colormap personalizado que garantice blanco para 0
    colors = ['white', '#deebf7', '#9ecae1', '#3182bd', '#08519c']
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_blues', colors, N=n_bins)
    
    # Asegurar que exactamente 0 mapee a blanco
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=conf_matrix_freq.max())

    
    # Invertir el eje y para que los valores menores estén arriba
    plt.imshow(conf_matrix_freq, interpolation='nearest', cmap=cmap, origin='upper', norm=norm)

    plt.xlabel(f'Predicted {variable} ({units})', fontsize=18)
    plt.ylabel(f'Simulated {variable} ({units})', fontsize=18)
    plt.xticks(np.arange(bins) - 0.5, labels=[f'{extended_limits[i]:.2f}' for i in range(bins)], rotation=45, fontsize=18)
    plt.yticks(np.arange(bins) - 0.5, labels=[f'{extended_limits[i]:.2f}' for i in range(bins)], fontsize=18)

    for i in range(bins):
        for j in range(bins):
            freq = conf_matrix_freq[i, j]
            rgba = cmap(norm(freq))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = 'black' if luminance > 0.7 else 'white'
            plt.text(j, i, f'{freq:.3f}', ha='center', va='center', color=text_color, fontsize=18)

    plt.tight_layout()
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=300)

def plot_wet_area_vs_time(wet_areas_sim: np.ndarray, wet_areas_pred: np.ndarray, 
                    wet_areas_int: np.ndarray, times: np.ndarray, variable='Depth', units='m',
                    limits=[0.1, 0.5, 1.0], savepath='wet_area_vs_time.png'):
    """
    Genera un gráfico comparativo de área inundada vs tiempo para tres conjuntos de datos
    (simulados, predichos e interpolados) en diferentes rangos de profundidad.
    """
    
    n_ranges = len(limits)
    datasets = [wet_areas_sim, wet_areas_pred, wet_areas_int]
    dataset_names = ['Sim', 'Pred', 'Int']
    
    for i, dataset in enumerate(datasets):
        if dataset.shape[0] != n_ranges:
            raise ValueError(f"{dataset_names[i]} debe tener {n_ranges} filas (rangos), pero tiene {dataset.shape[0]}")
        if len(times) != dataset.shape[1]:
            raise ValueError(f"times debe tener {dataset.shape[1]} elementos para {dataset_names[i]}, pero tiene {len(times)}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#77E5E9", "#391BE0", "#030207", "#A51392", "#460246"]
    colors = colors[:n_ranges]
    
    depth_labels = []
    for i in range(n_ranges):
        if i == n_ranges - 1:
            depth_labels.append(f'> {limits[-1]} {units}')
        else:
            depth_labels.append(f'{limits[i]} - {limits[i+1]} {units}')
    
    methodologies = [
        ('Sim.', wet_areas_sim, '-', 2.0),
        ('Pred.', wet_areas_pred, '--', 2.0),
        ('Int.', wet_areas_int, ':', 2.0)
    ]
    
    for method_name, data, line_style, line_width in methodologies:
        for i in range(n_ranges):
            ax.plot(times, data[i], 
                    color=colors[i],
                    linestyle=line_style,
                    linewidth=line_width,
                    alpha=0.9)
    
    legend_elements = []
    
    legend_elements.append(
        Rectangle((0, 0), 1, 1, fc="none", ec="none", alpha=0, label=f'{variable} ({units})')
    )
    for i in range(n_ranges):
        legend_elements.append(
            Line2D([0], [0], color=colors[i], linestyle='-', 
                  linewidth=3, label=depth_labels[i])
        )
    legend_elements.append(
        Rectangle((5, 0), 1, 1, fc="none", ec="none", alpha=0, label=f'')
    )
    for method_name, _, line_style, line_width in methodologies:
        legend_elements.append(
            Line2D([0], [0], color='black', linestyle=line_style, 
                  linewidth=line_width, label=method_name)
        )

    ax.legend(handles=legend_elements, 
             loc='upper left',
             frameon=True, 
             fancybox=True, 
             shadow=True,
             ncol=2,
             bbox_to_anchor=(0.02, 0.98),
             fontsize=14)

    ax.set_xlabel('Time (h)', fontsize=14)
    ax.set_ylabel('Area (km²)', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, max(times))
    max_value = max(np.max(wet_areas_sim), np.max(wet_areas_pred), np.max(wet_areas_int))
    ax.set_ylim(0, max_value * 1.15)
    xticks = np.arange(0, int(times[-1]) + 1)
    if len(xticks) > 10:
        step = max(1, int(np.ceil(len(xticks) / 9)))
        xticks = np.unique(np.concatenate(([0, int(times[-1])], xticks[::step])))
    else:
        xticks = np.unique(np.concatenate(([0, int(times[-1])], xticks)))
    ax.set_xticks(xticks)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=300, bbox_inches='tight')

    print(f"Gráfico comparativo guardado en: {savepath}")
    
def calculate_wet_area(image, cellsize, limits=[0.1, 0.5, 1.0]):
    """
    Calcula el área mojada para diferentes rangos de profundidad/velocidad en una imagen.
    
    Arguments
    ----------
    image : np.ndarray
        Array 2D con valores de profundidad/velocidad (H filas × W columnas)
    cellsize : float
        Tamaño de cada celda/píxel en metros (para calcular área real)
    limits : list, optional
        Límites de profundidad/velocidad para clasificar las áreas, por defecto [0.1, 0.5, 1.0]
    
    Returns
    -------
    np.ndarray
        Array 1D con las áreas mojadas para cada rango de profundidad/velocidad.
        Forma: (len(limits),) donde cada elemento corresponde al área en km²
    """
    
    # Validar que la imagen sea 2D
    if image.ndim != 2:
        raise ValueError("La imagen debe ser un array 2D")
    
    # Validar que cellsize sea positivo
    if cellsize <= 0:
        raise ValueError("cellsize debe ser un valor positivo")
    
    n_ranges = len(limits)
    areas = np.zeros(n_ranges)
    
    # Calcular el área de cada celda en km²
    cell_area_km2 = (cellsize * cellsize) / 1_000_000  # m² to km²
    
    # Contar celdas para cada rango de profundidad/velocidad
    for i in range(n_ranges):
        if i == n_ranges - 1:
            # Último rango (especial): mayor que el último límite
            mask = (image >= limits[i])
        else:
            # Resto de rangos inicial o intermedios: entre limits[i] y limits[i+1]
            mask = (image >= limits[i]) & (image < limits[i+1])
        
        # Contar número de celdas en este rango y calcular área total
        num_cells = np.sum(mask)
        areas[i] = num_cells * cell_area_km2
    
    return areas

def plot_metrics(ruta_pkl, metricas, savepath='metrics_evolucion.png'):
    """
    Genera un gráfico de la evolución de métricas (por ejemplo, pérdida de entrenamiento y validación) a lo largo de las épocas.
    Args:
        ruta_pkl (str): Ruta al archivo .pkl que contiene las métricas por época.
        metricas (list): Lista de nombres de métricas a graficar.
        savepath (str, opcional): Ruta donde guardar la imagen generada.
    El gráfico incluye:
        - Curvas de las métricas seleccionadas.
        - Etiquetas personalizadas y escala logarítmica en el eje Y.
        - Leyenda y formato optimizado para presentación.
    """
    with open(ruta_pkl, 'rb') as f:
        contenido = pickle.load(f)

    if isinstance(contenido, dict):
        keys = list(contenido.keys())
        num_epochs = len(contenido[keys[0]])
        epochs = list(range(1, num_epochs + 1))
    elif isinstance(contenido, list) and len(contenido) > 0 and isinstance(contenido[0], dict):
        keys = list(contenido[0].keys())
        num_epochs = len(contenido)
        epochs = list(range(1, num_epochs + 1))
    else:
        print("Formato de archivo pkl no soportado.")
        return

    plt.figure()
    label_map = {
        'val_loss': 'Validation loss',
        'val_loss_sync': 'Validation loss',
        'loss_sync': 'Training loss',
        'loss': 'Training loss'
    }

    for metrica in metricas:
        if metrica not in keys:
            print(f"Métrica '{metrica}' no encontrada en los datos.")
            continue

        if isinstance(contenido, dict):
            valores = contenido[metrica]
        else:
            valores = [d.get(metrica, None) for d in contenido]

        label = label_map.get(metrica, metrica)
        plt.plot(epochs, valores, label=label, linewidth=3)

    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Metrics', fontsize=18)
    plt.yscale('log')
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.gcf().set_size_inches(12, 6)  # Aumenta la anchura de la imagen
    plt.tight_layout()
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de métricas guardado en {savepath}")

def calculate_rain_x_time_x_surface_in_runoff_map(path_runoff_map, path_hyetographs, cellsize_in_m):
    # Leer runoff map y contar ocurrencias de cada entero
    runoff_map = np.loadtxt(path_runoff_map, dtype=int)
    max_label = runoff_map.max()
    counts = np.bincount(runoff_map.flatten(), minlength=max_label + 1)

    # Leer hyetographs, saltando la primera fila (header con %)
    with open(path_hyetographs, 'r') as f:
        lines = f.readlines()
    data = [line.strip() for line in lines if not line.strip().startswith('%')]
    hyeto_data = np.genfromtxt(data, delimiter=',')
    if hyeto_data.ndim == 1:
        hyeto_data = hyeto_data[np.newaxis, :]
    times = hyeto_data[:, 0]
    hyetographs = hyeto_data[:, 1:]

    # Calcular la integral bajo cada hyetograph (usando trapz)
    integrals = np.trapz(hyetographs, x=times, axis=0)

    assert len(counts) == hyetographs.shape[1], f"counts ({len(counts)}) y hyetographs ({hyetographs.shape[1]}) deben tener la misma longitud"

    print("Counts:", counts)
    print("Integrals:", integrals)
    product = [counts[i]*integrals[i]*cellsize_in_m*cellsize_in_m for i in range(len(counts))]
    print("Product of rain*time*surface for each zone:", product)
    print("Sum of all zones (mm*m²):", sum(product))

def create_constant_rainfall_hyetographs(rain_intensities, duration, output_filepath):
    """
    Genera un gráfico de hietogramas de lluvia constante para varias intensidades.

    Parámetros
    ----------
    rain_intensities : list of float
        Lista con las intensidades de lluvia (mm/h).
    duration : float
        Duración del evento de lluvia en horas.
    output_filepath : str
        Ruta donde se guardará la imagen (formato .png, .jpg, etc.).
    """
    plt.figure(figsize=(10,7))
    
    time = list(range(duration+1))  # Tiempo en horas
    
    # Colores automáticos
    for intensity in rain_intensities:
        plt.plot(time, [intensity]*len(time), label=f"{intensity} mm/h", linewidth=3)
    
    plt.xlabel("Time (h)", fontsize=14)
    plt.ylabel("Rain intensity (mm/h)", fontsize=14)
    plt.title("Constant Rainfall Hyetographs", fontsize=16)
    plt.legend(title="Intensity", loc="upper right", fontsize=14, title_fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Guardar y cerrar
    plt.savefig(output_filepath, dpi=300)
    plt.close()
    print(f"Gráfico guardado en {output_filepath}")

def process_logs_outputs_into_files(folder_paths, save_path, number_of_runs_to_write_per_parameter=50):
    """
    Analiza archivos .o, extrae métricas y guarda los resultados, limitando el número de filas
    por cada combinación de parámetros y calculando las estadísticas sobre esos mismos datos.
    """
    os.makedirs(save_path, exist_ok=True)
    # --- Cabeceras y Listas de Resultados (sin cambios) ---
    header_common = [
        "global_loss", "global_mse", "global_mae", "global_masked_mse", "global_masked_mae",
        "global_vol_error", "global_far", "global_hit", "global_csi",
        "real_loss", "real_mse", "real_mae", "real_masked_mse", "real_masked_mae",
        "real_vol_error", "real_far", "real_hit", "real_csi"
    ]
    header_MAEMassWetDry = ["name", "lambda", "beta"] + header_common
    header_MAESoftcsi = ["name", "alpha"] + header_common
    # ... (el resto de headers y listas de resultados)
    header_MSESoftcsi = ["name", "alpha"] + header_common
    header_MAEWetSoftcsi = ["name", "alpha"] + header_common
    header_MAEMSE = ["name", "omega"] + header_common
    header_failed_run = ["name", "FailReason"]
    results_mae_mass_wetdry, results_mae_softcsi, results_mse_softcsi, results_mae_wet_softcsi, results_mae_mse, results_failed_run = [], [], [], [], [], []

    # --- Búsqueda y Procesamiento de Archivos (sin cambios) ---
    # (Toda la lógica de búsqueda, filtrado por job, y parseo de cada archivo es idéntica)
    def natural_sort_key(s): return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', os.path.basename(s))]
    files = []
    for folder in folder_paths:
        files.extend(sorted(glob.glob(os.path.join(folder, "Models*.o")), key=natural_sort_key))
    best_by_conf = {}
    for fp in files:
        name = os.path.basename(fp).strip()
        m = re.match(r"(.+)-(\d+)\.o$", name)
        if m:
            conf, job = m.group(1), int(m.group(2))
            prev = best_by_conf.get(conf)
            if (prev is None) or (job > prev[0]): best_by_conf[conf] = (job, fp)
        else: best_by_conf[name] = (-1, fp)
    files = sorted([v[1] for v in best_by_conf.values()], key=natural_sort_key)
    for file in files:
        file_name = os.path.basename(file)
        with open(file, "r", encoding="utf-8", errors="ignore") as f: content = f.read()
        which_loss = re.search(r"WHICH_LOSS:\s*(\S+)", content)
        alpha_csi_mae = re.search(r"ALPHA_CSI:\s*([0-9.eE+-]+)", content)
        beta = re.search(r"BETA_MASS:\s*([0-9.eE+-]+)", content)
        lambd = re.search(r"LAMBDA_DRY:\s*([0-9.eE+-]+)", content)
        alpha_csi_mse = re.search(r"ALPHA_CSI_MSE:\s*([0-9.eE+-]+)", content)
        alpha_csi_wet_mae = re.search(r"ALPHA_CSI_WETMAE:\s*([0-9.eE+-]+)", content)
        omega = re.search(r"OMEGA_MSE:\s*([0-9.eE+-]+)", content)
        which_loss = which_loss.group(1) if which_loss else ""
        alpha_csi_mae = float(alpha_csi_mae.group(1)) if alpha_csi_mae else float('nan')
        beta = float(beta.group(1)) if beta else float('nan')
        lambd = float(lambd.group(1)) if lambd else float('nan')
        alpha_csi_mse = float(alpha_csi_mse.group(1)) if alpha_csi_mse else float('nan')
        alpha_csi_wet_mae = float(alpha_csi_wet_mae.group(1)) if alpha_csi_wet_mae else float('nan')
        omega = float(omega.group(1)) if omega else float('nan')
        def extract_metrics(block):
            def get(pattern):
                m = re.search(pattern, block)
                return float(m.group(1)) if m else float('nan')
            return [get(p) for p in [r"Loss:\s*([0-9.eE+-]+)", r"MSE:\s*([0-9.eE+-]+)", r"MAE:\s*([0-9.eE+-]+)", r"Masked MSE:\s*([0-9.eE+-]+)", r"Masked MAE:\s*([0-9.eE+-]+)", r"Volume Rel Error.*\(%\):\s*([0-9.eE+-]+)", r"FAR=([0-9.eE+-]+)%", r"HIT=([0-9.eE+-]+)%", r"CSI=([0-9.eE+-]+)%"]]
        global_block = re.search(r"=== Global Test Results ===(.*?)={5,}", content, re.DOTALL)
        real_block = re.search(r"=== Real Case Results ===(.*?)={5,}", content, re.DOTALL)
        global_metrics = extract_metrics(global_block.group(1)) if global_block else [float('nan')]*9
        real_metrics = extract_metrics(real_block.group(1)) if real_block else [float('nan')]*9
        if which_loss == "MAEMassConservationAndWetDryLoss": results_mae_mass_wetdry.append([file_name, lambd, beta] + global_metrics + real_metrics)
        elif which_loss == "MAESoftCSILoss": results_mae_softcsi.append([file_name, alpha_csi_mae] + global_metrics + real_metrics)
        elif which_loss == "MSESoftCSILoss": results_mse_softcsi.append([file_name, alpha_csi_mse] + global_metrics + real_metrics)
        elif which_loss == "WetMAEWithSoftCSILoss": results_mae_wet_softcsi.append([file_name, alpha_csi_wet_mae] + global_metrics + real_metrics)
        elif which_loss == "MAEMSELoss": results_mae_mse.append([file_name, omega] + global_metrics + real_metrics)
        else: results_failed_run.append([file_name, "WHICH_LOSS missing or invalid"])

    # --- Formateo y Guardado de Archivos ---
    output_configs = [
        (results_mae_mass_wetdry, os.path.join(save_path, "MAEMassConservationAndWetDryLoss.txt"), os.path.join(save_path, "MAEMassConservationAndWetDryLoss_Averages.txt"), header_MAEMassWetDry),
        (results_mae_softcsi, os.path.join(save_path, "MAESoftCSILoss.txt"), os.path.join(save_path, "MAESoftCSILoss_Averages.txt"), header_MAESoftcsi),
        (results_mse_softcsi, os.path.join(save_path, "MSESoftCSILoss.txt"), os.path.join(save_path, "MSESoftCSILoss_Averages.txt"), header_MSESoftcsi),
        (results_mae_wet_softcsi, os.path.join(save_path, "WetMAEWithSoftCSILoss.txt"), os.path.join(save_path, "WetMAEWithSoftCSILoss_Averages.txt"), header_MAEWetSoftcsi),
        (results_mae_mse, os.path.join(save_path, "MAEMSELoss.txt"), os.path.join(save_path, "MAEMSELoss_Averages.txt"), header_MAEMSE),
    ]
    def format_row(row, header, length_flag=35):
        row_filled = list(zip_longest(row, header, fillvalue="NaN"))
        return "".join(str("NaN" if isinstance(x[0], float) and np.isnan(x[0]) else x[0]).ljust(length_flag)[:length_flag] for x in row_filled)
    def format_header(header, length_flag=35):
        return "".join(str(h).ljust(length_flag)[:length_flag] for h in header)
    
    # --- Identificación y Extracción de Runs Fallidos (sin cambios) ---
    # (El código para identificar y guardar los FailedRuns es idéntico)
    with open(os.path.join(save_path, "FailedRuns.txt"), "w") as f: f.write(format_header(header_failed_run) + "\n")
    if results_failed_run:
        with open(os.path.join(save_path, "FailedRuns.txt"), "a") as f:
            for row in results_failed_run: f.write(format_row(row, header_failed_run) + "\n")
    for filtered_results, path, _, header in output_configs:
        metric_indices_range = range(header.index("global_loss"), header.index("real_csi") + 1)
        for row in filtered_results[:]:
            row_floats = [float(x) if str(x).replace('.', '', 1).isdigit() else np.nan for x in row]
            metrics_values = [row_floats[i] for i in metric_indices_range]
            if any(np.isnan(v) or v <= 0 for v in metrics_values):
                reasons = [f"{header[i]} is invalid" for i, v in enumerate(row_floats) if i in metric_indices_range and (np.isnan(v) or v <= 0)]
                with open(os.path.join(save_path, "FailedRuns.txt"), "a") as f: f.write(format_row([row[0], "; ".join(reasons)], header_failed_run) + "\n")
                filtered_results.remove(row)
    runs_malas_reentrenar, runs_malas_inferencia = [], []
    with open(os.path.join(save_path, "FailedRuns.txt"), "r") as f: lines = f.readlines()[1:]
    for line in lines:
        if not line.strip(): continue
        name, reason = line.strip().split()[0], " ".join(line.strip().split()[1:])
        match = re.match(r".*_(\d+)_(\d+)_(\d+)-.*", name)
        if match:
            model, n_config, n_try = map(int, match.groups())
            if "WHICH_LOSS" in reason or "GLOBAL" in reason: runs_malas_reentrenar.append((model, n_config, n_try))
            else: runs_malas_inferencia.append((model, n_config, n_try))
    with open(os.path.join(save_path, "FailedRuns_Reentrenar.txt"), "w") as f: f.write("% model n_config n_try\n"); [f.write(f"{r[0]} {r[1]} {r[2]}\n") for r in runs_malas_reentrenar]
    with open(os.path.join(save_path, "FailedRuns_Inferencia.txt"), "w") as f: f.write("% model n_config n_try\n"); [f.write(f"{r[0]} {r[1]} {r[2]}\n") for r in runs_malas_inferencia]

    # ============================================================
    # MODIFICADO: Escritura y cálculo de medias sobre datos limitados
    # ============================================================
    for filtered_results, path, avg_path, header in output_configs:
        if not filtered_results:
            print(f"\nNo hay resultados válidos para {os.path.basename(path)}")
            continue

        param_cols = [h for h in header if h not in ["name"] + header_common]
        param_indices = [header.index(h) for h in param_cols]
        metric_cols = [h for h in header if h in header_common]
        metric_indices = [header.index(h) for h in metric_cols]
        
        # 1. Agrupar todos los resultados válidos por parámetros
        all_grouped_by_params = {}
        for row in filtered_results:
            params_key = tuple(row[i] for i in param_indices)
            all_grouped_by_params.setdefault(params_key, []).append(row)
        
        # 2. Crear la lista final de resultados a escribir (con el límite)
        results_to_write = []
        print(f"\n📝 Resumen para {os.path.basename(path)}:")
        for params, rows in all_grouped_by_params.items():
            num_found = len(rows)
            print(f"  - Para parámetros {params}: Se encontraron {num_found} runs válidas.", end="")
            
            if num_found < number_of_runs_to_write_per_parameter:
                print(f" (Advertencia: menos de las {number_of_runs_to_write_per_parameter} solicitadas)")
            else:
                print()
            
            # Aplicar el límite a las filas que se escribirán
            results_to_write.extend(rows[:number_of_runs_to_write_per_parameter])

        # 3. Guardar el archivo de resultados con las filas limitadas
        with open(path, "w") as f:
            f.write(format_header(header) + "\n")
            for row in results_to_write:
                f.write(format_row(row, header) + "\n")
        print(f"-> Resultados (máx. {number_of_runs_to_write_per_parameter} por config) guardados en {path}.")

        # 4. MODIFICADO: Agrupar de nuevo, pero ahora solo los 'results_to_write'
        avg_groups = {}
        for row in results_to_write:
            params_key = tuple(row[i] for i in param_indices)
            metrics_vals = [float(row[i]) if str(row[i]) not in ["nan", "NaN", "None"] else np.nan for i in metric_indices]
            avg_groups.setdefault(params_key, []).append(metrics_vals)

        # 5. Calcular y guardar promedios sobre los grupos limitados
        avg_rows = []
        for params, metrics_list in avg_groups.items():
            arr = np.array(metrics_list, dtype=float)
            means = np.nanmean(arr, axis=0)
            stds = np.nanstd(arr, axis=0)
            
            avg_row = list(params)
            for m, s in zip(means, stds):
                avg_row += [m, s]
            avg_rows.append(avg_row)

        header_avg = param_cols
        for m in metric_cols:
            header_avg += [f"{m}_mean", f"{m}_std"]
        
        with open(avg_path, "w") as f:
            f.write(format_header(header_avg) + "\n")
            for row in avg_rows:
                f.write(format_row(row, header_avg) + "\n")
        print(f"-> Medias y desviaciones (sobre los runs escritos) guardadas en {avg_path}.")

    return runs_malas_reentrenar, runs_malas_inferencia

def plot_output_results_and_detect_outliers_from_files(
    file_paths, 
    save_paths, 
    which_metrics_to_check, 
    delete_outliers=False, 
    path_of_outliers=None,
    N_big=10,  # <-- Nuevo argumento
    N_low=10   # <-- Nuevo argumento
):
    """
    Lee archivos de resultados, detecta outliers (IQR) y valores extremos (N más grandes/pequeños),
    opcionalmente los borra, y genera gráficos.

    - Detecta outliers usando el método IQR para cada grupo de repeticiones.
    - Detecta los N_big más grandes y N_low más pequeños para cada grupo.
    - Si delete_outliers es True, borra los archivos .o, .e, etc. asociados (solo de outliers IQR).
    - Genera gráficos de 1 o 2 parámetros.

    Parámetros
    ----------
    file_paths : list of str
        Rutas a los archivos .txt con resultados.
    save_paths : list of str
        Rutas donde guardar los gráficos generados.
    which_metrics_to_check : list of str
        Nombres de las columnas (métricas) en las que se buscarán outliers/extremos.
    delete_outliers : bool, opcional
        Si es True, se procederá a borrar los archivos de los outliers (IQR) detectados.
    path_of_outliers : str, opcional
        Ruta al directorio que contiene los archivos de log (.o, .e) a borrar.
        Es obligatorio si delete_outliers es True.
    N_big : int, opcional
        Número de valores más grandes a reportar por grupo.
    N_low : int, opcional
        Número de valores más pequeños a reportar por grupo.
        
    Retorna
    -------
    dict
        Un diccionario con los resultados del análisis:
        { file_path: { metric: {'iqr': [], 'big': [], 'low': []} } }
    """
    if len(file_paths) != len(save_paths):
        raise ValueError("file_paths y save_paths deben tener la misma longitud")

    if delete_outliers and (not path_of_outliers or not os.path.isdir(path_of_outliers)):
        raise ValueError("Si 'delete_outliers' es True, se debe proporcionar un 'path_of_outliers' válido.")

    # El diccionario de retorno ahora contendrá 'iqr', 'big', y 'low' por métrica
    all_analysis_results = {} 

    metrics_full_list = [
        "global_loss", "global_mse", "global_mae", "global_masked_mse", "global_masked_mae",
        "global_vol_error", "global_far", "global_hit", "global_csi",
        "real_loss", "real_mse", "real_mae", "real_masked_mse", "real_masked_mae",
        "real_vol_error", "real_far", "real_hit", "real_csi"
    ]
    metrics_to_plot = [
        "global_mae", "global_masked_mae", "global_vol_error", "global_far", "global_hit", "global_csi",
        "real_mae", "real_masked_mae", "real_vol_error", "real_far", "real_hit", "real_csi"
    ]

    for file_path, save_path in zip(file_paths, save_paths):
        # --- Lectura y Parseo de Datos (sin cambios) ---
        if not os.path.isfile(file_path): continue
        with open(file_path, "r") as f: lines = f.readlines()
        if len(lines) < 2: continue
        header = lines[0].strip().split()
        all_data = np.array([line.strip().split() for line in lines[1:] if line.strip()], ndmin=2)
        names = np.array([line.strip().split()[0] for line in lines[1:] if line.strip()])
        parameters_to_study = [col for col in header if col not in metrics_full_list and col != "name"]
        if not parameters_to_study: continue
        
        param_idx = [header.index(p) for p in parameters_to_study]
        params = np.array([[float(row[i]) for i in param_idx] for row in all_data])

        # --- Detección de Outliers y Extremos POR GRUPO ---
        analysis_for_file = {}
        if len(parameters_to_study) == 1:
            param_values = params[:, 0]
            unique_vals = np.unique(param_values)
            param_name = parameters_to_study[0]
            
            for metric_to_check in which_metrics_to_check:
                if metric_to_check not in header: continue
                metric_idx_check = header.index(metric_to_check)
                metric_data_full_col = all_data[:, metric_idx_check].astype(float)
                
                # Listas para guardar los strings formateados
                outlier_names_for_metric = []
                big_values_for_metric = []
                low_values_for_metric = []

                for val in unique_vals:
                    group_mask = (param_values == val)
                    metric_vals_group = metric_data_full_col[group_mask]
                    names_group = names[group_mask]
                    if metric_vals_group.size == 0: continue

                    # 1. Detección de Outliers (IQR)
                    if len(metric_vals_group) >= 4:
                        q1, q3 = np.percentile(metric_vals_group, [25, 75])
                        iqr = q3 - q1
                        if iqr > 0: # Solo si iqr no es 0
                            lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                            outlier_mask_group = (metric_vals_group < lower_bound) | (metric_vals_group > upper_bound)
                            outlier_names = names_group[outlier_mask_group]
                            if outlier_names.size > 0:
                                for name in outlier_names:
                                    outlier_names_for_metric.append(f"{name} (at {param_name}={val})")
                    
                    # 2. Detección de N_low y N_big
                    sorted_indices = np.argsort(metric_vals_group)
                    
                    # N_low más pequeños
                    actual_N_low = min(N_low, len(sorted_indices))
                    low_indices = sorted_indices[:actual_N_low]
                    for idx in low_indices:
                        name = names_group[idx]
                        value = metric_vals_group[idx]
                        low_values_for_metric.append(f"{name} (value: {value:.4e}) (at {param_name}={val})")
                        
                    # N_big más grandes (invertimos los índices)
                    actual_N_big = min(N_big, len(sorted_indices))
                    big_indices = sorted_indices[-actual_N_big:][::-1] # Tomamos los últimos N y los invertimos
                    for idx in big_indices:
                        name = names_group[idx]
                        value = metric_vals_group[idx]
                        big_values_for_metric.append(f"{name} (value: {value:.4e}) (at {param_name}={val})")

                # Guardar resultados para esta métrica
                if outlier_names_for_metric or big_values_for_metric or low_values_for_metric:
                    analysis_for_file[metric_to_check] = {
                        'iqr': outlier_names_for_metric,
                        'big': big_values_for_metric,
                        'low': low_values_for_metric
                    }
        all_analysis_results[file_path] = analysis_for_file

        # ============================================================
        # Lógica de Borrado de Outliers (IQR)
        # ============================================================
        if delete_outliers and analysis_for_file:
            print(f"\n🗑️  Buscando archivos de outliers (IQR) para borrar en: {path_of_outliers}")
            base_names_to_delete = set()
            
            # 1. Recopilar nombres base de outliers (SOLO de 'iqr')
            for metric, data_dict in analysis_for_file.items():
                outlier_list = data_dict.get('iqr', []) # <-- Solo obtenemos la lista 'iqr'
                if not outlier_list:
                    continue
                
                for outlier_string in outlier_list:
                    filename_part = outlier_string.split(' (at ')[0]
                    base_name = filename_part.split('-')[0]
                    base_names_to_delete.add(base_name)
            
            # 2. Buscar y borrar
            if not base_names_to_delete:
                print("  - No se encontraron outliers (IQR) para borrar.")
                
            for base_name in sorted(list(base_names_to_delete)):
                pattern_to_delete = os.path.join(path_of_outliers, f"{base_name}-*")
                files_to_delete = glob.glob(pattern_to_delete)
                
                if not files_to_delete:
                    print(f"  - No se encontraron archivos para el outlier base '{base_name}'")
                
                for f_to_delete in files_to_delete:
                    try:
                        os.remove(f_to_delete)
                        print(f"  - ✅ Archivo de outlier borrado: {os.path.basename(f_to_delete)}")
                    except OSError as e:
                        print(f"  - 🔥 Error al borrar {os.path.basename(f_to_delete)}: {e}")

        # ============================================================
        # Lógica de Ploteo (sin cambios)
        # ============================================================
        metric_plot_idx = [header.index(m) for m in metrics_to_plot if m in header]
        results = np.array([[float(row[i]) for i in metric_plot_idx] for row in all_data])
        
        # --- Caso 1 parámetro ---
        if len(parameters_to_study) == 1:
            param_values = params[:, 0]
            unique_vals = np.unique(param_values)
            for j, metric in enumerate([m for m in metrics_to_plot if m in header]):
                means, mins, maxs = [], [], []
                for val in unique_vals:
                    metric_vals = results[param_values == val, j]
                    means.append(np.mean(metric_vals))
                    mins.append(np.min(metric_vals))
                    maxs.append(np.max(metric_vals))
                plt.figure(figsize=(8, 5))
                plt.fill_between(unique_vals, np.array(mins, dtype=float), np.array(maxs, dtype=float), color='lightgray', alpha=0.7, label='Range [min, max]')
                plt.plot(unique_vals, np.array(means, dtype=float), 'o-', color='blue', linewidth=2, markersize=6, label='Average')
                plt.xlabel(parameters_to_study[0], fontsize=14)
                plt.xscale('symlog', linthresh=1e-4)
                plt.xticks(fontsize=14)
                plt.ylabel(metric, fontsize=14)
                plt.yticks(fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.legend()
                plt.tight_layout()
                save_file = save_path + f"{metric}_vs_{parameters_to_study[0]}.png"
                plt.savefig(save_file, dpi=200)
                plt.close()

        # --- Caso 2 parámetros ---
        elif len(parameters_to_study) == 2:
            p1, p2 = parameters_to_study
            p1_vals, p2_vals = np.unique(params[:, 0]), np.unique(params[:, 1])
            for j, metric in enumerate([m for m in metrics_to_plot if m in header]):
                z = np.zeros((len(p2_vals), len(p1_vals)))
                for i1, v1 in enumerate(p1_vals):
                    for i2, v2 in enumerate(p2_vals):
                        mask = (params[:, 0] == v1) & (params[:, 1] == v2)
                        z[i2, i1] = np.mean(results[mask, j]) if np.any(mask) else np.nan
                plt.figure(figsize=(8, 6))
                pcm = plt.pcolormesh(p1_vals, p2_vals, z, cmap='viridis', shading='auto')
                cbar = plt.colorbar(pcm)
                cbar.set_label(metric, rotation=270, labelpad=15)
                plt.xlabel(p1, fontsize=14)
                plt.xticks(p1_vals, fontsize=14)
                plt.ylabel(p2, fontsize=14)
                plt.yticks(p2_vals, fontsize=14)
                min_x_val = np.min(np.abs(p1_vals[p1_vals != 0])) if np.any(p1_vals != 0) else 1e-4
                min_y_val = np.min(np.abs(p2_vals[p2_vals != 0])) if np.any(p2_vals != 0) else 1e-4
                plt.xscale('symlog', linthresh=min_x_val)
                plt.yscale('symlog', linthresh=min_y_val)
                plt.xlim(p1_vals.min(), p1_vals.max())
                plt.ylim(p2_vals.min(), p2_vals.max())
                plt.tight_layout()
                save_file = save_path + f"{metric}_heatmap_{p1}_{p2}.png"
                plt.savefig(save_file, dpi=200)
                plt.close()
            
    return all_analysis_results

def plot_alpha_omega_comparison(file_paths, save_paths):
    """
    Compara los resultados (media y rango min/max) de múltiples archivos 
    en un solo gráfico por métrica.
    
    Asume que cada archivo de entrada estudia UN solo parámetro.

    Parámetros
    ----------
    file_paths : list of str
        Rutas a los archivos .txt con resultados (ej. uno por loss function).
    save_paths : str 
        PREFIJO de la ruta donde guardar los gráficos. El nombre de la
        métrica se añadirá automáticamente (ej. "./plots/comparison_").
    """

    # Definición de métricas (basado en tu función anterior)
    metrics_full_list = [
        "global_loss", "global_mse", "global_mae", "global_masked_mse", "global_masked_mae",
        "global_vol_error", "global_far", "global_hit", "global_csi",
        "real_loss", "real_mse", "real_mae", "real_masked_mse", "real_masked_mae",
        "real_vol_error", "real_far", "real_hit", "real_csi"
    ]
    metrics_to_plot = [
        "global_mae", "global_masked_mae", "global_vol_error", "global_far", "global_hit", "global_csi",
        "real_mae", "real_masked_mae", "real_vol_error", "real_far", "real_hit", "real_csi"
    ]

    all_file_data = [] # Lista para guardar los datos procesados de cada archivo
    param_name_global = None # Para guardar el nombre del parámetro (ej. 'lr')

    # --- 1. Pre-procesamiento: Leer todos los archivos ---
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            print(f"Advertencia: No se encontró el archivo {file_path}, se omitirá.")
            continue
            
        with open(file_path, "r") as f: lines = f.readlines()
        if len(lines) < 2: continue

        header = lines[0].strip().split()
        all_data = np.array([line.strip().split() for line in lines[1:] if line.strip()], ndmin=2)
        parameters_to_study = [col for col in header if col not in metrics_full_list and col != "name"]

        if len(parameters_to_study) != 1:
            print(f"Advertencia: El archivo {file_path} no estudia 1 parámetro. Se omitirá.")
            continue
        
        current_param_name = parameters_to_study[0]

        if param_name_global is None:
            param_name_global = current_param_name 
        elif param_name_global != current_param_name:
            print(f"Advertencia: Parámetro '{current_param_name}' en {os.path.basename(file_path)} difiere del primero ('{param_name_global}').")
            print(f"           Se usará '{param_name_global}' como etiqueta del eje X.")

        param_idx = header.index(current_param_name) 
        params = np.array([float(row[param_idx]) for row in all_data])
        
        metric_plot_idx = [header.index(m) for m in metrics_to_plot if m in header]
        results = np.array([[float(row[i]) for i in metric_plot_idx] for row in all_data])
        
        param_values = params
        unique_vals = np.unique(param_values)
        metrics_data_for_file = {}

        for j, metric in enumerate([m for m in metrics_to_plot if m in header]):
            
            # <<< CAMBIO: Listas para min, max y media >>>
            means, mins, maxs = [], [], []
            
            for val in unique_vals:
                metric_vals = results[param_values == val, j]
                if metric_vals.size > 0:
                    means.append(np.mean(metric_vals))
                    mins.append(np.min(metric_vals))
                    maxs.append(np.max(metric_vals))
                else:
                    means.append(np.nan) 
                    mins.append(np.nan)
                    maxs.append(np.nan)
            
            # <<< CAMBIO: Guardar el diccionario de estadísticas >>>
            metrics_data_for_file[metric] = {
                "mean": np.array(means, dtype=float),
                "min": np.array(mins, dtype=float),
                "max": np.array(maxs, dtype=float)
            }
        
        legend_name = os.path.basename(file_path).split('.')[0]
        
        all_file_data.append({
            "legend_name": legend_name,
            "x_values": unique_vals,
            "metrics": metrics_data_for_file
        })

    if not all_file_data:
        print("No se procesaron datos válidos. No se generarán gráficos.")
        return

    # --- 2. Ploteo: Crear un gráfico por métrica ---
    print(f"\n📈 Generando gráficos de comparación con prefijo: {save_paths}")
    
    metrics_to_iterate = all_file_data[0]["metrics"].keys()

    for metric in metrics_to_iterate:
        plt.figure(figsize=(8, 5))
        
        # Dibujar la curva para cada archivo
        for file_data in all_file_data:
            if metric in file_data["metrics"]:
                
                # <<< CAMBIO: Extraer todas las estadísticas >>>
                x_vals = file_data["x_values"]
                stats = file_data["metrics"][metric]
                means = stats["mean"]
                mins = stats["min"]
                maxs = stats["max"]
                label = file_data["legend_name"]

                # Dibujar la línea de la media
                plot_line = plt.plot(
                    x_vals, 
                    means, 
                    'o-', 
                    linewidth=2, 
                    markersize=6, 
                    label=label
                )
                
                # <<< CAMBIO: Añadir la sombra (fill_between) >>>
                # Obtener el color que matplotlib acaba de usar
                line_color = plot_line[0].get_color()
                
                # Rellenar el área entre min y max con el mismo color
                # pero con baja opacidad (alpha)
                plt.fill_between(
                    x_vals,
                    mins,
                    maxs,
                    color=line_color,
                    alpha=0.2, # <-- Este es el "tono clarito"
                    label=None # No añadir a la leyenda
                )
        
        # Configuración del gráfico (sin cambios)
        plt.xlabel(param_name_global if param_name_global else "Parámetro", fontsize=14) 
        plt.xscale('symlog', linthresh=1e-4)
        plt.xticks(fontsize=14)
        plt.ylabel(metric, fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        
        # Guardar el gráfico
        save_file = save_paths + f"{metric}_comparison.png"
        plt.savefig(save_file, dpi=200)
        plt.close()
        print(f"  - Gráfico guardado: {save_file}")

    print("--- Proceso de ploteo completado. ---")

def print_analysis_list(title, string_list, param_name_guess="parámetro"):
    """
    Toma una lista de strings de análisis y la imprime de forma agrupada
    por valor del parámetro.
    """
    if not string_list:
        print(f"    {title}: Ninguno.")
        return
    
    print(f"    {title}:")
    grouped_items = defaultdict(list)
    param_name_found = param_name_guess
    
    # Regex para encontrar el parámetro y su valor (siempre al final)
    param_regex = re.compile(r"\(at (.+?)=([\d.e-]+)\)")
    # Regex para encontrar el nombre principal (todo antes del primer " (at")
    name_regex = re.compile(r"^(.*?)\s\(")
    
    for s in string_list:
        param_match = param_regex.search(s)
        name_match = name_regex.search(s) # Busca desde el inicio
        
        if param_match and name_match:
            param_name, value_str = param_match.groups()
            param_name_found = param_name
            param_value = float(value_str)
            
            # El 'display_name' es todo lo que hay antes del " (at"
            display_name = name_match.group(1).strip()
            grouped_items[param_value].append(display_name)
        else:
            grouped_items["_formato_desconocido_"].append(s)

    sorted_param_values = sorted([k for k in grouped_items.keys() if k != "_formato_desconocido_"])
    
    for param_value in sorted_param_values:
        names_list = grouped_items[param_value]
        print(f"      - Para {param_name_found} = {param_value}:")
        for name in names_list:
            print(f"        - {name}")
    
    if "_formato_desconocido_" in grouped_items:
        print(f"      - Con formato no reconocido:")
        for name in grouped_items["_formato_desconocido_"]:
            print(f"        - {name}")

def borrar_outputs(directories_outputs, bases_file_path):
    """
    Elimina archivos en una lista de directorios basándose en los prefijos 
    listados en un archivo de texto.

    Para cada línea en 'bases_file_path' (ej. 'Models_18_78_48-3841964.o'),
    extrae el prefijo hasta el guion inclusive (ej. 'Models_18_78_48-').
    Luego, busca y elimina todos los archivos que comiencen con ese prefijo
    en cada uno de los directorios de 'directories_outputs'.

    Parámetros
    ----------
    directories_outputs : list of str or str
        Una lista de rutas a directorios donde buscar, o una sola ruta (str).
    bases_file_path : str
        Ruta al archivo .txt que contiene las bases de los archivos a eliminar.
    """

    # --- 1. Hacer la función robusta (aceptar string o lista) ---
    if isinstance(directories_outputs, str):
        directories_outputs = [directories_outputs] # Convertir string a lista

    # --- 2. Leer el archivo de bases y extraer los prefijos ---
    base_names_to_delete = set()
    try:
        with open(bases_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # Ignorar líneas vacías
                
                # Encontrar el primer guion
                hyphen_index = line.find('-')
                
                if hyphen_index != -1:
                    # Extraer la base HASTA el guion (inclusive)
                    base_name = line[:hyphen_index + 1] 
                    base_names_to_delete.add(base_name)
                else:
                    print(f"Advertencia: Línea '{line}' en {bases_file_path} ignorada (no se encontró guion).")
    
    except FileNotFoundError:
        print(f"🔥 Error: No se pudo encontrar el archivo de bases: {bases_file_path}")
        return
    except Exception as e:
        print(f"🔥 Error al leer el archivo {bases_file_path}: {e}")
        return

    if not base_names_to_delete:
        print(f"ℹ️ El archivo '{bases_file_path}' estaba vacío o no contenía líneas válidas. No se borrará nada.")
        return

    print(f"ℹ️ Bases de prefijos a eliminar: {sorted(list(base_names_to_delete))}")

    # --- 3. Iterar por los directorios y eliminar archivos ---
    total_borrados = 0
    for directory in directories_outputs:
        if not os.path.isdir(directory):
            print(f"Error: El directorio '{directory}' no existe o no es un directorio. Omitiendo.")
            continue

        print(f"\n📂 Comprobando directorio: {directory}")
        
        for base_name in sorted(list(base_names_to_delete)):
            # Creamos el patrón de búsqueda, ej: /ruta/a/logs/Models_18_78_48-*
            pattern_to_delete = os.path.join(directory, f"{base_name}*")
            
            # Usamos glob para encontrar todos los archivos que coincidan
            files_to_delete = glob.glob(pattern_to_delete)
            
            if not files_to_delete:
                # Opcional: imprimir si no se encuentra nada
                # print(f"  - No se encontraron archivos para la base '{base_name}'")
                pass
            
            for f_to_delete in files_to_delete:
                try:
                    os.remove(f_to_delete)
                    print(f"  - ✅ Archivo borrado: {os.path.basename(f_to_delete)}")
                    total_borrados += 1
                except OSError as e:
                    print(f"  - 🔥 Error al borrar {os.path.basename(f_to_delete)}: {e}")

    print(f"\n--- Proceso completado. Total de archivos borrados: {total_borrados} ---")

def extract_bicubic_and_subgrid_metrics_from_logs(subgrid_files, bicubic_files, number_of_samples_max=None):
    """
    Lee una lista de archivos de log, extrae métricas específicas para Subgrid y Bicubic,
    filtra opcionalmente para quedarse con las N mejores (basado en csi_global),
    y devuelve los datos organizados en listas.

    Args:
        subgrid_files (list): Lista de rutas a los archivos de log para Subgrid.
        bicubic_files (list): Lista de rutas a los archivos de log para Bicubic.
        number_of_samples_max (int, optional):
            El número máximo de simulaciones (las mejores) a retener por grupo,
            basado en 'csi_global'. Si es None, se devuelven todas.

    Returns:
        dict: Un diccionario que contiene 16 listas con las métricas extraídas.
    """
    # Diccionario para almacenar todas las listas de métricas finales
    results = {
        "mae_global_bicubic": [], "masked_mae_global_bicubic": [], "vol_err_global_bicubic": [], "csi_global_bicubic": [],
        "mae_real_bicubic": [], "masked_mae_real_bicubic": [], "vol_err_real_bicubic": [], "csi_real_bicubic": [],
        "mae_global_subgrid": [], "masked_mae_global_subgrid": [], "vol_err_global_subgrid": [], "csi_global_subgrid": [],
        "mae_real_subgrid": [], "masked_mae_real_subgrid": [], "vol_err_real_subgrid": [], "csi_real_subgrid": []
    }

    def _parse_file(filepath):
        """Función auxiliar para procesar un único archivo de log."""
        metrics = {}
        try:
            with open(filepath, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"⚠️  Advertencia: El archivo no fue encontrado: {filepath}")
            return None

        # Expresiones regulares para encontrar los bloques de texto
        global_match = re.search(r"=== Global Test Results ===(.+?)===========================", content, re.DOTALL)
        real_match = re.search(r"=== Real Case Results ===(.+?)===========================", content, re.DOTALL)

        # Procesar bloque "Global Test Results"
        if global_match:
            block = global_match.group(1)
            mae_match = re.search(r"MAE:\s+([\d.]+)", block)
            masked_mae_match = re.search(r"Masked MAE:\s+([\d.]+)", block)
            vol_err_match = re.search(r"Volume Rel Error.*?\(%\):\s+([\d.]+)", block)
            csi_match = re.search(r"Threshold.*CSI=([\d.]+)%", block)
            
            if mae_match:
                metrics['mae_global'] = float(mae_match.group(1)) * 100
            if masked_mae_match:
                metrics['masked_mae_global'] = float(masked_mae_match.group(1)) * 100
            if vol_err_match:
                metrics['vol_err_global'] = float(vol_err_match.group(1))
            if csi_match:
                metrics['csi_global'] = float(csi_match.group(1))

        # Procesar bloque "Real Case Results"
        if real_match:
            block = real_match.group(1)
            mae_match = re.search(r"MAE:\s+([\d.]+)", block)
            masked_mae_match = re.search(r"Masked MAE:\s+([\d.]+)", block)
            vol_err_match = re.search(r"Volume Rel Error.*?\(%\):\s+([\d.]+)", block)
            csi_match = re.search(r"Threshold.*CSI=([\d.]+)%", block)

            if mae_match:
                metrics['mae_real'] = float(mae_match.group(1)) * 100
            if masked_mae_match:
                metrics['masked_mae_real'] = float(masked_mae_match.group(1)) * 100
            if vol_err_match:
                metrics['vol_err_real'] = float(vol_err_match.group(1))
            if csi_match:
                metrics['csi_real'] = float(csi_match.group(1))
        
        return metrics

    def _process_and_filter_list(file_list):
        """
        Helper para parsear una lista de archivos, filtrar por N mejores (si se pide)
        y devolver la lista de diccionarios de métricas (simulaciones).
        """
        # 1. Parsear todos los archivos y guardar sus métricas (como dicts)
        simulations_data = []
        for file in file_list:
            parsed_data = _parse_file(file)
            if parsed_data:
                simulations_data.append(parsed_data)
        
        # 2. Si no se pide filtrar, o no hay datos, devolver todo
        if number_of_samples_max is None or not simulations_data:
            return simulations_data

        # 3. Filtrado: Ordenar por csi_global (mayor es mejor)
        def get_csi_safe(sim):
            csi = sim.get('csi_global')
            return csi if csi is not None else -1.0 # Tratar Nones/faltantes como lo peor

        sorted_simulations = sorted(
            simulations_data,
            key=get_csi_safe,
            reverse=True
        )
        
        # 4. Devolver solo las N mejores
        print(f"Filtrando {len(simulations_data)} simulaciones a las {number_of_samples_max} mejores por csi_global.")
        return sorted_simulations[:number_of_samples_max]

    # --- Rellenar los datos de Bicubic ---
    # 1. Obtener la lista de simulaciones (ya filtrada si es necesario)
    processed_bicubic_data = _process_and_filter_list(bicubic_files)
    
    # 2. Poblar el diccionario de resultados final
    for parsed_data in processed_bicubic_data:
        results["mae_global_bicubic"].append(parsed_data.get('mae_global'))
        results["masked_mae_global_bicubic"].append(parsed_data.get('masked_mae_global'))
        results["vol_err_global_bicubic"].append(parsed_data.get('vol_err_global'))
        results["csi_global_bicubic"].append(parsed_data.get('csi_global'))
        results["mae_real_bicubic"].append(parsed_data.get('mae_real'))
        results["masked_mae_real_bicubic"].append(parsed_data.get('masked_mae_real'))
        results["vol_err_real_bicubic"].append(parsed_data.get('vol_err_real'))
        results["csi_real_bicubic"].append(parsed_data.get('csi_real'))

    # --- Rellenar los datos de Subgrid ---
    # 1. Obtener la lista de simulaciones (ya filtrada si es necesario)
    processed_subgrid_data = _process_and_filter_list(subgrid_files)
    
    # 2. Poblar el diccionario de resultados final
    for parsed_data in processed_subgrid_data:
        results["mae_global_subgrid"].append(parsed_data.get('mae_global'))
        results["masked_mae_global_subgrid"].append(parsed_data.get('masked_mae_global'))
        results["vol_err_global_subgrid"].append(parsed_data.get('vol_err_global'))
        results["csi_global_subgrid"].append(parsed_data.get('csi_global'))
        results["mae_real_subgrid"].append(parsed_data.get('mae_real'))
        results["masked_mae_real_subgrid"].append(parsed_data.get('masked_mae_real'))
        results["vol_err_real_subgrid"].append(parsed_data.get('vol_err_real'))
        results["csi_real_subgrid"].append(parsed_data.get('csi_real'))

    return results

def plot_comparison_bicubic_vs_subgrid(global_bicubic, global_subgrid, real_bicubic, real_subgrid, save_path='.'):
    """
    Crea y guarda imágenes de comparación entre Bicubic (rojo) y Subgrid (azul).

    Para cada métrica, genera dos gráficos (Global y Real) con puntos de datos,
    y una línea discontinua para la media de cada método.

    Args:
        global_bicubic (dict): Métricas del Test Set para Bicubic.
        global_subgrid (dict): Métricas del Test Set para Subgrid.
        real_bicubic (dict): Métricas del Real Case para Bicubic.
        real_subgrid (dict): Métricas del Real Case para Subgrid.
        save_path (str, optional): Directorio donde se guardarán las imágenes.
                                   Por defecto, es el directorio actual.
    """
    # 1. Asegurarse de que el directorio de guardado exista
    os.makedirs(save_path, exist_ok=True)
    
    metrics = list(global_bicubic.keys())
    
    # 2. Iterar sobre cada métrica para crear sus dos gráficos
    for metric in metrics:
        
        # --- Gráfico 1: Global (Test Set) ---
        fig_global, ax_global = plt.subplots(figsize=(6, 5))
        
        gb_data = global_bicubic.get(metric, [])
        gs_data = global_subgrid.get(metric, [])

        x_bicubic_g = np.random.normal(0, 0.04, size=len(gb_data))
        x_subgrid_g = np.random.normal(1, 0.04, size=len(gs_data))
        
        # Puntos
        ax_global.scatter(x_bicubic_g, gb_data, alpha=0.7, color='coral', edgecolor='k', linewidth=0.5, s=80, label='Bicubic')
        ax_global.scatter(x_subgrid_g, gs_data, alpha=0.7, color='dodgerblue', edgecolor='k', linewidth=0.5, s=80, label='Subgrid')

        # Líneas de media (ignorando NaNs en el cálculo)
        if len(gb_data) > 0:
            mean_gb = np.nanmean(gb_data)
            ax_global.axhline(y=mean_gb, color='coral', linestyle='--', linewidth=2, label='Mean Bicubic')
        if len(gs_data) > 0:
            mean_gs = np.nanmean(gs_data)
            ax_global.axhline(y=mean_gs, color='dodgerblue', linestyle='--', linewidth=2, label='Mean Subgrid')

        # Estilo
        ax_global.set_ylabel(metric, fontsize=14, weight='bold')
        ax_global.set_xticks([0, 1])
        ax_global.set_xticklabels(['Bicubic', 'Subgrid'], fontsize=12)
        ax_global.grid(True, linestyle='--', alpha=0.6, axis='y')
        ax_global.legend(fontsize=10)

        # Guardar
        clean_metric_name = metric.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'perc')
        filename_global = f"comparison_BI_SG_global_{clean_metric_name}.png"
        full_path_global = os.path.join(save_path, filename_global)
        plt.tight_layout()
        plt.savefig(full_path_global, dpi=150)
        plt.close(fig_global)

        # --- Gráfico 2: Real Case ---
        fig_real, ax_real = plt.subplots(figsize=(6, 5))

        rb_data = real_bicubic.get(metric, [])
        rs_data = real_subgrid.get(metric, [])

        x_bicubic_r = np.random.normal(0, 0.04, size=len(rb_data))
        x_subgrid_r = np.random.normal(1, 0.04, size=len(rs_data))

        # Puntos
        ax_real.scatter(x_bicubic_r, rb_data, alpha=0.7, color='coral', edgecolor='k', linewidth=0.5, s=80, label='Bicubic')
        ax_real.scatter(x_subgrid_r, rs_data, alpha=0.7, color='dodgerblue', edgecolor='k', linewidth=0.5, s=80, label='Subgrid')
        
        # Líneas de media
        if len(rb_data) > 0:
            mean_rb = np.nanmean(rb_data)
            ax_real.axhline(y=mean_rb, color='coral', linestyle='--', linewidth=2, label='Mean Bicubic')
        if len(rs_data) > 0:
            mean_rs = np.nanmean(rs_data)
            ax_real.axhline(y=mean_rs, color='dodgerblue', linestyle='--', linewidth=2, label='Mean Subgrid')
        
        # Estilo
        ax_real.set_ylabel(metric, fontsize=14, weight='bold')
        ax_real.set_xticks([0, 1])
        ax_real.set_xticklabels(['Bicubic', 'Subgrid'], fontsize=12)
        ax_real.grid(True, linestyle='--', alpha=0.6, axis='y')
        ax_real.legend(fontsize=10)
        
        # Guardar
        filename_real = f"comparison_BI_SG_real_{clean_metric_name}.png"
        full_path_real = os.path.join(save_path, filename_real)
        plt.tight_layout()
        plt.savefig(full_path_real, dpi=150)
        plt.close(fig_real)

def _parse_file(filepath):
    """Función auxiliar para procesar un único archivo de log del estudio topográfico."""
    metrics = {}
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"⚠️  Advertencia: El archivo no fue encontrado: {filepath}")
        return None

    # 1. Extraer Tiempo de Entrenamiento
    time_match = re.search(r"Total training time:.*?\((\d+\.\d+)\s+minutes\)", content)
    if time_match:
        metrics['time'] = round(float(time_match.group(1)), 2)

    # 2. Expresiones regulares para los bloques de resultados
    global_match = re.search(r"=== Global Test Results ===(.+?)===========================", content, re.DOTALL)
    real_match = re.search(r"=== Real Case Results ===(.+?)===========================", content, re.DOTALL)

    # 3. Procesar bloque "Global Test Results"
    if global_match:
        block = global_match.group(1)
        mae_match = re.search(r"MAE:\s+([\d.]+)", block)
        masked_mae_match = re.search(r"Masked MAE:\s+([\d.]+)", block)
        vol_err_match = re.search(r"Volume Rel Error.*?\(%\):\s+([\d.]+)", block)
        csi_match = re.search(r"Threshold.*CSI=([\d.]+)%", block)
        
        # Multiplicamos por 100 para que sean 'cm' (como en tu función original)
        if mae_match:
            metrics['mae_global'] = float(mae_match.group(1)) * 100
        if masked_mae_match:
            metrics['masked_mae_global'] = float(masked_mae_match.group(1)) * 100
        if vol_err_match:
            metrics['vol_err_global'] = float(vol_err_match.group(1))
        if csi_match:
            metrics['csi_global'] = float(csi_match.group(1))

    # 4. Procesar bloque "Real Case Results"
    if real_match:
        block = real_match.group(1)
        mae_match = re.search(r"MAE:\s+([\d.]+)", block)
        masked_mae_match = re.search(r"Masked MAE:\s+([\d.]+)", block)
        vol_err_match = re.search(r"Volume Rel Error.*?\(%\):\s+([\d.]+)", block)
        csi_match = re.search(r"Threshold.*CSI=([\d.]+)%", block)

        if mae_match:
            metrics['mae_real'] = float(mae_match.group(1)) * 100
        if masked_mae_match:
            metrics['masked_mae_real'] = float(masked_mae_match.group(1)) * 100
        if vol_err_match:
            metrics['vol_err_real'] = float(vol_err_match.group(1))
        if csi_match:
            metrics['csi_real'] = float(csi_match.group(1))
    
    return metrics

def extract_study_metrics(log_files_dict, save_path, number_of_samples_max=50, calculate_average_of_averages=False):
    """
    Lee logs, filtra los mejores por CSI, extrae métricas y guarda resúmenes de medias.

    Args:
        log_files_dict (dict): Grupos -> Lista de archivos.
        save_path (str): Directorio de salida.
        number_of_samples_max (int): Top N simulaciones a guardar.
        calculate_average_of_averages (bool): Si True, guarda un archivo extra con 
                                              la media de las medias de todos los grupos.
    """
    
    metric_keys = [
        "time", "mae_global", "masked_mae_global", "vol_err_global", "csi_global",
        "mae_real", "masked_mae_real", "vol_err_real", "csi_real"
    ]
    
    # Diccionario de salida: metric -> group -> list of values
    final_results = {key: defaultdict(list) for key in metric_keys}

    # --- 1. PROCESAMIENTO Y FILTRADO DE LOGS ---
    for group_name, file_list in log_files_dict.items():
        print(f"Procesando grupo: {group_name} ({len(file_list)} archivos)")
        
        group_simulations = [] 
        for file in file_list:
            parsed_data = _parse_file(file)
            if parsed_data:
                group_simulations.append({'filepath': file, 'metrics': parsed_data})

        if not group_simulations:
            print(f"  -> No se encontraron datos válidos para {group_name}.")
            continue

        # Función helper para ordenar
        def get_csi_safe(sim_wrapper):
            csi = sim_wrapper['metrics'].get('csi_real')
            return csi if csi is not None else -1.0

        sorted_simulations = sorted(group_simulations, key=get_csi_safe, reverse=True)
        top_n_simulations = sorted_simulations[:number_of_samples_max]

        # Guardar lista de mejores archivos .txt
        os.makedirs(save_path, exist_ok=True)
        clean_group_name = group_name.replace(' ', '_').replace('=', '')
        output_txt_path = os.path.join(save_path, f"top_{number_of_samples_max}_files_{clean_group_name}.txt")
        try:
            with open(output_txt_path, 'w') as f:
                f.write(f"# Top {len(top_n_simulations)} archivos para: {group_name}\n")
                for sim_wrapper in top_n_simulations:
                    filename = os.path.basename(sim_wrapper['filepath'])
                    f.write(f"{filename}\n")
        except Exception as e:
            print(f"-> ⚠️ Error guardando lista archivos: {e}")
        
        # Poblar final_results solo con los TOP N
        for sim_wrapper in top_n_simulations:
            sim_dict = sim_wrapper['metrics'] 
            for key in metric_keys:
                val = sim_dict.get(key)
                final_results[key][group_name].append(val)


    # --- 2. CÁLCULO DE MEDIAS POR GRUPO Y GUARDADO ---
    print("\n📊 Calculando medias y guardando resúmenes...")
    
    # Estructura auxiliar para guardar las medias calculadas: {metric: {group: mean_val}}
    calculated_means = {key: {} for key in metric_keys}
    
    avg_file_path = os.path.join(save_path, "metrics_averages_per_group.txt")
    
    try:
        with open(avg_file_path, 'w') as f:
            f.write(f"=== RESUMEN DE MEDIAS (Top {number_of_samples_max} samples) ===\n")
            f.write("Valores promedios para cada métrica y grupo.\n\n")
            
            for metric in metric_keys:
                f.write(f"--- {metric.upper()} ---\n")
                
                # Iteramos sobre los grupos que tienen datos para esta métrica
                groups_for_metric = final_results[metric]
                
                for group_name, values in groups_for_metric.items():
                    # Filtrar Nones antes de hacer la media
                    clean_values = [v for v in values if v is not None]
                    
                    if clean_values:
                        mean_val = np.mean(clean_values)
                        calculated_means[metric][group_name] = mean_val
                        f.write(f"  {group_name}: {mean_val:.6f}\n")
                    else:
                        f.write(f"  {group_name}: N/A (Sin datos válidos)\n")
                f.write("\n")
        print(f"-> Medias por grupo guardadas en: {avg_file_path}")
        
    except Exception as e:
        print(f"-> ⚠️ Error guardando medias por grupo: {e}")


    # --- 3. CÁLCULO DE 'MEDIA DE MEDIAS' (OPCIONAL) ---
    if calculate_average_of_averages:
        avg_avg_file_path = os.path.join(save_path, "metrics_global_average_of_averages.txt")
        try:
            with open(avg_avg_file_path, 'w') as f:
                f.write("=== MEDIA DE MEDIAS (PROMEDIO GLOBAL) ===\n")
                f.write("Promedio aritmético de las medias de cada grupo.\n")
                f.write(f"Grupos considerados: {list(log_files_dict.keys())}\n\n")
                
                f.write(f"{'MÉTRICA':<25} | {'MEDIA DE MEDIAS':<15}\n")
                f.write("-" * 45 + "\n")
                
                for metric in metric_keys:
                    # Recuperamos las medias grupales que calculamos en el paso anterior
                    group_means_list = list(calculated_means[metric].values())
                    
                    if group_means_list:
                        avg_of_avgs = np.mean(group_means_list)
                        f.write(f"{metric:<25} | {avg_of_avgs:.6f}\n")
                    else:
                        f.write(f"{metric:<25} | N/A\n")
            
            print(f"-> Media de medias guardada en: {avg_avg_file_path}")

        except Exception as e:
            print(f"-> ⚠️ Error guardando media de medias: {e}")


    # Convertir los defaultdict a dict normales para el retorno original
    final_results_dict = {key: dict(value) for key, value in final_results.items()}
    return final_results_dict

def plot_study_comparison(results_dict, save_path='.', metric_name_map=None, 
                          name_of_images='comparison', max_groups_per_plot=10,
                          global_real_comparison=False):
    """
    Crea y guarda imágenes de comparación para un estudio con N categorías.
    CORREGIDA: Ahora dibuja todas las métricas individuales si global_real_comparison es False.
    """
    
    # 1. Asegurarse de que el directorio de guardado exista
    os.makedirs(save_path, exist_ok=True)
    if metric_name_map is None:
        metric_name_map = {}

    # Definir las métricas base que tienen versión global y real
    COMPARISON_METRICS = ["mae", "masked_mae", "vol_err", "csi"]

    all_metric_keys = list(results_dict.keys())
    
    # -------------------------------------------------------------
    # Bucle 1: Gráficos individuales
    # -------------------------------------------------------------
    
    # [!!!] CORRECCIÓN AQUÍ [!!!]
    if global_real_comparison:
        # Si vamos a comparar, EXCLUIMOS del plot individual las métricas que tienen pareja
        # para no tener gráficos repetidos (uno individual y otro comparativo).
        metrics_to_plot_individual = [
            k for k in all_metric_keys 
            if k not in (m + '_global' for m in COMPARISON_METRICS) 
            and k not in (m + '_real' for m in COMPARISON_METRICS)
        ]
    else:
        # Si NO vamos a comparar, las pintamos TODAS individualmente
        metrics_to_plot_individual = all_metric_keys

    for metric_key in metrics_to_plot_individual:
        group_data_dict = results_dict.get(metric_key, {})
        all_group_names = list(group_data_dict.keys())
        if not all_group_names:
            # print(f"Saltando métrica (sin datos): {metric_key}") # Opcional para no ensuciar
            continue
        
        # --- Lógica de Paginación ---
        for i_chunk, start_index in enumerate(range(0, len(all_group_names), max_groups_per_plot)):
            group_names_chunk = all_group_names[start_index : start_index + max_groups_per_plot]
            if not group_names_chunk: continue

            # Configuración de FIGURA
            width = max(6, 1.5 * len(group_names_chunk)) 
            fig, ax = plt.subplots(figsize=(width, 5))

            # Iterar sobre cada categoría
            for i_local, group_name in enumerate(group_names_chunk):
                data = [d for d in group_data_dict[group_name] if d is not None]
                if not data: continue

                color = plt.cm.tab10(i_local % 10) 

                # Puntos dispersos
                x_jitter = np.random.normal(i_local, 0.04, size=len(data)) 
                ax.scatter(x_jitter, data, alpha=0.7, color=color, edgecolor='k', 
                           linewidth=0.5, s=80, marker='o', zorder=3)

                # Línea de media
                mean_val = np.nanmean(data)
                ax.plot([i_local - 0.4, i_local + 0.4], [mean_val, mean_val], 
                        color=color, linestyle='--', linewidth=2.5, zorder=2)

            # Estilo y Guardar
            display_name = metric_name_map.get(metric_key, metric_key.replace("_", " ").title())
            ax.set_ylabel(display_name, fontsize=14)
            ax.set_xticks(range(len(group_names_chunk))) 
            ax.set_xticklabels(group_names_chunk, fontsize=12, rotation=10)
            ax.grid(True, linestyle='--', alpha=0.6, axis='y')
            
            # Si es paginado, añadimos sufijo
            suffix = f"_part_{i_chunk + 1}" if len(all_group_names) > max_groups_per_plot else ""
            filename = f"{name_of_images}_{metric_key}{suffix}.png"
            
            full_path = os.path.join(save_path, filename)
            plt.tight_layout()
            plt.savefig(full_path, dpi=150)
            plt.close(fig)


    # -------------------------------------------------------------
    # Bucle 2: Gráficos de comparación Global (O) vs Real (X)
    # -------------------------------------------------------------
    if global_real_comparison:
        
        for base_metric in COMPARISON_METRICS:
            key_global = f"{base_metric}_global"
            key_real   = f"{base_metric}_real"

            if key_global not in all_metric_keys or key_real not in all_metric_keys:
                continue

            group_data_global = results_dict[key_global]
            group_data_real   = results_dict[key_real]
            all_group_names   = list(group_data_global.keys())
            
            if not all_group_names: continue

            # --- Lógica de Paginación ---
            for i_chunk, start_index in enumerate(range(0, len(all_group_names), max_groups_per_plot)):
                
                group_names_chunk = all_group_names[start_index : start_index + max_groups_per_plot]
                if not group_names_chunk: continue

                # Configuración de FIGURA
                display_name = metric_name_map.get(base_metric, base_metric.replace("_", " ").title())
                title_suffix = " (Global vs. Real)" 
                width = max(6, 1.5 * len(group_names_chunk)) 
                fig, ax = plt.subplots(figsize=(width, 5))
                
                handle_g = Line2D([0], [0], color='gray', marker='o', linestyle='None', markersize=8, label='Media Global')
                handle_r = Line2D([0], [0], color='gray', marker='x', linestyle='None', markersize=8, label='Media Real')
                legend_handles = [handle_g, handle_r]
                
                for i_local, group_name in enumerate(group_names_chunk):
                    
                    data_global = [d for d in group_data_global[group_name] if d is not None]
                    data_real   = [d for d in group_data_real[group_name] if d is not None]
                    
                    if not data_global and not data_real:
                        continue

                    color = plt.cm.tab10(i_local % 10) 
                    
                    # --- Dibujar GLOBAL (Círculos 'o') ---
                    if data_global:
                        ax.scatter(np.random.normal(i_local, 0.04, size=len(data_global)), data_global, 
                                   alpha=0.8, color=color, edgecolor='k', 
                                   linewidth=0.8, s=80, marker='o', zorder=3)
                        mean_val_g = np.nanmean(data_global)
                        ax.plot([i_local - 0.4, i_local + 0.4], [mean_val_g, mean_val_g], 
                                color=color, linestyle='-', linewidth=2.5, zorder=2) 

                    # --- Dibujar REAL (Cruces 'x') ---
                    if data_real:
                        ax.scatter(np.random.normal(i_local, 0.04, size=len(data_real)), data_real, 
                                   alpha=0.7, color=color, 
                                   linewidth=1.5, s=100, marker='x', zorder=3)
                        mean_val_r = np.nanmean(data_real)
                        ax.plot([i_local - 0.4, i_local + 0.4], [mean_val_r, mean_val_r], 
                                color=color, linestyle='--', linewidth=2.5, zorder=2)

                # Estilo
                ax.set_ylabel(display_name + title_suffix, fontsize=14, weight='bold')
                ax.set_xticks(range(len(group_names_chunk))) 
                ax.set_xticklabels(group_names_chunk, fontsize=12, rotation=10)
                ax.grid(True, linestyle='--', alpha=0.6, axis='y')
                ax.legend(handles=legend_handles, fontsize=10, loc='upper right')

                suffix = f"_part_{i_chunk + 1}" if len(all_group_names) > max_groups_per_plot else ""
                filename = f"{name_of_images}_{base_metric}_Global_Real{suffix}.png"
                full_path = os.path.join(save_path, filename)
                plt.tight_layout()
                plt.savefig(full_path, dpi=150)
                plt.close(fig)

    print(f"\n✅ Gráficos de comparación guardados en: {os.path.abspath(save_path)}")

def plot_model_metrics(model_metrics: list[str], labels: list[str], save_path: str, plot_averages: bool):
    """
    Genera dos gráficos (MAE y CSI vs. Tiempo) a partir de múltiples archivos de métricas.

    Args:
        model_metrics (List[str]): Lista de rutas a los archivos de métricas (.txt).
        labels (List[str]): Lista de etiquetas para la leyenda (debe coincidir con model_metrics).
        save_path (str): Directorio donde se guardarán los gráficos (se creará si no existe).
    """

    if len(model_metrics) != len(labels):
        raise ValueError("Las listas 'model_metrics' y 'labels' deben tener la misma longitud.")

    # 1. Crear el directorio de guardado si no existe
    try:
        os.makedirs(save_path, exist_ok=True)
    except Exception as e:
        print(f"Error al crear el directorio de guardado: {e}")
        return

    # 2. Inicializar figuras para MAE y CSI
    fig_mae, ax_mae = plt.subplots(figsize=(10, 6))
    fig_csi, ax_csi = plt.subplots(figsize=(10, 6))

    # Colores base para las líneas (se repiten si hay más de 10 modelos)
    required_colors = ['#1f77b4', '#d62728', '#2ca02c', 'orange', 'cyan', 'purple', 'brown', 'pink', 'gray', 'cyan']

    # 3. Procesar cada archivo de métricas
    for idx, (file_path, label) in enumerate(zip(model_metrics, labels)):
        
        try:
            # Leer el archivo, saltando líneas que comienzan con '#'
            # Usamos `comments='#'` para ignorar las cabeceras
            # Usamos `usecols` para seleccionar las columnas (0:Time, 1:MAE, 2:CSI)
            data = np.loadtxt(file_path, comments='#', usecols=(0, 1, 2))
            
            # 4. Extracción y Transformación de Datos
            
            # Columna 0: Time step (T) -> Tiempo en horas (T / 2)
            time_steps = data[:, 0]
            time_h = time_steps / 2.0
            
            # Columna 1: MAE -> MAE en cm (MAE * 100)
            mae_cm = data[:, 1] * 100.0
            
            # Columna 2: CSI -> CSI en porcentaje (CSI * 100)
            csi_percent = data[:, 2] * 100.0

            # 5. Dibujar en los gráficos y los valores medios de cada métrica
            current_color = required_colors[idx % len(required_colors)]
            # Gráfico de MAE y su media
            ax_mae.plot(time_h, mae_cm, label=label, color=current_color, linewidth=2)     
            if plot_averages:
                ax_mae.axhline(y=np.nanmean(mae_cm), color='red', linestyle='--', linewidth=2, alpha=0.7, label=f"Mean value: {np.nanmean(mae_cm):.2f} cm")
            # Gráfico de CSI
            ax_csi.plot(time_h, csi_percent, label=label, color=current_color, linewidth=2)
            if plot_averages:
                ax_csi.axhline(y=np.nanmean(csi_percent), color='red', linestyle='--', linewidth=2, alpha=0.7, label=f"Mean value: {np.nanmean(csi_percent):.2f} %")
                ax_csi.axhline(y=np.nanmean(csi_percent[15:]), color='red', linestyle=':', linewidth=2, alpha=0.7, label=f"Mean from hour 8: {np.nanmean(csi_percent[15:]):.2f} %")

            print(f"Procesado correctamente: {label} ({os.path.basename(file_path)})")

        except FileNotFoundError:
            print(f"[ERROR] Archivo no encontrado: {file_path}. Saltando.")
        except IndexError:
            print(f"[ERROR] Datos insuficientes o formato incorrecto en: {file_path}. Se esperaban al menos 3 columnas.")
        except Exception as e:
            print(f"[ERROR] Error inesperado al procesar {file_path}: {e}")


    # 6. Configuración y Guardado del Gráfico MAE
    # ax_mae.set_title('Error Absoluto Medio (MAE) vs. Time')
    ax_mae.set_xlabel('Time (h)', fontsize=14)
    ax_mae.set_ylabel('MAE (cm)', fontsize=14)
    ax_mae.tick_params(axis='y', labelsize=14)
    ax_mae.tick_params(axis='x', labelsize=14)
    ax_mae.legend(loc='best', frameon=True, shadow=True, fancybox=True, fontsize=14)
    ax_mae.grid(True, linestyle='--', alpha=0.7)
    
    # Guardar la figura MAE
    mae_output_file = os.path.join(save_path, 'MAE_vs_Time.png')
    fig_mae.savefig(mae_output_file, bbox_inches='tight')
    print(f"Gráfico MAE guardado en: {mae_output_file}")
    plt.close(fig_mae) # Cierra la figura para liberar memoria

    # 7. Configuración y Guardado del Gráfico CSI
    # ax_csi.set_title('Critical Success Index (CSI) vs. Time')
    ax_csi.set_xlabel('Time (h)', fontsize=14)
    ax_csi.set_ylabel('CSI (%)', fontsize=14)
    # ytixks cada 10 puntos y tamaño 14 de letra
    ax_csi.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax_csi.tick_params(axis='y', labelsize=14)
    ax_csi.tick_params(axis='x', labelsize=14)
    ax_csi.legend(loc='best', frameon=True, shadow=True, fancybox=True, fontsize=14)
    ax_csi.grid(True, linestyle='--', alpha=0.7)
    # Establecer límites del eje Y para CSI (0% a 100%)
    # ax_csi.set_ylim(0, 100)

    # Guardar la figura CSI
    csi_output_file = os.path.join(save_path, 'CSI_vs_Time.png')
    fig_csi.savefig(csi_output_file, bbox_inches='tight')
    print(f"Gráfico CSI guardado en: {csi_output_file}")
    plt.close(fig_csi) # Cierra la figura para liberar memoria


def central_crop_to_match(arr, target_h, target_w, target_c):
    """Central-crop arr (T,H,W,C) to (T,target_h,target_w,target_c)."""
    _, h, w, c = arr.shape
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    return arr[:, start_h:start_h + target_h, start_w:start_w + target_w, :target_c]

def denorm_inverse_clip(x, *, y_min, y_max, log1p_exp1m_transform, clip_min=None, clip_max=None, clip=True):
    """Denormalize [0,1] -> [y_min,y_max], optional expm1, optional clip."""
    x = x * (y_max - y_min) + y_min
    if log1p_exp1m_transform:
        x = np.expm1(x)
    if clip:
        if clip_min is not None:
            x = np.clip(x, a_min=clip_min, a_max=None)
        if clip_max is not None:
            x = np.clip(x, a_min=None, a_max=clip_max)
    return x

def generate_real_case_extra_visualizations(
    *,
    model,
    X_real_case,
    y_real_case,
    device,
    cfg,
    norm_params,
    save_base_path,
    batch_size=16,
    conf_matrix_limits=(0.0, 0.1, 0.5, 1.0),
    wet_limits=(0.1, 0.5, 1.0),
    use_interpolation="subgrid",  # "subgrid" or "bicubic" or None
    specific_indices_print_progress=True,
    evaluation_mask=None,
):
    """
    Generates:
    - scatter plot sim vs pred
    - error histogram sim - pred
    - confusion matrix sim vs pred
    - wet area vs time (sim/pred/interp)
    """

    model.eval()

    # Prepare loader
    dataset = TensorDataset(X_real_case[0], X_real_case[1])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Resize layers to match y_real_case H,W
    target_h, target_w = y_real_case.shape[1], y_real_case.shape[2]
    resize_layer_bicubic = ResizeWithBicubic(target_h, target_w).to(device)
    resize_layer_subgrid = ResizeWithClassicalSubGrid(target_h, target_w).to(device)

    preds_list, bicubic_list, subgrid_list = [], [], []

    with torch.no_grad():
        for batch_idx, (xb0, xb1) in enumerate(loader):
            xb0, xb1 = xb0.to(device), xb1.to(device)

            pred = model(xb0, xb1)
            bicubic = resize_layer_bicubic(xb0[:, :, :, :1])
            subgrid = resize_layer_subgrid(xb0, xb1[:, :, :, :1], group_size_rows=cfg.GROUPING_PARAMETER_FOR_SUBGRID_ROWS, group_size_cols=cfg.GROUPING_PARAMETER_FOR_SUBGRID_COLS)


            preds_list.append(pred.detach().cpu().numpy())
            bicubic_list.append(bicubic.detach().cpu().numpy())
            subgrid_list.append(subgrid.detach().cpu().numpy())

            if specific_indices_print_progress:
                print(f"Processed batch {batch_idx + 1}/{len(loader)}")

    predictions_np = np.concatenate(preds_list, axis=0)
    bicubic_np = np.concatenate(bicubic_list, axis=0)
    subgrid_np = np.concatenate(subgrid_list, axis=0)
    y_np = y_real_case.detach().cpu().numpy()

    # Denormalize / inverse / clip (we only clip predictions)
    y_min, y_max = float(norm_params[4]), float(norm_params[5])
    predictions_np = denorm_inverse_clip(predictions_np, y_min=y_min, y_max=y_max, log1p_exp1m_transform=cfg.LOG1P_EXP1M_TRANSFORM, clip_min=cfg.CLIP_VALUE_MIN, clip_max=cfg.CLIP_VALUE_MAX, clip=True)
    y_np = denorm_inverse_clip(y_np, y_min=y_min, y_max=y_max, log1p_exp1m_transform=cfg.LOG1P_EXP1M_TRANSFORM, clip=False)
    bicubic_np = denorm_inverse_clip(bicubic_np, y_min=y_min, y_max=y_max, log1p_exp1m_transform=False, clip=False)
    subgrid_np = denorm_inverse_clip(subgrid_np, y_min=y_min, y_max=y_max, log1p_exp1m_transform=False, clip=False)

    # Crop everything to match predictions shape
    t, h_pred, w_pred, c_pred = predictions_np.shape
    y_crop = central_crop_to_match(y_np, h_pred, w_pred, c_pred)
    bicubic_crop = central_crop_to_match(bicubic_np, h_pred, w_pred, c_pred)
    subgrid_crop = central_crop_to_match(subgrid_np, h_pred, w_pred, c_pred)

    print(f"predictions shape: {predictions_np.shape}")
    print(f"y_crop shape:      {y_crop.shape}")
    print(f"bicubic_crop shape:{bicubic_crop.shape}")
    print(f"subgrid_crop shape:{subgrid_crop.shape}")

    if evaluation_mask is not None:
        evaluation_mask = np.asarray(evaluation_mask, dtype=bool)
        if evaluation_mask.shape != y_np.shape[1:3]:
            raise ValueError(
                f"Tamaño incompatible de máscara en visualizaciones adicionales: "
                f"máscara {evaluation_mask.shape}, referencia completa {y_np.shape[1:3]}"
            )
        if cfg.PATCH_BASED_MAPPING:
            raise ValueError(
                "PATCH_BASED_MAPPING=True no admite visualizaciones adicionales "
                "con máscara geográfica completa."
            )
        crop_h = (y_np.shape[1] - y_crop.shape[1]) // 2
        crop_w = (y_np.shape[2] - y_crop.shape[2]) // 2
        evaluation_mask = evaluation_mask[
            crop_h:crop_h + y_crop.shape[1],
            crop_w:crop_w + y_crop.shape[2],
        ]
        if evaluation_mask.shape != y_crop.shape[1:3]:
            raise ValueError(
                f"Tamaño incompatible tras recortar la máscara adicional: "
                f"{evaluation_mask.shape}, datos {y_crop.shape[1:3]}"
            )

    # 1) Scatter + histogram
    if evaluation_mask is not None:
        domain3d = np.broadcast_to(
            evaluation_mask[None, :, :],
            y_crop.shape[:3],
        )
        sim_points = y_crop[..., 0][domain3d]
        pred_points = predictions_np[..., 0][domain3d]
    else:
        sim_points = y_crop.flatten()
        pred_points = predictions_np.flatten()

    scatter_plot_sim_vs_pred(
        sim_points=sim_points,
        pred_points=pred_points,
        variable="Depth",
        units="m",
        savepath=save_base_path + "_scatter.png"
    )
    histogram_errors_plot_sim_vs_pred(
        sim_points=sim_points,
        pred_points=pred_points,
        variable="Depth",
        units="m",
        savepath=save_base_path + "_histogram_errors.png"
    )

    # 2) Confusion matrix
    confusion_matrix_sim_vs_pred(
        sim_points=sim_points,
        pred_points=pred_points,
        variable="Depth",
        units="m",
        limits=list(conf_matrix_limits),
        savepath=save_base_path + "_confusion.png"
    )

    # 3) Wet area vs time
    n_timesteps = y_crop.shape[0]
    print(f"\nCalculating wet area for {n_timesteps} timesteps...")

    wet_areas_sim, wet_areas_pred, wet_areas_int = [], [], []

    if use_interpolation.lower() == "bicubic":
        interp_stack = bicubic_crop
    elif use_interpolation.lower() == "subgrid":
        interp_stack = subgrid_crop
    else:
        interp_stack = np.zeros_like(predictions_np) # fallback to zeros if we do not want to use interpolation

    for ti in range(n_timesteps):
        img_sim = y_crop[ti]
        img_pred = predictions_np[ti]
        img_int = interp_stack[ti]

        # ensure 2D
        if img_sim.ndim > 2: img_sim = img_sim[..., 0]
        if img_pred.ndim > 2: img_pred = img_pred[..., 0]
        if img_int.ndim > 2: img_int = img_int[..., 0]

        if evaluation_mask is not None:
            img_sim = np.where(evaluation_mask, img_sim, 0.0)
            img_pred = np.where(evaluation_mask, img_pred, 0.0)
            img_int = np.where(evaluation_mask, img_int, 0.0)

        wet_areas_sim.append(calculate_wet_area(img_sim, cfg.CELLSIZE_FINE, list(wet_limits)))
        wet_areas_pred.append(calculate_wet_area(img_pred, cfg.CELLSIZE_FINE, list(wet_limits)))
        wet_areas_int.append(calculate_wet_area(img_int, cfg.CELLSIZE_FINE, list(wet_limits)))

    wet_areas_sim = np.array(wet_areas_sim).T
    wet_areas_pred = np.array(wet_areas_pred).T
    wet_areas_int = np.array(wet_areas_int).T

    times = np.arange(1, n_timesteps + 1)

    plot_wet_area_vs_time(
        wet_areas_sim, wet_areas_pred, wet_areas_int, times,
        variable="Depth", units="m",
        limits=list(wet_limits),
        savepath=save_base_path + "_wet_area.png"
    )

    return {
        "pred": predictions_np,
        "y": y_crop,
        "bicubic": bicubic_crop,
        "subgrid": subgrid_crop,
    }