import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import os
import time
import pickle

from data_preparation import load_coarse_simulation_data, load_fine_simulation_data, load_fine_topographic_features, load_z_fine_data

# --- Custom Layers ---
class ResizeWithBicubic(nn.Module):
    def __init__(self, target_height, target_width):
        super().__init__()
        self.target_height = target_height
        self.target_width = target_width

    def forward(self, inputs):
        inputs = inputs.permute(0, 3, 1, 2) # Convertimos a N C H W para que lo entienda PyTorch
        out = F.interpolate(inputs, size=(self.target_height, self.target_width), mode='bicubic', align_corners=False)
        out = torch.clamp(out, min=0.0) #! Aseguramos que no haya valores negativos (es necesario porque bicubic puede generar negativos aunque no haya en la entrada)
        out = out.permute(0, 2, 3, 1) # Convertimos de nuevo a N H W C para ser consistentes con nuestro formato
        return out

class ResizeWithClassicalSubGrid(nn.Module):
    """
    Realiza un downscaling hidrológico usando el método analítico de "llenado por niveles".
    Puede operar sobre bloques rectangulares de M_rows x M_cols celdas gruesas.
    """
    def __init__(self, target_height, target_width):
        super().__init__()
        self.target_height = target_height
        self.target_width = target_width

    def _forward_single(self, coarse_depth, fine_dem):
        """
        Aplica el algoritmo de subgrid a celdas individuales (1x1).
        Nuevo: no es necesario que H_f y W_f sean múltiplos exactos de H_c y W_c.
        """
        B_fine, H_f_target, W_f_target, C_fine = fine_dem.shape
        B_coarse, H_c, W_c, C_coarse = coarse_depth.shape
        assert C_fine == 1 and C_coarse == 1
        assert B_coarse == B_fine 
        scale_h = math.ceil(H_f_target / H_c)
        scale_w = math.ceil(W_f_target / W_c)
        H_f_needed = H_c * scale_h 
        W_f_needed = W_c * scale_w  
        pad_h = H_f_needed - H_f_target
        pad_w = W_f_needed - W_f_target
        if pad_h > 0 or pad_w > 0:
            # F.pad orden: (left, right, top, bottom)
            fine_dem_padded = F.pad(fine_dem.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h), mode='replicate').permute(0, 2, 3, 1)
        else:
            fine_dem_padded = fine_dem

        N_sq = float(scale_h * scale_w)

        coarse_depth_nchw = coarse_depth.permute(0, 3, 1, 2)
        fine_dem_nchw = fine_dem_padded.permute(0, 3, 1, 2)
        
        coarse_depth_flat = coarse_depth_nchw.squeeze(1)
        V_target_flat = coarse_depth_flat * N_sq

        fine_dem_unfolded = fine_dem_nchw.unfold(2, scale_h, scale_h).unfold(3, scale_w, scale_w)
        z_blocks = fine_dem_unfolded.permute(0, 2, 3, 1, 4, 5).reshape(B_coarse, H_c, W_c, -1)

        # Solver de Llenado
        sorted_z, _ = torch.sort(z_blocks, dim=-1)
        z_diffs = sorted_z[..., 1:] - sorted_z[..., :-1]
        multipliers = torch.arange(1, N_sq, device=z_blocks.device, dtype=z_blocks.dtype)
        incremental_volumes = z_diffs * multipliers
        cumulative_volumes = torch.cumsum(incremental_volumes, dim=-1)
        cumulative_volumes = torch.cat([torch.zeros_like(V_target_flat.unsqueeze(-1)), cumulative_volumes], dim=-1)
        k = torch.sum(V_target_flat.unsqueeze(-1) > cumulative_volumes, dim=-1).clamp(min=1)
        k_idx = (k - 1).unsqueeze(-1)
        base_wse = torch.gather(sorted_z, -1, k_idx).squeeze(-1)
        volume_at_base = torch.gather(cumulative_volumes, -1, k_idx).squeeze(-1)
        remaining_volume = V_target_flat - volume_at_base
        additional_height = remaining_volume / k.float()
        final_wse = base_wse + additional_height

        # Calcular Salida 
        final_depths_blocks = torch.clamp(final_wse.unsqueeze(-1) - z_blocks, min=0)
        final_depths_unfolded = final_depths_blocks.view(B_coarse, H_c, W_c, scale_h, scale_w)
        out_nchw = final_depths_unfolded.permute(0, 1, 3, 2, 4).contiguous().view(B_fine, 1, H_f_needed, W_f_needed)
        out = out_nchw.permute(0, 2, 3, 1)
        if pad_h > 0 or pad_w > 0: # Recortar al tamaño target
            out = out[:, :H_f_target, :W_f_target, :]
        
        return out

    def forward(self, coarse_depth, fine_dem, group_size_rows, group_size_cols):
        """
        Aplica el algoritmo de subgrid agrupando group_size_rows x group_size_cols celdas gruesas.
        #! EN CASO DE QUE H_f y W_f NO SEAN MÚLTIPLOS EXACTOS DE H_c Y W_c, FORZAMOS MODO SINGLE (MUCHO LIO DE ÍNDICES SI NO)
        #TODO: MODIFICAR CÓDIGO PARA QUE HAGA FORWARD CON M>1
        """
        H_c, W_c = coarse_depth.shape[1], coarse_depth.shape[2]
        H_f, W_f = fine_dem.shape[1], fine_dem.shape[2]
        
        # Si la división no es exacta (o el usuario pide M=1), forzamos el método _forward_single que está listo para padding
        if (H_f % H_c != 0) or (W_f % W_c != 0) or (group_size_rows == 1 and group_size_cols == 1):
            return self._forward_single(coarse_depth, fine_dem)

        # --- 1. Configuración y Separación del Dominio ---
        B, H_f, W_f, _ = fine_dem.shape
        _, H_c, W_c, _ = coarse_depth.shape
        scale_h = H_f // H_c
        scale_w = W_f // W_c
        
        # Calcular dimensiones del bloque principal
        H_c_main = (H_c // group_size_rows) * group_size_rows
        W_c_main = (W_c // group_size_cols) * group_size_cols
        H_f_main = H_c_main * scale_h
        W_f_main = W_c_main * scale_w
        
        final_output = torch.zeros_like(fine_dem)
        
        # --- 2. Procesar el Bloque Principal Agrupado ---
        if H_c_main > 0 and W_c_main > 0:
            coarse_main = coarse_depth[:, :H_c_main, :W_c_main, :]
            fine_main = fine_dem[:, :H_f_main, :W_f_main, :]
            
            coarse_main_nchw = coarse_main.permute(0, 3, 1, 2)
            fine_main_nchw = fine_main.permute(0, 3, 1, 2)

            # A. AGREGAR ENTRADAS - promedio en bloques rectangulares
            avg_coarse_depth = F.avg_pool2d(
                coarse_main_nchw, 
                kernel_size=(group_size_rows, group_size_cols), 
                stride=(group_size_rows, group_size_cols)
            )
            
            H_g, W_g = avg_coarse_depth.shape[2], avg_coarse_depth.shape[3]
            group_scale_h = group_size_rows * scale_h
            group_scale_w = group_size_cols * scale_w
            N_sq_grouped = float(group_scale_h * group_scale_w)

            V_target_flat = avg_coarse_depth.squeeze(1) * N_sq_grouped
            
            # Desplegar el DEM fino en super-bloques rectangulares
            fine_dem_unfolded = fine_main_nchw.unfold(2, group_scale_h, group_scale_h).unfold(3, group_scale_w, group_scale_w)
            z_blocks = fine_dem_unfolded.permute(0, 2, 3, 1, 4, 5).reshape(B, H_g, W_g, -1)

            # B. APLICAR EL SOLVER DE LLENADO A LOS BLOQUES AGRUPADOS
            sorted_z, _ = torch.sort(z_blocks, dim=-1)
            z_diffs = sorted_z[..., 1:] - sorted_z[..., :-1]
            multipliers = torch.arange(1, N_sq_grouped, device=z_blocks.device, dtype=z_blocks.dtype)
            incremental_volumes = z_diffs * multipliers
            cumulative_volumes = torch.cumsum(incremental_volumes, dim=-1)
            cumulative_volumes = torch.cat([torch.zeros_like(V_target_flat.unsqueeze(-1)), cumulative_volumes], dim=-1)
            k = torch.sum(V_target_flat.unsqueeze(-1) > cumulative_volumes, dim=-1).clamp(min=1)
            k_idx = (k - 1).unsqueeze(-1)
            base_wse = torch.gather(sorted_z, -1, k_idx).squeeze(-1)
            volume_at_base = torch.gather(cumulative_volumes, -1, k_idx).squeeze(-1)
            remaining_volume = V_target_flat - volume_at_base
            additional_height = remaining_volume / k.float()
            final_wse = base_wse + additional_height

            # C. CALCULAR Y REFORMAR SALIDA
            final_depths_blocks = torch.clamp(final_wse.unsqueeze(-1) - z_blocks, min=0)
            final_depths_unfolded = final_depths_blocks.view(B, H_g, W_g, group_scale_h, group_scale_w)
            out_nchw = final_depths_unfolded.permute(0, 1, 3, 2, 4).contiguous().view(B, 1, H_f_main, W_f_main)
            
            final_output[:, :H_f_main, :W_f_main, :] = out_nchw.permute(0, 2, 3, 1)

        # --- 3. Procesar los Remanentes (si existen) ---
        rem_H = H_c - H_c_main
        rem_W = W_c - W_c_main

        # Remanentes en el eje W (columnas derechas)
        if rem_W > 0:
            coarse_rem_W = coarse_depth[:, :H_c_main, W_c_main:, :]
            fine_rem_W = fine_dem[:, :H_f_main, W_f_main:, :]
            if coarse_rem_W.numel() > 0:
                # Para remanentes en W, usar group_size_rows x 1 (mantener agrupamiento en filas)
                # rem_output_W = self.forward(coarse_rem_W, fine_rem_W, group_size_rows, 1) #! otra vertiente de pensamiento
                rem_output_W = self.forward(coarse_rem_W, fine_rem_W, group_size_rows, rem_W) # corregido, más parecido a conservar la lógica de tamaños de bloques
                final_output[:, :H_f_main, W_f_main:, :] = rem_output_W

        # Remanentes en el eje H (filas inferiores)
        if rem_H > 0:
            coarse_rem_H = coarse_depth[:, H_c_main:, :W_c_main, :]  # Solo tomar hasta W_c_main para no solaparse
            fine_rem_H = fine_dem[:, H_f_main:, :W_f_main, :]
            if coarse_rem_H.numel() > 0:
                # Para remanentes en H, usar 1 x group_size_cols (mantener agrupamiento en columnas)
                # rem_output_H = self.forward(coarse_rem_H, fine_rem_H, 1, group_size_cols) #! otra vertiente de pensamiento
                rem_output_H = self.forward(coarse_rem_H, fine_rem_H, rem_H, group_size_cols) # corregido, más parecido a conservar la lógica de tamaños de bloques
                final_output[:, H_f_main:, :W_f_main, :] = rem_output_H

        # Esquina inferior derecha (remanentes en ambos ejes)
        if rem_H > 0 and rem_W > 0:
            coarse_rem_corner = coarse_depth[:, H_c_main:, W_c_main:, :]
            fine_rem_corner = fine_dem[:, H_f_main:, W_f_main:, :]
            if coarse_rem_corner.numel() > 0:
                # Para la esquina, usar el método single (1x1)
                # rem_output_corner = self._forward_single(coarse_rem_corner, fine_rem_corner) #! otra vertiente de pensamiento
                rem_output_corner = self.forward(coarse_rem_corner, fine_rem_corner, rem_H, rem_W) # corregido, más parecido a conservar la lógica de tamaños de bloques
                final_output[:, H_f_main:, W_f_main:, :] = rem_output_corner
                
        return final_output

class CropAndConcat(nn.Module):
    """
    Custom layer to crop x1 to match x2 size and concatenate
    """
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        # Get sizes
        # x1: (N, C, H1, W1), x2: (N, C', H2, W2)
        x2_shape = x2.shape
        x1_shape = x1.shape

        # Calculate cropping
        height_diff = x1_shape[2] - x2_shape[2] # H1 - H2
        width_diff = x1_shape[3] - x2_shape[3]  # W1 - W2

        # Crop x1
        # PyTorch uses integer division for cropping indices
        crop_top = height_diff // 2
        crop_bottom = height_diff - crop_top # to handle odd differences
        crop_left = width_diff // 2
        crop_right = width_diff - crop_left

        # Slicing in PyTorch: tensor[N, C, H_start:H_end, W_start:W_end]
        # x1_cropped = x1[:, :, crop_top:x1_shape[2] - crop_bottom,
        #                      crop_left:x1_shape[3] - crop_right]

        # Use F.center_crop or manual slicing to achieve the center crop
        if height_diff > 0 or width_diff > 0:
            x1_cropped = x1[:, :, crop_top:crop_top + x2_shape[2],
                                 crop_left:crop_left + x2_shape[3]]
        else:
            x1_cropped = x1 # No cropping needed if x1 is smaller or equal

        # Concatenate along the channel dimension (axis=1 in PyTorch)
        return torch.cat([x1_cropped, x2], dim=1)

# --- He Initialization Helper ---
# def init_weights(m):
#     if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
#         # He Normal initialization
#         nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)
        
def init_weights_relu(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_weights_final_relu(m):  
    if isinstance(m, nn.Conv2d):  
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  
        if m.bias is not None:  
            nn.init.constant_(m.bias, 1.0)  # push outputs into ReLU's active region

def init_weights_leaky_relu(m, negative_slope=0.01):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=negative_slope)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

#--- Probamos con pesos inicializados positivos --- Juanma
def positive_weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # He positivo: similar a kaiming_uniform pero solo valores positivos
        fan_in = m.weight.data.size(1) * m.weight.data[0][0].numel()
        bound = math.sqrt(6.0 / fan_in)
        m.weight.data.uniform_(0, bound)  # Solo positivos
        if m.bias is not None:
            m.bias.data.fill_(0.01)  # Bias pequeño positivo

# --- Residual Unit ---
class ResUnit(nn.Module):
    """
    Residual Unit with He initialization and ReLU activation.
    """
    def __init__(self, C_in, C_out, name_prefix='res_unit'):
        super().__init__()
        self.name_prefix = name_prefix

        self.shortcut_conv = nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0)

        # First conv block
        self.bn1 = nn.BatchNorm2d(C_in) # BN before ReLU as per He initialization recommendation
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(C_in, C_out, kernel_size=3, stride=1, padding=1)

        # Second conv block
        self.bn2 = nn.BatchNorm2d(C_out)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(C_out, C_out, kernel_size=3, stride=1, padding=1)

        # Apply He initialization to all layers as they are followed by ReLU
        self.apply(init_weights_relu)

    def forward(self, x):
        x_input = x # Save the input for the residual connection

        shortcut = self.shortcut_conv(x_input)

        # First conv block
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        # Second conv block
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
                
        x = x + shortcut # Element-wise addition
        return x

# --- U-Net Model 3 niveles de profundidad: no realiza cálculos para entradas coarse nulas, directamente las pone a 0. Pero es menos robusto y más lento---
class UNetWithTopographicFeatures3_PREDICCIONES_NULAS(nn.Module):
    def __init__(self, last_activation_function, alpha_for_leaky_relu=0.01, coarse_input_channels=1, output_channels=1, fine_grid_topographic_features=None):
        super().__init__()
        self.coarse_input_channels = coarse_input_channels
        self.output_channels = output_channels
        self.fine_grid_topographic_features = fine_grid_topographic_features

        # Determine input channels for the first encoder block
        initial_input_channels = coarse_input_channels + fine_grid_topographic_features

        # Encoder Path
        # Level 1
        self.enc_res1 = ResUnit(initial_input_channels, 64, 'enc_res1')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Level 2
        self.enc_res2 = ResUnit(64, 128, 'enc_res2')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Level 3
        self.enc_res3 = ResUnit(128, 256, 'enc_res3')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Bottleneck
        self.bottleneck = ResUnit(256, 512, 'bottleneck_res')

        # Decoder Path
        # Level 3 decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.crop_concat3 = CropAndConcat()
        self.dec_res3 = ResUnit(256 + 256, 256, 'dec_res3') # Channels: up3 output + enc_res3 output

        # Level 2 decoder
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.crop_concat2 = CropAndConcat()
        self.dec_res2 = ResUnit(128 + 128, 128, 'dec_res2') # Channels: up2 output + enc_res2 output

        # Level 1 decoder
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.crop_concat1 = CropAndConcat()
        self.dec_res1 = ResUnit(64 + 64, 64, 'dec_res1') # Channels: up1 output + enc_res1 output

        # Apply He initialization to all layers as they are followed by ReLU
        self.apply(init_weights_relu)
        
        # Final layers
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1, stride=1, padding=0)


        if 'leaky' in last_activation_function.lower():
            alpha = alpha_for_leaky_relu
            self.final_conv.apply(lambda m: init_weights_leaky_relu(m, negative_slope=alpha))
            self.output_activation = nn.LeakyReLU(negative_slope=alpha)
        elif 'relu' in last_activation_function.lower():
            self.final_conv.apply(init_weights_final_relu)
            self.output_activation = nn.ReLU()
        elif 'none' in last_activation_function.lower() or last_activation_function is None:
            self.output_activation = nn.Identity()
        elif 'sigmoid' in last_activation_function.lower():
            self.output_activation = nn.Sigmoid()
        elif 'tanh' in last_activation_function.lower():
            self.output_activation = nn.Tanh()
        elif 'softplus' in last_activation_function.lower():
            self.output_activation = nn.Softplus()
        else:
            raise ValueError(f"Unsupported activation function: {last_activation_function}")
            
        # self.output_activation = nn.SiLU()
        # self.output_activation = nn.GELU()
        # self.apply(init_weights_relu) # descomment if RELU
        # self.output_activation = nn.ReLU(inplace=True) # descomment if RELU

        

    def forward(self, coarse_input, fine_topographical_input):
        # coarse_input: (N, Hc, Wc, Cc) -> PyTorch needs (N, Cc, Hc, Wc)
        coarse_input = coarse_input.permute(0, 3, 1, 2)
        fine_topographical_input = fine_topographical_input.permute(0, 3, 1, 2) # (N, C_topo, Hf, Wf)

        # target_height and target_width are derived from fine_topographical_input
        N_images = coarse_input.shape[0]
        fine_grid_height = fine_topographical_input.shape[2]
        fine_grid_width = fine_topographical_input.shape[3]
        C_out = self.output_channels

        # Detectar cuáles son completamente cero
        zero_mask = (coarse_input.abs().sum(dim=(1, 2, 3)) == 0)
        nonzero_idx = (~zero_mask).nonzero(as_tuple=True)[0]

        output = None

        if len(nonzero_idx) > 0: #! Solamente perdemos tiempo en calcular la salida cuando la entrada es no nula
            coarse_nonzero = coarse_input[nonzero_idx]
            topo_nonzero = fine_topographical_input[nonzero_idx]

            coarse_input_interpolated = F.interpolate(coarse_nonzero,
                                                    size=(fine_grid_height, fine_grid_width),
                                                    mode='bicubic', align_corners=False)

            # Concatenate main input and topographic features
            input_concatenated = torch.cat([coarse_input_interpolated, topo_nonzero], dim=1)

            # ENCODER PATH
            enc_res1 = self.enc_res1(input_concatenated)
            pool1 = self.pool1(enc_res1)

            enc_res2 = self.enc_res2(pool1)
            pool2 = self.pool2(enc_res2)

            enc_res3 = self.enc_res3(pool2)
            pool3 = self.pool3(enc_res3)

            # Bottleneck
            bottleneck = self.bottleneck(pool3)

            # DECODER PATH
            up3 = self.up3(bottleneck)
            merge3 = self.crop_concat3(enc_res3, up3)
            dec_res3 = self.dec_res3(merge3)

            up2 = self.up2(dec_res3)
            merge2 = self.crop_concat2(enc_res2, up2)
            dec_res2 = self.dec_res2(merge2)

            up1 = self.up1(dec_res2)
            merge1 = self.crop_concat1(enc_res1, up1)
            dec_res1 = self.dec_res1(merge1)

            # Final layers
            output_nonzero = self.final_conv(dec_res1)
            output_nonzero = self.output_activation(output_nonzero)
            output_nonzero = output_nonzero.permute(0, 2, 3, 1)  # (N, H, W, C) for output consistency

            H_out, W_out = output_nonzero.shape[1], output_nonzero.shape[2]
            output = torch.zeros((N_images, H_out, W_out, C_out), 
                         device=coarse_input.device, dtype=coarse_input.dtype) # lo llenamos incialmente de ceros con las dimensiones correctas

            # Finally insert the non-zero outputs back into the full output tensor
            output[nonzero_idx] = output_nonzero
        
        else: #! en caso (improbable) de que todo el batch pasado sean imágenes de ceros (es lo único que podría dar problemas)
            pixels_cropped = 88 # hardcoded (3 level UNet)
            output = torch.zeros((N_images, fine_grid_height - pixels_cropped, fine_grid_width - pixels_cropped, C_out), device=coarse_input.device, dtype=coarse_input.dtype)

        return output

# --- U-Net Model 3 niveles de profundidad ---
class UNetWithTopographicFeatures3(nn.Module):
    def __init__(self, last_activation_function, alpha_for_leaky_relu=0.01, coarse_input_channels=1, output_channels=1, fine_grid_topographic_features=None):
        super().__init__()
        self.coarse_input_channels = coarse_input_channels
        self.output_channels = output_channels
        self.fine_grid_topographic_features = fine_grid_topographic_features

        # Determine input channels for the first encoder block
        initial_input_channels = coarse_input_channels + fine_grid_topographic_features

        # Encoder Path
        # Level 1
        self.enc_res1 = ResUnit(initial_input_channels, 64, 'enc_res1')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Level 2
        self.enc_res2 = ResUnit(64, 128, 'enc_res2')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Level 3
        self.enc_res3 = ResUnit(128, 256, 'enc_res3')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Bottleneck
        self.bottleneck = ResUnit(256, 512, 'bottleneck_res')

        # Decoder Path
        # Level 3 decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.crop_concat3 = CropAndConcat()
        self.dec_res3 = ResUnit(256 + 256, 256, 'dec_res3') # Channels: up3 output + enc_res3 output

        # Level 2 decoder
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.crop_concat2 = CropAndConcat()
        self.dec_res2 = ResUnit(128 + 128, 128, 'dec_res2') # Channels: up2 output + enc_res2 output

        # Level 1 decoder
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.crop_concat1 = CropAndConcat()
        self.dec_res1 = ResUnit(64 + 64, 64, 'dec_res1') # Channels: up1 output + enc_res1 output

        # Apply He initialization to all layers as they are followed by ReLU
        self.apply(init_weights_relu)
        
        # Final layers
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1, stride=1, padding=0)


        if 'leaky' in last_activation_function.lower():
            alpha = alpha_for_leaky_relu
            self.final_conv.apply(lambda m: init_weights_leaky_relu(m, negative_slope=alpha))
            self.output_activation = nn.LeakyReLU(negative_slope=alpha)
        elif 'relu' in last_activation_function.lower():
            self.final_conv.apply(init_weights_final_relu)
            self.output_activation = nn.ReLU()
        elif 'none' in last_activation_function.lower() or last_activation_function is None:
            self.output_activation = nn.Identity()
        elif 'sigmoid' in last_activation_function.lower():
            self.output_activation = nn.Sigmoid()
        elif 'tanh' in last_activation_function.lower():
            self.output_activation = nn.Tanh()
        elif 'softplus' in last_activation_function.lower():
            self.output_activation = nn.Softplus()
        else:
            raise ValueError(f"Unsupported activation function: {last_activation_function}")
            
        # self.output_activation = nn.SiLU()
        # self.output_activation = nn.GELU()
        # self.apply(init_weights_relu) # descomment if RELU
        # self.output_activation = nn.ReLU(inplace=True) # descomment if RELU

        

    def forward(self, coarse_input, fine_topographical_input):
        # PyTorch needs (N, Cc, Hc, Wc)

        # Bicubic interpolation (ResizeWithBicubic)
        # target_height and target_width are derived from fine_topographical_input
        fine_grid_height = fine_topographical_input.shape[1]
        fine_grid_width = fine_topographical_input.shape[2]
        bicubic_resizer = ResizeWithBicubic(fine_grid_height, fine_grid_width).to(coarse_input.device)
        coarse_input_interpolated_nhwc = bicubic_resizer(coarse_input)

        coarse_input_interpolated_nchw = coarse_input_interpolated_nhwc.permute(0, 3, 1, 2)
        fine_topographical_input_nchw = fine_topographical_input.permute(0, 3, 1, 2)

        # Concatenate main input and topographic features
        input_concatenated = torch.cat([coarse_input_interpolated_nchw, fine_topographical_input_nchw], dim=1)

        # ENCODER PATH
        enc_res1 = self.enc_res1(input_concatenated)
        pool1 = self.pool1(enc_res1)
        enc_res2 = self.enc_res2(pool1)
        pool2 = self.pool2(enc_res2)
        enc_res3 = self.enc_res3(pool2)
        pool3 = self.pool3(enc_res3)

        # Bottleneck
        bottleneck = self.bottleneck(pool3)

        # DECODER PATH
        up3 = self.up3(bottleneck)
        up3 = F.interpolate(up3, size=enc_res3.shape[-2:], mode='bilinear', align_corners=False)
        merge3 = self.crop_concat3(enc_res3, up3)
        dec_res3 = self.dec_res3(merge3)
        up2 = self.up2(dec_res3)
        up2 = F.interpolate(up2, size=enc_res2.shape[-2:], mode='bilinear', align_corners=False)
        merge2 = self.crop_concat2(enc_res2, up2)
        dec_res2 = self.dec_res2(merge2)
        up1 = self.up1(dec_res2)
        up1 = F.interpolate(up1, size=enc_res1.shape[-2:], mode='bilinear', align_corners=False)
        merge1 = self.crop_concat1(enc_res1, up1)
        dec_res1 = self.dec_res1(merge1)

        # Final layers
        output = self.final_conv(dec_res1)
        output = self.output_activation(output)

        # Permute back to (N, H, W, C) if needed for consistency with original output
        # Or keep as (N, C, H, W) for further PyTorch processing
        output = output.permute(0, 2, 3, 1)  # Convert to (N, H, W, C) for output consistency
        return output


class UNetWithTopographicFeatures3_SG(nn.Module):
    def __init__(self, last_activation_function, alpha_for_leaky_relu=0.01, coarse_input_channels=1, output_channels=1, fine_grid_topographic_features=None, group_size_rows=1, group_size_cols=1):
        super().__init__()
        self.coarse_input_channels = coarse_input_channels
        self.output_channels = output_channels
        self.fine_grid_topographic_features = fine_grid_topographic_features
        self.group_size_rows = group_size_rows
        self.group_size_cols = group_size_cols

        # Determine input channels for the first encoder block
        initial_input_channels = coarse_input_channels + fine_grid_topographic_features

        # Encoder Path
        # Level 1
        self.enc_res1 = ResUnit(initial_input_channels, 64, 'enc_res1')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Level 2
        self.enc_res2 = ResUnit(64, 128, 'enc_res2')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Level 3
        self.enc_res3 = ResUnit(128, 256, 'enc_res3')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Bottleneck
        self.bottleneck = ResUnit(256, 512, 'bottleneck_res')

        # Decoder Path
        # Level 3 decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.crop_concat3 = CropAndConcat()
        self.dec_res3 = ResUnit(256 + 256, 256, 'dec_res3') # Channels: up3 output + enc_res3 output

        # Level 2 decoder
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.crop_concat2 = CropAndConcat()
        self.dec_res2 = ResUnit(128 + 128, 128, 'dec_res2') # Channels: up2 output + enc_res2 output

        # Level 1 decoder
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.crop_concat1 = CropAndConcat()
        self.dec_res1 = ResUnit(64 + 64, 64, 'dec_res1') # Channels: up1 output + enc_res1 output

        # Apply He initialization to all layers as they are followed by ReLU
        self.apply(init_weights_relu)
        
        # Final layers
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1, stride=1, padding=0)


        if 'leaky' in last_activation_function.lower():
            alpha = alpha_for_leaky_relu
            self.final_conv.apply(lambda m: init_weights_leaky_relu(m, negative_slope=alpha))
            self.output_activation = nn.LeakyReLU(negative_slope=alpha)
        elif 'relu' in last_activation_function.lower():
            self.final_conv.apply(init_weights_final_relu)
            self.output_activation = nn.ReLU()
        elif 'none' in last_activation_function.lower() or last_activation_function is None:
            self.output_activation = nn.Identity()
        elif 'sigmoid' in last_activation_function.lower():
            self.output_activation = nn.Sigmoid()
        elif 'tanh' in last_activation_function.lower():
            self.output_activation = nn.Tanh()
        elif 'softplus' in last_activation_function.lower():
            self.output_activation = nn.Softplus()
        else:
            raise ValueError(f"Unsupported activation function for last layer of the UNet: {last_activation_function}")
            
        # self.output_activation = nn.SiLU()
        # self.output_activation = nn.GELU()
        # self.apply(init_weights_relu) # descomment if RELU
        # self.output_activation = nn.ReLU(inplace=True) # descomment if RELU

        

    def forward(self, coarse_input, fine_topographical_inputs):
        # PyTorch needs (N, Cc, Hc, Wc)

        # SubGrid algorithm (ResizeWithClassicalSubGrid)
        # target_height and target_width are derived from fine_topographical_inputs
        fine_grid_height = fine_topographical_inputs.shape[1]
        fine_grid_width = fine_topographical_inputs.shape[2]
        subgrid_resizer = ResizeWithClassicalSubGrid(fine_grid_height, fine_grid_width).to(coarse_input.device)
        coarse_input_interpolated_nhwc = subgrid_resizer(coarse_input, fine_topographical_inputs[:,:,:,:1], group_size_rows=self.group_size_rows, group_size_cols=self.group_size_cols)  # Only DEM (first) channel is needed for subgrid

        coarse_input_interpolated_nchw = coarse_input_interpolated_nhwc.permute(0, 3, 1, 2)
        fine_topographical_inputs_nchw = fine_topographical_inputs.permute(0, 3, 1, 2)

        # Concatenate main input and topographic features
        input_concatenated = torch.cat([coarse_input_interpolated_nchw, fine_topographical_inputs_nchw], dim=1)

        # ENCODER PATH
        enc_res1 = self.enc_res1(input_concatenated)
        pool1 = self.pool1(enc_res1)
        enc_res2 = self.enc_res2(pool1)
        pool2 = self.pool2(enc_res2)
        enc_res3 = self.enc_res3(pool2)
        pool3 = self.pool3(enc_res3)

        # Bottleneck
        bottleneck = self.bottleneck(pool3)

        # DECODER PATH
        up3 = self.up3(bottleneck)
        up3 = F.interpolate(up3, size=enc_res3.shape[-2:], mode='bilinear', align_corners=False)
        merge3 = self.crop_concat3(enc_res3, up3)

        dec_res3 = self.dec_res3(merge3)

        up2 = self.up2(dec_res3)
        up2 = F.interpolate(up2, size=enc_res2.shape[-2:], mode='bilinear', align_corners=False)
        merge2 = self.crop_concat2(enc_res2, up2)

        dec_res2 = self.dec_res2(merge2)

        up1 = self.up1(dec_res2)
        up1 = F.interpolate(up1, size=enc_res1.shape[-2:], mode='bilinear', align_corners=False)
        merge1 = self.crop_concat1(enc_res1, up1)
        dec_res1 = self.dec_res1(merge1)

        # Final layers
        output = self.final_conv(dec_res1)
        output = self.output_activation(output)

        # Permute back to (N, H, W, C) if needed for consistency with original output
        # Or keep as (N, C, H, W) for further PyTorch processing
        output = output.permute(0, 2, 3, 1)  # Convert to (N, H, W, C) for output consistency
        return output



# --- U-Net Model 2 niveles de profundidad ---
class UNetWithTopographicFeatures2(nn.Module):
    def __init__(self, last_activation_function, alpha_for_leaky_relu=0.01, coarse_input_channels=1, output_channels=1, fine_grid_topographic_features=None):
        super().__init__()
        self.coarse_input_channels = coarse_input_channels
        self.output_channels = output_channels
        self.fine_grid_topographic_features = fine_grid_topographic_features

        # Determine input channels for the first encoder block
        # coarse_input (interpolated) + fine_topographical_input
        initial_input_channels = coarse_input_channels + fine_grid_topographic_features

        # Encoder Path
        # Level 1
        self.enc_res1 = ResUnit(initial_input_channels, 64, 'enc_res1')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Level 2
        self.enc_res2 = ResUnit(64, 128, 'enc_res2')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Level 3
        # self.enc_res3 = ResUnit(128, 256, 'enc_res3')
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Bottleneck
        self.bottleneck = ResUnit(128, 256, 'bottleneck_res')

        # Decoder Path
        # Level 3 decoder
        # self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        # self.crop_concat3 = CropAndConcat()
        # self.dec_res3 = ResUnit(256 + 256, 256, 'dec_res3') # Channels: up3 output + enc_res3 output

        # Level 2 decoder
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.crop_concat2 = CropAndConcat()
        self.dec_res2 = ResUnit(128 + 128, 128, 'dec_res2') # Channels: up2 output + enc_res2 output

        # Level 1 decoder
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.crop_concat1 = CropAndConcat()
        self.dec_res1 = ResUnit(64 + 64, 64, 'dec_res1') # Channels: up1 output + enc_res1 output

        # Apply He initialization to all layers as they are followed by ReLU
        self.apply(init_weights_relu)
        
        # Final layers
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1, stride=1, padding=0)


        if 'leaky' in last_activation_function.lower():
            alpha = alpha_for_leaky_relu
            self.final_conv.apply(lambda m: init_weights_leaky_relu(m, negative_slope=alpha))
            self.output_activation = nn.LeakyReLU(negative_slope=alpha)
        elif 'relu' in last_activation_function.lower():
            self.final_conv.apply(init_weights_final_relu)
            self.output_activation = nn.ReLU()
        elif 'sigmoid' in last_activation_function.lower():
            self.output_activation = nn.Sigmoid()
        elif 'tanh' in last_activation_function.lower():
            self.output_activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {last_activation_function}")
            
        # self.output_activation = nn.SiLU()
        # self.output_activation = nn.GELU()
        # self.apply(init_weights_relu) # descomment if RELU
        # self.output_activation = nn.ReLU(inplace=True) # descomment if RELU

        

    def forward(self, coarse_input, fine_topographical_input):
        # coarse_input: (N, Hc, Wc, Cc) -> PyTorch needs (N, Cc, Hc, Wc)
        coarse_input = coarse_input.permute(0, 3, 1, 2)
        fine_topographical_input = fine_topographical_input.permute(0, 3, 1, 2) # (N, C_topo, Hf, Wf)

        # target_height and target_width are derived from fine_topographical_input
        fine_grid_height = fine_topographical_input.shape[2]
        fine_grid_width = fine_topographical_input.shape[3]
        coarse_input_interpolated = F.interpolate(coarse_input,
                                                  size=(fine_grid_height, fine_grid_width),
                                                  mode='bicubic', align_corners=False)

        # Concatenate main input and topographic features
        input_concatenated = torch.cat([coarse_input_interpolated, fine_topographical_input], dim=1)

        # ENCODER PATH
        enc_res1 = self.enc_res1(input_concatenated)
        pool1 = self.pool1(enc_res1)

        enc_res2 = self.enc_res2(pool1)
        pool2 = self.pool2(enc_res2)

        # enc_res3 = self.enc_res3(pool2)
        # pool3 = self.pool3(enc_res3)

        # Bottleneck
        bottleneck = self.bottleneck(pool2)

        # DECODER PATH
        # up3 = self.up3(bottleneck)
        # merge3 = self.crop_concat3(enc_res3, up3)
        # dec_res3 = self.dec_res3(merge3)

        up2 = self.up2(bottleneck)
        merge2 = self.crop_concat2(enc_res2, up2)
        dec_res2 = self.dec_res2(merge2)

        up1 = self.up1(dec_res2)
        merge1 = self.crop_concat1(enc_res1, up1)
        dec_res1 = self.dec_res1(merge1)

        # Final layers
        output = self.final_conv(dec_res1)
        output = self.output_activation(output)

        # Permute back to (N, H, W, C) if needed for consistency with original output
        # Or keep as (N, C, H, W) for further PyTorch processing
        output = output.permute(0, 2, 3, 1) # Convert to (N, H, W, C) for output consistency

        return output

# --- Training and Evaluation Functions ---

def custom_non_negative_mse_loss(y_true, y_pred):
    # Standard Mean Squared Error (MSE)
    mse = F.mse_loss(y_pred, y_true, reduction='mean')
    
    # Penalty for negative predictions
    # torch.clamp(y_pred, max=0.0) will result in y_pred if y_pred < 0, else 0
    # Then, we take the absolute value or square it to get a positive penalty.
    # Using torch.relu(-y_pred) creates a tensor where negative values become positive, and positive/zero values become zero.
    negative_penalty_component = torch.relu(-y_pred)
    
    # You can experiment with different forms of penalty (e.g., just negative_penalty_component**2 for a quadratic penalty)
    negative_penalty = torch.mean(negative_penalty_component)
    
    # Combine MSE with the negative penalty
    penalty_weight = 20000000.0 # Adjust this weight
    
    total_loss = mse + penalty_weight * negative_penalty
    
    return total_loss

def weighted_mse_loss(y_true, y_pred, threshold=0.001, weight_high=100.0, weight_low=1.0):    
    weights = torch.where(y_true > threshold,
                          torch.tensor(weight_high, device=y_true.device, dtype=y_true.dtype),
                          torch.tensor(weight_low, device=y_true.device, dtype=y_true.dtype))
    
    loss = torch.mean(weights * (y_pred - y_true) ** 2)
    return loss


def weighted_mse_loss2(y_true, y_pred, threshold=0.001, weight_high=100.0, weight_low=1.0):
    eps = 1e-6  # Para evitar divisiones por cero
    weights = torch.where(
        y_true > threshold,
        weight_high / (y_true + eps),  # Más peso cuando y_true es pequeño, pero siempre superando un umbral
        torch.tensor(weight_low, device=y_true.device, dtype=y_true.dtype)
    )
    
    loss = torch.mean(weights * (y_pred - y_true) ** 2)
    return loss

def mse_pesado(y_pred, y_true, gamma=3.0, alpha=0.05): # ATENTO AL ORDEN EN QUE SE PASAN y_true, y_pred PORQUE AQUÍ IMPORTA!!! 
    weights_tensor = 1 + gamma * y_true ** alpha  # y = 1  + gamma*x^alpha 
    loss = torch.mean(weights_tensor * (y_pred - y_true) ** 2)
    return loss

def weighted_mse_loss2_wse(y_true, y_pred, z_true, threshold=0.001, weight_high=100.0, weight_low=1.0):
    eps = 1e-6
    
    # Aseguramos que no sean negativas las h predichas: h = max(0, WSE-z)
    h_true = torch.clamp(y_true - z_true, min=0.0)
    h_pred = torch.clamp(y_pred - z_true, min=0.0)

    weights = torch.where(
        h_true > threshold,
        weight_high / (h_true + eps),
        torch.tensor(weight_low, device=h_true.device, dtype=h_true.dtype)
    )
    
    loss = torch.mean(weights * (h_pred - h_true) ** 2)
    return loss

def weighted_mse_loss3(y_true, y_pred, threshold=0.001, weight_high=100.0, weight_low=1.0):
    weights = torch.where(
        y_true > threshold,
        weight_high * y_true,
        torch.tensor(weight_low, device=y_true.device, dtype=y_true.dtype)
    )
    
    loss = torch.mean(weights * (y_pred - y_true) ** 2)
    return loss

def conservation_volume_loss(X_coarse_cropped, y_pred): # TODO: comprobar que en las simulaciones se cumpla lo que aquí intento imponer, no vaya a ser que no se dé y lo esté forzando yo...
    """
    Impone conservación de volumen: la media de alturas predichas fine debe aproximar la altura coarse.

    Args:
        X_coarse_cropped (Tensor): Coarse grid de entrada, shape (N, H_coarse, W_coarse, C)
        y_pred (Tensor): Predicción en fine grid, shape (N, H_fine, W_fine, C)

    Returns:
        Scalar tensor con pérdida de conservación de volumen
    """
    N, H_fine, W_fine, C = y_pred.shape 
    _, H_coarse, W_coarse, _ = X_coarse_cropped.shape

    # Calcular downscale factor
    downscale_h = H_fine // H_coarse
    downscale_w = W_fine // W_coarse

    if downscale_h != downscale_w:
        raise ValueError(f"El factor de downscaling no es igual en H y W: {downscale_h} vs {downscale_w}")
    downscale_factor = downscale_h

    # Transponer a NCHW para pooling (si no, en Pytorch no irá)
    y_pred_nchw = y_pred.permute(0, 3, 1, 2)  # (N, C, H, W)

    # Promediar bloques fine → coarse
    y_pred_coarse_nchw = F.avg_pool2d(y_pred_nchw, kernel_size=downscale_factor, stride=downscale_factor)

    # Volver a NHWC
    y_pred_coarse = y_pred_coarse_nchw.permute(0, 2, 3, 1)  # (N, H_coarse, W_coarse, C)

    # Calcular MSE
    loss = F.mse_loss(y_pred_coarse, X_coarse_cropped)
    return loss

def loss_mse2_conservation(y_true, y_pred, X_coarse_cropped, lambda_vol=200.0):
    loss1 = weighted_mse_loss2(y_true, y_pred)
    loss2 = conservation_volume_loss(X_coarse_cropped, y_pred)
    return loss1 + lambda_vol * loss2

def loss_mse3_conservation(y_true, y_pred, X_coarse_cropped, lambda_vol=100.0):
    loss1 = weighted_mse_loss3(y_true, y_pred)
    loss2 = conservation_volume_loss(X_coarse_cropped, y_pred)
    return loss1 + lambda_vol * loss2

def lossWithRelative(y_true, y_pred, X_coarse_cropped, lambda_mse=1.0, lambda_rel=5.0, lambda_vol=10.0):
    eps = 1e-6
    abs_error = torch.abs(y_pred - y_true)
    rel_error = torch.abs(y_pred - y_true) / (y_true + eps)
    loss1 = torch.mean(abs_error**2)
    loss2 = torch.mean(rel_error**2)
    loss3 = conservation_volume_loss(X_coarse_cropped, y_pred)
    return lambda_mse*loss1 + lambda_rel*loss2 + lambda_vol*loss3

class MAEWithMassConservation(nn.Module):
    def __init__(self, alpha=1.0, normalize=True):
        super(MAEWithMassConservation, self).__init__()
        self.alpha = alpha
        self.normalize = normalize

    def forward(self, y_pred, y_true):
        mae_loss = F.l1_loss(y_pred, y_true, reduction='mean')
        mass_diff = torch.sum(y_pred) - torch.sum(y_true)
        if self.normalize:
            mass_conservation_loss = torch.abs(mass_diff) / y_true.numel()
        else:
            mass_conservation_loss = torch.abs(mass_diff)

        total_loss = mae_loss + self.alpha * mass_conservation_loss
        return total_loss

class MAEWetDryLoss(nn.Module):
    def __init__(self, alpha=1.0, threshold=1e-3, eps=1e-3):
        """
        alpha: peso para la parte 'dry'
        threshold: valor a partir del cual se considera 'wet'
        eps: pequeño valor para evitar divisiones por cero
        """
        super(MAEWetDryLoss, self).__init__()
        self.alpha = alpha
        self.threshold = threshold
        self.eps = eps

    def forward(self, y_pred, y_true):
        wet_mask = (y_true >= self.threshold).float()
        dry_mask = (y_true < self.threshold).float()

        N_wet = torch.sum(wet_mask) + self.eps
        N_dry = torch.sum(dry_mask) + self.eps

        abs_err = torch.abs(y_true - y_pred)

        wet_loss = torch.sum(abs_err * wet_mask) / N_wet
        dry_loss = torch.sum(abs_err * dry_mask) / N_dry

        total_loss = wet_loss + self.alpha * dry_loss
        return total_loss

class MAEMassConservationAndWetDryLoss(nn.Module):
    def __init__(self, alpha_mass=1.0, alpha_dry=1.0, threshold=1e-3, eps=1e-3, normalize_mass=True):
        """
        alpha_mass: peso para el término de conservación de masa
        alpha_dry: peso relativo para la parte 'dry'
        threshold: valor a partir del cual se considera 'wet'
        eps: pequeño valor para evitar divisiones por cero
        normalize_mass: si True, divide la pérdida de masa por el nº de elementos
        """
        super(MAEMassConservationAndWetDryLoss, self).__init__()
        self.alpha_mass = alpha_mass
        self.alpha_dry = alpha_dry
        self.threshold = threshold
        self.eps = eps
        self.normalize_mass = normalize_mass

    def forward(self, y_pred, y_true):
        # --- Wet vs Dry ---
        wet_mask = (y_true >= self.threshold).float()
        dry_mask = (y_true < self.threshold).float()
        N_wet = torch.sum(wet_mask)
        N_dry = torch.sum(dry_mask)
        abs_err = torch.abs(y_true - y_pred)
        wet_loss = torch.sum(abs_err * wet_mask) / max(1, N_wet)
        dry_loss = torch.sum(abs_err * dry_mask) / max(1, N_dry)
        wet_dry_loss = wet_loss + self.alpha_dry * dry_loss

        # --- Mass conservation ---
        mass_diff = torch.sum(y_pred) - torch.sum(y_true)
        if self.normalize_mass:
            mass_loss = torch.abs(mass_diff) / y_true.numel()
        else:
            mass_loss = torch.abs(mass_diff)

        # --- Total ---
        total_loss = wet_dry_loss + self.alpha_mass * mass_loss
        return total_loss


class SoftCSILoss(nn.Module):
    def __init__(self, threshold=1e-3, eps=1e-6, smooth_factor=1e-2):
        """
        SoftCSI (Soft IoU) Loss con threshold suavizado.

        Args:
            threshold: umbral de calado para considerar 'wet'
            eps: valor pequeño para evitar divisiones por cero
            smooth_factor: controla lo abrupta que es la transición en torno al threshold
        """
        super(SoftCSILoss, self).__init__()
        self.threshold = threshold
        self.eps = eps
        self.smooth_factor = smooth_factor

    def forward(self, y_pred, y_true):
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shapes no coinciden: pred={y_pred.shape}, true={y_true.shape}")

        # Aplanar
        y_pred_flat = y_pred.reshape(-1)
        y_true_flat = y_true.reshape(-1)

        # Convertir a "probabilidad de estar inundado" usando sigmoide suavizada
        wet_true = torch.sigmoid((y_true_flat - self.threshold) / self.smooth_factor)
        wet_pred = torch.sigmoid((y_pred_flat - self.threshold) / self.smooth_factor)

        # Soft IoU
        intersection = torch.sum(wet_true * wet_pred)
        union = torch.sum(wet_true) + torch.sum(wet_pred) - intersection

        soft_csi = intersection / (union + self.eps)
        return 1.0 - soft_csi

class MAESoftCSILoss(nn.Module):
    def __init__(self, alpha_softcsi=1.0, threshold=1e-3, eps=1e-6, smooth_factor=1e-2):
        """
        Loss combinada: MAE + alpha_softcsi * (1 - SoftCSI)

        Args:
            alpha_softcsi: peso del término SoftCSI
            threshold: umbral de calado para considerar 'wet'
            eps: valor pequeño para evitar divisiones por cero
            smooth_factor: controla lo abrupta que es la transición en torno al threshold
        """
        super(MAESoftCSILoss, self).__init__()
        self.alpha_softcsi = alpha_softcsi
        self.threshold = threshold
        self.eps = eps
        self.smooth_factor = smooth_factor

    def forward(self, y_pred, y_true):
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shapes no coinciden al calcular la Loss: pred={y_pred.shape}, true={y_true.shape}")

        # --- MAE ---
        mae_loss = torch.mean(torch.abs(y_pred - y_true))

        # --- SoftCSI ---
        y_pred_flat = y_pred.reshape(-1)
        y_true_flat = y_true.reshape(-1)

        wet_true = torch.sigmoid((y_true_flat - self.threshold) / self.smooth_factor)
        wet_pred = torch.sigmoid((y_pred_flat - self.threshold) / self.smooth_factor)

        intersection = torch.sum(wet_true * wet_pred)
        union = torch.sum(wet_true) + torch.sum(wet_pred) - intersection

        soft_csi = intersection / (union + self.eps)
        softcsi_loss = 1.0 - soft_csi

        # --- Total ---
        total_loss = mae_loss + self.alpha_softcsi * softcsi_loss
        return total_loss

class MSESoftCSILoss(nn.Module):
    def __init__(self, alpha_softcsi=1.0, threshold=1e-3, eps=1e-6, smooth_factor=1e-2):
        """
        Loss combinada: MSE + alpha_softcsi * (1 - SoftCSI)

        Args:
            alpha_softcsi: peso del término SoftCSI
            threshold: umbral de calado para considerar 'wet'
            eps: valor pequeño para evitar divisiones por cero
            smooth_factor: controla lo abrupta que es la transición en torno al threshold
        """
        super(MSESoftCSILoss, self).__init__()
        self.alpha_softcsi = alpha_softcsi
        self.threshold = threshold
        self.eps = eps
        self.smooth_factor = smooth_factor

    def forward(self, y_pred, y_true):
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shapes no coinciden al calcular la Loss: pred={y_pred.shape}, true={y_true.shape}")

        # --- MSE ---
        mse_loss = F.mse_loss(y_pred, y_true, reduction='mean')

        # --- SoftCSI ---
        y_pred_flat = y_pred.reshape(-1)
        y_true_flat = y_true.reshape(-1)

        wet_true = torch.sigmoid((y_true_flat - self.threshold) / self.smooth_factor)
        wet_pred = torch.sigmoid((y_pred_flat - self.threshold) / self.smooth_factor)

        intersection = torch.sum(wet_true * wet_pred)
        union = torch.sum(wet_true) + torch.sum(wet_pred) - intersection

        soft_csi = intersection / (union + self.eps)
        softcsi_loss = 1.0 - soft_csi

        # --- Total ---
        total_loss = mse_loss + self.alpha_softcsi * softcsi_loss
        return total_loss
    
class MAEMSELoss(nn.Module):
    def __init__(self, omega_mse=1.0):
        super(MAEMSELoss, self).__init__()
        self.omega_mse = omega_mse

    def forward(self, y_pred, y_true):
        mae_loss = F.l1_loss(y_pred, y_true, reduction='mean')
        mse_loss = F.mse_loss(y_pred, y_true, reduction='mean')
        total_loss = mae_loss + self.omega_mse * mse_loss
        return total_loss

class WetMAEWithSoftCSILoss(nn.Module):
    def __init__(self, alpha_softcsi=1.0, threshold=1e-3, eps=1e-6, smooth_factor=1e-2):
        super(WetMAEWithSoftCSILoss, self).__init__()
        self.alpha_softcsi = alpha_softcsi
        self.threshold = threshold
        self.eps = eps
        self.smooth_factor = smooth_factor

    def forward(self, y_pred, y_true):
        # --- Wet MAE ---
        wet_mask = (y_true >= self.threshold).float()
        N_wet = torch.sum(wet_mask) + self.eps
        abs_err = torch.abs(y_true - y_pred)
        wet_loss = torch.sum(abs_err * wet_mask) / N_wet

        # --- SoftCSI ---
        y_pred_flat = y_pred.reshape(-1)
        y_true_flat = y_true.reshape(-1)
        wet_true = torch.sigmoid((y_true_flat - self.threshold) / self.smooth_factor)
        wet_pred = torch.sigmoid((y_pred_flat - self.threshold) / self.smooth_factor)
        intersection = torch.sum(wet_true * wet_pred)
        union = torch.sum(wet_true) + torch.sum(wet_pred) - intersection
        soft_csi = intersection / (union + self.eps)
        softcsi_loss = 1.0 - soft_csi

        total_loss = wet_loss + self.alpha_softcsi * softcsi_loss
        return total_loss

    
def set_seed(seed):
    random.seed(seed) # Python
    np.random.seed(seed) # Numpy
    torch.manual_seed(seed) # PyTorch CPU
    torch.cuda.manual_seed(seed) # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # PyTorch multi-GPU
    torch.backends.cudnn.deterministic = True # cuDNN deterministic ops
    torch.backends.cudnn.benchmark = False # # si True, hace que cuDNN busque automáticamente la implementación más rápida para las convoluciones, pero esas elecciones no son deterministas
    torch.use_deterministic_algorithms(True) # Force PyTorch to use deterministic algorithms
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # Needed for deterministic cublas (optional)

def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

def train_model(
    model, optimizer, criterion, 
    X_train, y_train, X_val, y_val, 
    z_fine_train, z_fine_val,
    total_pixels_to_crop_on_each_dim,
    error_threshold=0.001,
    clip_value_min=None, clip_value_max=None,
    log1p_exp1m_transform=False,
    epochs=50, batch_size=32, early_stopping_patience=10,
    model_save_path='best_unet_model_pytorch.pth',
    device='cpu',
    y_min=None, y_max=None,
    MASK_FINE_MAP_TO_THRESHOLD=False,
    world_size=1, rank=0, SEED=42
):
    """
    Trains the U-Net model with learning rate scheduling and early stopping.
    """
    # Mantener los datos en CPU; moveremos cada batch individualmente al device
    X_train_coarse_tensor = X_train[0].cpu()
    X_train_topo_tensor = X_train[1].cpu()
    y_train_tensor = y_train.cpu()
    z_fine_train_tensor = z_fine_train.cpu()
    X_val_coarse_tensor = X_val[0].cpu()
    X_val_topo_tensor = X_val[1].cpu()
    y_val_tensor = y_val.cpu()
    z_fine_val_tensor = z_fine_val.cpu()

    if y_min is not None and y_max is not None:
        y_min = y_min.to(y_train_tensor.device)
        y_max = y_max.to(y_train_tensor.device)

    if MASK_FINE_MAP_TO_THRESHOLD:
        assert y_min is not None and y_max is not None, "Se requiere y_min y y_max para aplicar el umbral en datos normalizados."
        norm_threshold = (error_threshold - y_min) / (y_max - y_min + 1e-8) # convertimos el error_threshold al espacio normalizado
        y_train_tensor = torch.where(y_train_tensor < norm_threshold, torch.tensor(0.0, dtype=y_train_tensor.dtype), y_train_tensor)

    g = torch.Generator()
    g.manual_seed(SEED)

    # Crear Datasets y DataLoaders
    train_dataset = torch.utils.data.TensorDataset(X_train_coarse_tensor, X_train_topo_tensor, y_train_tensor, z_fine_train_tensor)
    # train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) #! no lo usamos porque dividimos datos en el main
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=2, pin_memory=True, worker_init_fn=seed_worker, generator=g, persistent_workers=True, prefetch_factor=2)

    val_dataset = torch.utils.data.TensorDataset(X_val_coarse_tensor, X_val_topo_tensor, y_val_tensor, z_fine_val_tensor)
    # val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) #! no lo usamos porque dividimos datos en el main
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=2, pin_memory=True, worker_init_fn=seed_worker, generator=g, persistent_workers=True, prefetch_factor=2)
    
    # Para monitorizar y ver evolución de las métricas
    best_val_loss = torch.tensor(float('inf')).to(device)
    patience_counter = torch.tensor(0).to(device)
    history = {
        'loss_sync': [], 'val_loss_sync': [],
        # 'mse_sync': [], 'val_mse_sync': [],
        # 'mae_sync': [], 'val_mae_sync': [],
        # 'val_denorm_loss_sync': [],
        # 'val_denorm_mse_sync': [],
        'val_denorm_mae_sync': [],
        'lr': []
    }

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=5, min_lr=1e-10)

    start_time = time.time()
    if rank == 0:
        print("🚀 Starting training...")
        print(f"📊 Training data (rank {rank}): {len(train_dataset)} samples")
        print(f"📊 Validation data (rank {rank}): {len(val_dataset)} samples")
        print(f"⚙️  Batch size: {batch_size}")
        print(f"🔄 Max epochs: {epochs}")
        print(f"📉 Initial learning rate: {optimizer.param_groups[0]['lr']:.9f}")
        print(f"Using device (rank {rank}): {device}")
        print("-" * 50)

    for epoch in range(epochs):

        model.train() #! training
        running_loss = 0.0
        # running_mse = 0.0
        # running_mae = 0.0

        for batch_idx, (coarse_batch, topo_batch, fine_batch, z_fine_batch) in enumerate(train_loader):
            coarse_batch = coarse_batch.to(device, non_blocking=True)
            topo_batch = topo_batch.to(device, non_blocking=True)
            fine_batch = fine_batch.to(device, non_blocking=True)
            z_fine_batch = z_fine_batch.to(device, non_blocking=True)
    
            optimizer.zero_grad()
            outputs = model(coarse_batch, topo_batch)

            # Crop central de fine_batch y z_fine_batch para que coincida con dimensiones de outputs (de normal ya coinciden, es en caso de que haya una división //2 no entera)
            H_pred, W_pred = outputs.shape[1:3]
            H_true_y, W_true_y = fine_batch.shape[1:3]
            H_true_z, W_true_z = z_fine_batch.shape[1:3]
            crop_h_y = (H_true_y - H_pred) // 2
            crop_w_y = (W_true_y - W_pred) // 2
            crop_h_z = (H_true_z - H_pred) // 2
            crop_w_z = (W_true_z - W_pred) // 2
            fine_batch = fine_batch[:, crop_h_y:H_true_y - crop_h_y, crop_w_y:W_true_y - crop_w_y, :]
            z_fine_batch = z_fine_batch[:, crop_h_z:H_true_z - crop_h_z, crop_w_z:W_true_z - crop_w_z, :]

            # Normalized metrics (for training it is enough, we do not care about denormalized ones until validation)
            loss = criterion(outputs, fine_batch)
            # mse_batch = F.mse_loss(outputs, fine_batch)
            # mae_batch = F.l1_loss(outputs, fine_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #! we clip the gradient to avoid it from exploding
            optimizer.step()

            running_loss += loss.item()
            # running_mse += mse_batch
            # running_mae += mae_batch

        avg_train_loss = running_loss / len(train_loader)
        # avg_train_mse = running_mse / len(train_loader)
        # avg_train_mae = running_mae / len(train_loader)

        model.eval() #! validation

        val_running_loss = 0.0
        val_running_denorm_loss = 0.0
        # val_running_denorm_mse = 0.0
        val_running_denorm_mae = 0.0

        with torch.no_grad():
            for batch_idx, (coarse_batch_val, topo_batch_val, fine_batch_val, z_fine_batch_val) in enumerate(val_loader):
                coarse_batch_val = coarse_batch_val.to(device, non_blocking=True)
                topo_batch_val = topo_batch_val.to(device, non_blocking=True)
                fine_batch_val = fine_batch_val.to(device, non_blocking=True)
                z_fine_batch_val = z_fine_batch_val.to(device, non_blocking=True)

                val_outputs = model(coarse_batch_val, topo_batch_val)

                # Crop central de fine_batch_val y z_fine_batch_val para que coincida con dimensiones de val_outputs (de normal ya coinciden, es en caso de que haya una división //2 no entera)
                H_pred, W_pred = val_outputs.shape[1:3]
                H_true_y, W_true_y = fine_batch_val.shape[1:3]
                H_true_z, W_true_z = z_fine_batch_val.shape[1:3]
                crop_h_y = (H_true_y - H_pred) // 2
                crop_w_y = (W_true_y - W_pred) // 2
                crop_h_z = (H_true_z - H_pred) // 2
                crop_w_z = (W_true_z - W_pred) // 2
                fine_batch_val = fine_batch_val[:, crop_h_y:H_true_y - crop_h_y, crop_w_y:W_true_y - crop_w_y, :]
                z_fine_batch_val = z_fine_batch_val[:, crop_h_z:H_true_z - crop_h_z, crop_w_z:W_true_z - crop_w_z, :]

                val_loss = criterion(val_outputs, fine_batch_val)

                # Denormalized metrics, and inverse transform to original scale and clip negatives to zero (optional)
                val_outputs_denorm = val_outputs
                fine_batch_val_denorm = fine_batch_val
                if y_min is not None and y_max is not None:
                    y_min = y_min.to(val_outputs.device)
                    y_max = y_max.to(val_outputs.device)
                    val_outputs_denorm = val_outputs * (y_max - y_min + 1e-8) + y_min
                    fine_batch_val_denorm = fine_batch_val * (y_max - y_min + 1e-8) + y_min

                if log1p_exp1m_transform:
                    val_outputs_denorm = torch.expm1(val_outputs_denorm)
                    fine_batch_val_denorm = torch.expm1(fine_batch_val_denorm)

                if clip_value_min is not None:
                    val_outputs_denorm = torch.clamp(val_outputs_denorm, min=clip_value_min)
                if clip_value_max is not None:
                    val_outputs_denorm = torch.clamp(val_outputs_denorm, max=clip_value_max)

                val_denorm_loss = criterion(val_outputs_denorm, fine_batch_val_denorm)
                # val_denorm_mse = F.mse_loss(val_outputs_denorm, fine_batch_val_denorm).item()
                val_denorm_mae = F.l1_loss(val_outputs_denorm, fine_batch_val_denorm).item()

                val_running_loss += val_loss.item()
                val_running_denorm_loss += val_denorm_loss.item()
                # val_running_denorm_mse += val_denorm_mse
                val_running_denorm_mae += val_denorm_mae
                

        avg_val_loss = val_running_loss / len(val_loader)
        avg_val_denorm_loss = val_running_denorm_loss / len(val_loader)
        # avg_val_denorm_mse = val_running_denorm_mse / len(val_loader)
        avg_val_denorm_mae = val_running_denorm_mae / len(val_loader)

        #! === SINCRONIZAR entre todos los ranks ===
        # Train
        avg_train_loss_tensor = torch.tensor(avg_train_loss).to(device)
        dist.all_reduce(avg_train_loss_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss_sync = avg_train_loss_tensor.item() / world_size

        # avg_train_denorm_mae_tensor = torch.tensor(avg_train_denorm_mae).to(device)
        # dist.all_reduce(avg_train_denorm_mae_tensor, op=dist.ReduceOp.SUM)
        # avg_train_denorm_mae_sync = avg_train_denorm_mae_tensor.item() / world_size

        # avg_train_denorm_mse_tensor = torch.tensor(avg_train_denorm_mse).to(device)
        # dist.all_reduce(avg_train_denorm_mse_tensor, op=dist.ReduceOp.SUM)
        # avg_train_denorm_mse_sync = avg_train_denorm_mse_tensor.item() / world_size

        # Validation
        avg_val_loss_tensor = torch.tensor(avg_val_loss).to(device)
        dist.all_reduce(avg_val_loss_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss_sync = avg_val_loss_tensor.item() / world_size

        avg_val_denorm_mae_tensor = torch.tensor(avg_val_denorm_mae).to(device)
        dist.all_reduce(avg_val_denorm_mae_tensor, op=dist.ReduceOp.SUM)
        avg_val_denorm_mae_sync = avg_val_denorm_mae_tensor.item() / world_size

        # avg_val_denorm_mse_tensor = torch.tensor(avg_val_denorm_mse).to(device)
        # dist.all_reduce(avg_val_denorm_mse_tensor, op=dist.ReduceOp.SUM)
        # avg_val_denorm_mse_sync = avg_val_denorm_mse_tensor.item() / world_size

        # Hacemos el scheduler con la avg_val_loss_sync para que se base en el rendimiento global del modelo
        scheduler.step(avg_val_loss_sync)

        #! === EARLY STOPPING sincronizado ===
        if avg_val_loss_sync < best_val_loss.item():
            best_val_loss = torch.tensor(avg_val_loss_sync).to(device)
            patience_counter = torch.tensor(0).to(device)
            
            if rank == 0: # solo rank 0 guarda el modelo
                torch.save(model.module.state_dict(), model_save_path)
                print(f"Epoch {epoch+1}/{epochs}: val_loss improved to {best_val_loss.item():.9f}, saving model.")
        else:
            patience_counter += 1
            if patience_counter.item() >= early_stopping_patience:
                if rank == 0:
                    print(f"⏹️ Early stopping triggered at epoch {epoch+1}/{epochs}")
                break
        
        #! === LOGGING y guardar HISTORY sincronizado ===
        if rank == 0:
            # now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            # print(f"📅 Current date/time: {now_str}")
            print(f'Epoch {epoch+1}/{epochs}, Sync Val Loss: {avg_val_loss_sync:.9f}, '
                  f'Best Val Loss: {best_val_loss.item():.9f}, '
                  f'Sync Val Denorm MAE: {avg_val_denorm_mae_sync:.9f}, '
                  f'Patience: {patience_counter.item()}/{early_stopping_patience}, '
                  f'lr: {optimizer.param_groups[0]["lr"]:.9f}')

        history['loss_sync'].append(avg_train_loss_sync)
        # history['mse_sync'].append(avg_train_mse_sync)
        # history['mae_sync'].append(avg_train_mae_sync)

        history['val_loss_sync'].append(avg_val_loss_sync)
        # history['val_denorm_loss_sync'].append(avg_val_denorm_loss_sync)
        # history['val_denorm_mse_sync'].append(avg_val_denorm_mse_sync)
        history['val_denorm_mae_sync'].append(avg_val_denorm_mae_sync)

        history['lr'].append(optimizer.param_groups[0]['lr'])

        dist.barrier()

    if rank == 0:
        print(f"✅ Training completed (rank {rank})")
        end_time = time.time()
        elapsed = end_time - start_time
        print("-" * 50)
        print(f"⏱️  Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print(f"⏱️  Training ended at:   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"⏱️  Total training time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    return history

def plot_training_history(history, save_path='results/training_history'):
    """
    Visualizes and saves the training history and plots for PyTorch.

    Saves:
    - A PNG plot of the training curves.
    - A PKL file with the `history` dictionary.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PyTorch U-Net Training History', fontsize=16)

    # Loss (denormalized)
    axes[0, 0].plot(history['loss_sync'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss_sync'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # MSE (denormalized)
    axes[0, 1].axis('off')
    # axes[0, 1].plot(history['denorm_mse_sync'], label='Training MSE', linewidth=2)
    # axes[0, 1].plot(history['val_denorm_mse_sync'], label='Validation MSE', linewidth=2)
    # axes[0, 1].set_title('Mean Squared Error')
    # axes[0, 1].set_xlabel('Epoch')
    # axes[0, 1].set_ylabel('MSE')
    # axes[0, 1].set_yscale('log')
    # axes[0, 1].legend()
    # axes[0, 1].grid(True, alpha=0.3)

    # MAE (denormalized)
    # axes[1, 0].plot(history['denorm_mae_sync'], label='Training MAE', linewidth=2)
    axes[1, 0].plot(history['val_denorm_mae_sync'], label='Validation MAE', linewidth=2)
    axes[1, 0].set_title('Mean Absolute Error')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning Rate
    axes[1, 1].plot(history['lr'], linewidth=2, color='red')
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    fig_path = save_path + '_history_plot.png'
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved training plot to: {fig_path}")

    # Save training history as PKL
    history_path = save_path + '_history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Saved training history to: {history_path}")

    best_epoch = int(np.argmin(history['val_loss_sync']) + 1)
    best_val_loss = min(history['val_loss_sync'])
    print(f"\nBest epoch: {best_epoch} (val_loss_sync: {best_val_loss:.9f})")

def evaluate_model(model, criterion, X_test, y_test, z_fine_test, # z_fine_test no se usa
                           log1p_exp1m_transform=False,
                           normalization_parameters=None,
                           error_threshold=0.001,
                           device='cpu',
                           clip_value_min=None, clip_value_max=None,
                           grouping_parameter_for_subgrid_rows=1,
                           grouping_parameter_for_subgrid_cols=1,
                           save_txt_with_metrics=False,
                           save_predictions=False,
                           predictions_save_path=None,
                           PATCH_BASED_MAPPING=False,
                           NUMBER_OF_SUBDOMAINS=None,
                           measure_unet_cost=False,
                           evaluation_mask=None,
                           evaluation_mask_path=None):
    """
    Evalúa el modelo y benchmarks. Devuelve SUMAS y CONTADORES para un all_reduce correcto.
    """
    model.eval()
    rank = dist.get_rank() if dist.is_initialized() else 0
    if evaluation_mask is not None:
        if evaluation_mask.ndim != 2:
            raise ValueError(
                f"La máscara geográfica debe ser 2D; se recibió shape {tuple(evaluation_mask.shape)}"
            )
        if not bool(np.isfinite(evaluation_mask).all()):
            raise ValueError("La máscara geográfica contiene valores no booleanos o no finitos.")
        evaluation_mask = np.asarray(evaluation_mask, dtype=bool)
        if evaluation_mask.sum() == 0:
            raise ValueError("La máscara geográfica está vacía.")
    if evaluation_mask is not None and PATCH_BASED_MAPPING:
        raise ValueError(
            "PATCH_BASED_MAPPING=True no admite una máscara geográfica completa: "
            "los parches no tienen una partición inequívoca identificable."
        )

    # --- Preparación de Datos y DataLoader ---
    X_test_coarse_tensor, X_test_topo_tensor = X_test[0].to(device), X_test[1].to(device)
    y_test_tensor = y_test.to(device)

    test_dataset = torch.utils.data.TensorDataset(X_test_coarse_tensor, X_test_topo_tensor, y_test_tensor)
    
    is_distributed = dist.is_initialized() and dist.get_world_size() > 1
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False) if is_distributed else None
    # test_sampler = None #! no lo usamos porque dividimos datos en el main
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

    # --- Inicialización de Acumuladores ---
    loss_sum_model, sum_sq_err_model, sum_abs_err_model, count_total_model = 0.0, 0.0, 0.0, 0
    sum_sq_err_masked_model, sum_abs_err_masked_model, count_masked_model = 0.0, 0.0, 0
    vol_rel_error_sum_model, valid_vol_count_model = 0.0, 0
    TP_model, FP_model, FN_model = 0, 0, 0
    
    sum_sq_err_subgrid, sum_abs_err_subgrid = 0.0, 0.0
    sum_sq_err_masked_subgrid, sum_abs_err_masked_subgrid, count_masked_subgrid = 0.0, 0.0, 0
    vol_rel_error_sum_subgrid, valid_vol_count_subgrid = 0.0, 0
    TP_subgrid, FP_subgrid, FN_subgrid = 0, 0, 0

    sum_sq_err_bicubic, sum_abs_err_bicubic = 0.0, 0.0
    sum_sq_err_masked_bicubic, sum_abs_err_masked_bicubic, count_masked_bicubic = 0.0, 0.0, 0
    vol_rel_error_sum_bicubic, valid_vol_count_bicubic = 0.0, 0
    TP_bicubic, FP_bicubic, FN_bicubic = 0, 0, 0

    if save_predictions and predictions_save_path is not None:
        rank_save_path = os.path.join(predictions_save_path, f"test_set_results/rank_{rank}")
        os.makedirs(rank_save_path, exist_ok=True)
    
    # Archivo TXT para métricas detalladas por imagen
    metrics_file = None 
    if save_txt_with_metrics and predictions_save_path is not None and rank == 0:
        txt_path = predictions_save_path + "_detailed_metrics_per_image.txt"
        os.makedirs(os.path.dirname(txt_path), exist_ok=True) # asegurarse de que el directorio existe
        metrics_file = open(txt_path, "w")
        # Cabecera
        if evaluation_mask is not None:
            metrics_file.write("# Dominio de evaluación: máscara geográfica\n")
            if evaluation_mask_path:
                metrics_file.write(f"# DEM de evaluación: {evaluation_mask_path}\n")
            metrics_file.write(
                f"# Mask shape: {evaluation_mask.shape}; valid cells per field: {int(evaluation_mask.sum())}\n"
            )
        else:
            metrics_file.write("# Dominio de evaluación: rectangular\n")
        if PATCH_BASED_MAPPING and NUMBER_OF_SUBDOMAINS is not None:
            metrics_file.write(f"# Note: The test set was divided into {NUMBER_OF_SUBDOMAINS} subdomains for patch-based processing.\n")
        metrics_file.write(f"# {'Time step':<10} {'MAE':<15} {'CSI':<15} {'TP':<10} {'TN':<10} {'FP':<10} {'FN':<10}\n")

    subgrid_solver, bicubic_resizer = None, None
    global_sample_index = 0
    mae_accum = 0.0
    tp_accum, tn_accum, fp_accum, fn_accum = 0, 0, 0, 0
    samples_in_group = 0

    if measure_unet_cost:
        torch.cuda.synchronize(device)
        baseline_memory = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        inference_time = 0.0

    with torch.no_grad():
        for batch_idx, (coarse_batch, topo_batch_orig, fine_batch_orig) in enumerate(test_loader):
            if measure_unet_cost:
                torch.cuda.synchronize(device)
                start_time = time.perf_counter()

            outputs = model(coarse_batch, topo_batch_orig)

            if measure_unet_cost:
                torch.cuda.synchronize(device)
                inference_time += time.perf_counter() - start_time
                if batch_idx == len(test_loader) - 1:
                    peak_memory_allocated = torch.cuda.max_memory_allocated(device)
                    peak_memory_reserved = torch.cuda.max_memory_reserved(device)
                    additional_peak_memory = peak_memory_allocated - baseline_memory

            fine_batch = fine_batch_orig.clone()
            
            coarse_batch_processed = coarse_batch.clone()
            topo_batch_processed = topo_batch_orig.clone()

            if normalization_parameters is not None:
                Xc_min, Xc_max, Xt_min, Xt_max, y_min, y_max, z_min, z_max = normalization_parameters
                Xc_min, Xc_max = Xc_min.to(coarse_batch_processed.device), Xc_max.to(coarse_batch_processed.device)
                Xt_min, Xt_max = Xt_min.to(topo_batch_processed.device), Xt_max.to(topo_batch_processed.device)
                y_min, y_max = y_min.to(fine_batch.device), y_max.to(fine_batch.device)
                
                outputs = outputs * (y_max - y_min + 1e-8) + y_min
                fine_batch = fine_batch * (y_max - y_min + 1e-8) + y_min
                coarse_batch_processed = coarse_batch_processed * (Xc_max - Xc_min + 1e-8) + Xc_min
                topo_batch_processed = topo_batch_processed * (Xt_max - Xt_min + 1e-8) + Xt_min

            if log1p_exp1m_transform:
                outputs = torch.expm1(outputs)
                fine_batch = torch.expm1(fine_batch)
                coarse_batch_processed = torch.expm1(coarse_batch_processed)

            H_true, W_true = fine_batch.shape[1:3]
            if subgrid_solver is None:
                subgrid_solver = ResizeWithClassicalSubGrid(H_true, W_true).to(device)
                bicubic_resizer = ResizeWithBicubic(H_true, W_true).to(device)
            
            subgrid_outputs = subgrid_solver(coarse_batch_processed, topo_batch_processed[:,:,:,:1], group_size_rows=grouping_parameter_for_subgrid_rows, group_size_cols=grouping_parameter_for_subgrid_cols)
            bicubic_outputs = bicubic_resizer(coarse_batch_processed)

            if clip_value_min is not None: outputs = torch.clamp(outputs, min=clip_value_min)
            if clip_value_max is not None: outputs = torch.clamp(outputs, max=clip_value_max)
            
            H_pred, W_pred = outputs.shape[1:3]
            crop_h = (H_true - H_pred) // 2
            crop_w = (W_true - W_pred) // 2
            
            h_true = fine_batch[:, crop_h:crop_h + H_pred, crop_w:crop_w + W_pred, :]
            h_pred_model = outputs
            h_pred_subgrid = subgrid_outputs[:, crop_h:crop_h + H_pred, crop_w:crop_w + W_pred, :]
            h_pred_bicubic = bicubic_outputs[:, crop_h:crop_h + H_pred, crop_w:crop_w + W_pred, :]

            if evaluation_mask is not None:
                if evaluation_mask.shape != fine_batch.shape[1:3]:
                    raise ValueError(
                        f"Tamaño incompatible de máscara geográfica: máscara {evaluation_mask.shape}, "
                        f"referencia completa {fine_batch.shape[1:3]}"
                    )
                cropped_mask = evaluation_mask[
                    crop_h:crop_h + H_pred,
                    crop_w:crop_w + W_pred,
                ]
                if cropped_mask.shape != h_true.shape[1:3]:
                    raise ValueError(
                        f"Tamaño incompatible tras recortar la máscara: máscara "
                        f"{cropped_mask.shape}, referencia/reconstrucciones {h_true.shape[1:3]}"
                    )
                valid_mask = torch.as_tensor(
                    cropped_mask, dtype=torch.bool, device=device
                ).unsqueeze(0).unsqueeze(-1).expand(h_true.shape)
                if (
                    not torch.isfinite(h_true[valid_mask]).all()
                    or not torch.isfinite(h_pred_model[valid_mask]).all()
                    or not torch.isfinite(h_pred_subgrid[valid_mask]).all()
                    or not torch.isfinite(h_pred_bicubic[valid_mask]).all()
                ):
                    raise ValueError(
                        "NaN o infinito encontrado dentro del dominio geográfico válido "
                        "en la referencia o en una reconstrucción."
                    )
                domain_mask = valid_mask
                num_pixels = int(domain_mask.sum().item())
            else:
                domain_mask = torch.ones_like(h_true, dtype=torch.bool)
                if (
                    not torch.isfinite(h_true).all()
                    or not torch.isfinite(h_pred_model).all()
                    or not torch.isfinite(h_pred_subgrid).all()
                    or not torch.isfinite(h_pred_bicubic).all()
                ):
                    raise ValueError(
                        "NaN o infinito encontrado en la evaluación rectangular "
                        "en la referencia o en una reconstrucción."
                    )
                num_pixels = h_true.numel()

            loss_sum_model += criterion(h_pred_model, h_true).item()
            count_total_model += num_pixels
            diff_model = h_pred_model - h_true
            diff_subgrid = h_pred_subgrid - h_true
            diff_bicubic = h_pred_bicubic - h_true
            sum_sq_err_model += (diff_model[domain_mask] ** 2).sum().item()
            sum_abs_err_model += torch.abs(diff_model[domain_mask]).sum().item()
            sum_sq_err_subgrid += (diff_subgrid[domain_mask] ** 2).sum().item()
            sum_abs_err_subgrid += torch.abs(diff_subgrid[domain_mask]).sum().item()
            sum_sq_err_bicubic += (diff_bicubic[domain_mask] ** 2).sum().item()
            sum_abs_err_bicubic += torch.abs(diff_bicubic[domain_mask]).sum().item()
            
            if metrics_file is not None:
                batch_size_curr = h_true.shape[0]
                
                for i in range(batch_size_curr):
                    img_true = h_true[i]
                    img_pred = h_pred_model[i]
                    
                    img_domain = domain_mask[i, ..., 0]
                    if evaluation_mask is not None:
                        mae_i = torch.abs(img_pred - img_true)[img_domain].mean().item()
                    else:
                        mae_i = torch.mean(torch.abs(img_pred - img_true)).item()
                    
                    bin_true = ((img_true >= error_threshold) & img_domain.unsqueeze(-1)).int()
                    bin_pred = ((img_pred >= error_threshold) & img_domain.unsqueeze(-1)).int()
                    
                    valid_i = img_domain.unsqueeze(-1)
                    tp_i = (valid_i & (bin_true == 1) & (bin_pred == 1)).sum().item()
                    tn_i = (valid_i & (bin_true == 0) & (bin_pred == 0)).sum().item()
                    fp_i = (valid_i & (bin_true == 0) & (bin_pred == 1)).sum().item()
                    fn_i = (valid_i & (bin_true == 1) & (bin_pred == 0)).sum().item()

                    # --- Lógica de Agrupación (Patch-Based) ---
                    if PATCH_BASED_MAPPING and NUMBER_OF_SUBDOMAINS is not None:
                        # Inicializar acumuladores si es el primer elemento del grupo
                        if 'mae_accum' not in locals():
                            mae_accum = 0.0
                            tp_accum, tn_accum, fp_accum, fn_accum = 0, 0, 0, 0
                            samples_in_group = 0

                        # Acumular
                        mae_accum += mae_i
                        tp_accum += tp_i
                        tn_accum += tn_i
                        fp_accum += fp_i
                        fn_accum += fn_i
                        samples_in_group += 1

                        # Comprobar si hemos completado un grupo (ej. 42 imágenes)
                        # Usamos global_sample_index + 1 porque el índice es 0-based
                        if (global_sample_index + 1) % NUMBER_OF_SUBDOMAINS == 0:
                            # Calcular medias y métricas del grupo
                            mae_group = mae_accum / samples_in_group
                            
                            denom_group = tp_accum + fp_accum + fn_accum
                            csi_group = (tp_accum / denom_group) if denom_group > 0 else 1.0
                            if denom_group == 0 and tn_accum > 0: csi_group = 1.0

                            # Escribir la línea del GRUPO (Time Step completo)
                            # El índice mostrado será el bloque (0, 1, 2...) no la imagen
                            group_idx = (global_sample_index + 1) // NUMBER_OF_SUBDOMAINS
                            metrics_file.write(f"{group_idx:<10} {mae_group:<15.6f} {csi_group:<15.4f} {tp_accum:<10} {tn_accum:<10} {fp_accum:<10} {fn_accum:<10}\n")
                            
                            # Reiniciar acumuladores (borrando la variable o reseteando)
                            del mae_accum 
                    
                    # --- Lógica Estándar (Por Imagen Individual) ---
                    else:
                        denominator = tp_i + fp_i + fn_i
                        csi_i = (tp_i / denominator) if denominator > 0 else 1.0
                        if denominator == 0 and tn_i > 0: csi_i = 1.0 
                        
                        metrics_file.write(f"{(global_sample_index+1):<10} {mae_i:<15.6f} {csi_i:<15.4f} {tp_i:<10} {tn_i:<10} {fp_i:<10} {fn_i:<10}\n")
                    
                    global_sample_index += 1

            wet_domain = domain_mask & (h_true > error_threshold)
            mask = wet_domain
            if mask.sum() > 0:
                masked_pixel_count = mask.sum().item()
                count_masked_model += masked_pixel_count
                sum_sq_err_masked_model += ((h_pred_model - h_true)[mask] ** 2).sum().item()
                sum_abs_err_masked_model += torch.abs((h_pred_model - h_true)[mask]).sum().item()
                count_masked_subgrid += masked_pixel_count
                sum_sq_err_masked_subgrid += ((h_pred_subgrid - h_true)[mask] ** 2).sum().item()
                sum_abs_err_masked_subgrid += torch.abs((h_pred_subgrid - h_true)[mask]).sum().item()
                count_masked_bicubic += masked_pixel_count
                sum_sq_err_masked_bicubic += ((h_pred_bicubic - h_true)[mask] ** 2).sum().item()
                sum_abs_err_masked_bicubic += torch.abs((h_pred_bicubic - h_true)[mask]).sum().item()
            
            vol_true = torch.where(
                domain_mask,
                h_true,
                torch.zeros_like(h_true),
            ).sum(dim=[1, 2, 3])
            vol_mask = vol_true > 0
            y_true_binary = ((h_true >= error_threshold) & domain_mask).int()
            
            vol_pred_model = torch.where(domain_mask, h_pred_model, torch.zeros_like(h_pred_model)).sum(dim=[1, 2, 3])
            if vol_mask.sum() > 0:
                vol_rel_error_sum_model += (torch.abs(vol_true[vol_mask] - vol_pred_model[vol_mask]) / vol_true[vol_mask]).sum().item()
                valid_vol_count_model += vol_mask.sum().item()
            y_pred_binary_model = ((h_pred_model >= error_threshold) & domain_mask).int()
            TP_model += int(((y_true_binary == 1) & (y_pred_binary_model == 1)).sum().item())
            FP_model += int(((y_true_binary == 0) & (y_pred_binary_model == 1)).sum().item())
            FN_model += int(((y_true_binary == 1) & (y_pred_binary_model == 0)).sum().item())

            vol_pred_subgrid = torch.where(domain_mask, h_pred_subgrid, torch.zeros_like(h_pred_subgrid)).sum(dim=[1, 2, 3])
            if vol_mask.sum() > 0:
                vol_rel_error_sum_subgrid += (torch.abs(vol_true[vol_mask] - vol_pred_subgrid[vol_mask]) / vol_true[vol_mask]).sum().item()
                valid_vol_count_subgrid += vol_mask.sum().item()
            y_pred_binary_subgrid = ((h_pred_subgrid >= error_threshold) & domain_mask).int()
            TP_subgrid += int(((y_true_binary == 1) & (y_pred_binary_subgrid == 1)).sum().item())
            FP_subgrid += int(((y_true_binary == 0) & (y_pred_binary_subgrid == 1)).sum().item())
            FN_subgrid += int(((y_true_binary == 1) & (y_pred_binary_subgrid == 0)).sum().item())
            
            vol_pred_bicubic = torch.where(domain_mask, h_pred_bicubic, torch.zeros_like(h_pred_bicubic)).sum(dim=[1, 2, 3])
            if vol_mask.sum() > 0:
                vol_rel_error_sum_bicubic += (torch.abs(vol_true[vol_mask] - vol_pred_bicubic[vol_mask]) / vol_true[vol_mask]).sum().item()
                valid_vol_count_bicubic += vol_mask.sum().item()
            y_pred_binary_bicubic = ((h_pred_bicubic >= error_threshold) & domain_mask).int()
            TP_bicubic += int(((y_true_binary == 1) & (y_pred_binary_bicubic == 1)).sum().item())
            FP_bicubic += int(((y_true_binary == 0) & (y_pred_binary_bicubic == 1)).sum().item())
            FN_bicubic += int(((y_true_binary == 1) & (y_pred_binary_bicubic == 0)).sum().item())

    if metrics_file is not None:
        metrics_file.close()
        print(f"Métricas detalladas por imagen guardadas en: {txt_path}")

    if measure_unet_cost:
        number_of_real_case_samples = len(y_test_tensor)
        number_of_forward_batches = len(test_loader)
        print("=" * 64)
        print("UNET REAL-CASE COMPUTATIONAL COST")
        print("=" * 64)
        print(f"Samples: {number_of_real_case_samples}")
        print(f"Batch size: {test_loader.batch_size}")
        print(f"Number of forward batches: {number_of_forward_batches}")
        print(f"UNet inference time: {inference_time:.6f} s")
        print(f"Mean inference time per sample: {inference_time / number_of_real_case_samples:.6f} s")
        print(f"Baseline GPU memory allocated: {baseline_memory / (1024 ** 3):.3f} GB")
        print(f"Peak GPU memory allocated: {peak_memory_allocated / (1024 ** 3):.3f} GB")
        print(f"Additional peak GPU memory: {additional_peak_memory / (1024 ** 3):.3f} GB")
        print(f"Peak GPU memory reserved: {peak_memory_reserved / (1024 ** 3):.3f} GB")
        print("=" * 64)

    model_metrics = (loss_sum_model, sum_sq_err_model, sum_abs_err_model, float(count_total_model),
                     sum_sq_err_masked_model, sum_abs_err_masked_model, float(count_masked_model),
                     vol_rel_error_sum_model, float(valid_vol_count_model),
                     float(TP_model), float(FP_model), float(FN_model))
    
    subgrid_metrics = (sum_sq_err_subgrid, sum_abs_err_subgrid, float(count_total_model),
                       sum_sq_err_masked_subgrid, sum_abs_err_masked_subgrid, float(count_masked_subgrid),
                       vol_rel_error_sum_subgrid, float(valid_vol_count_subgrid),
                       float(TP_subgrid), float(FP_subgrid), float(FN_subgrid))
    
    bicubic_metrics = (sum_sq_err_bicubic, sum_abs_err_bicubic, float(count_total_model),
                       sum_sq_err_masked_bicubic, sum_abs_err_masked_bicubic, float(count_masked_bicubic),
                       vol_rel_error_sum_bicubic, float(valid_vol_count_bicubic),
                       float(TP_bicubic), float(FP_bicubic), float(FN_bicubic))

    return model_metrics, subgrid_metrics, bicubic_metrics

def compute_metrics(vec, *, has_loss: bool, n_batches: float, threshold: float):
    """
    Docstring for compute_metrics
    
    :param vec: metrics vector with sums and counts needed to compute final metrics
    :param has_loss: boolean indicating if the first element of vec is the sum of losses (if False, vec should not contain loss_sum and n_batches will be ignored)
    :param n_batches: number of batches used to compute the average loss (only needed if has_loss is True)
    :param threshold: value used for binary wet/dry classification metrics (HIT, FAR, CSI)
    :return: dictionary with computed metrics (MSE, MAE, Masked MSE, Masked MAE, Volume Relative Error, HIT, FAR, CSI)

    sse: sum of squared errors (for MSE)
    sae: sum of absolute errors (for MAE)
    n_total: total number of pixels (for average MSE and MAE)
    sse_m: sum of squared errors for masked pixels (where true value > threshold)
    sae_m: sum of absolute errors for masked pixels
    n_m: number of masked pixels (where true value > threshold)
    vrel_sum: sum of relative volume errors (|vol_true - vol_pred| / vol_true) for samples where vol_true > 0
    vrel_n: count of samples where vol_true > 0 (for average volume relative error)
    TP: true positives (for HIT, FAR, CSI)
    FP: false positives (for HIT, FAR, CSI)
    FN: false negatives (for HIT, FAR, CSI)
    """
    vals = vec.tolist()
    if has_loss:
        (loss_sum, sse, sae, n_total,
         sse_m, sae_m, n_m,
         vrel_sum, vrel_n,
         TP, FP, FN) = vals
    else:
        (sse, sae, n_total,
         sse_m, sae_m, n_m,
         vrel_sum, vrel_n,
         TP, FP, FN) = vals
        loss_sum = None

    n_total = max(1.0, float(n_total))
    n_m = max(1.0, float(n_m))
    vrel_n = max(1.0, float(vrel_n))

    out = {
        "MSE": sse / n_total,
        "MAE": sae / n_total,
        "Masked MSE": sse_m / n_m,
        "Masked MAE": sae_m / n_m,
        "Volume Rel Err (%)": (vrel_sum / vrel_n) * 100.0,
    }

    if has_loss:
        out["Diagnostic Loss (Original Model Loss)"] = loss_sum / max(1.0, n_batches)

    denom_far = TP + FP
    denom_hit = TP + FN
    denom_csi = TP + FP + FN
    out[f"HIT (%) @ {threshold}"] = (TP / denom_hit) * 100.0 if denom_hit > 0 else 0.0
    out[f"FAR (%) @ {threshold}"] = (FP / denom_far) * 100.0 if denom_far > 0 else 0.0
    out[f"CSI (%) @ {threshold}"] = (TP / denom_csi) * 100.0 if denom_csi > 0 else 0.0
    return out

def print_comparison_table(summary, threshold, title="Global Test Results"):
    cols = list(summary.keys())
    header = f"{'Metric':<20} | " + " | ".join(f"{c:^12}" for c in cols)
    line = "-" * len(header)

    def row(name, fmt="{:12.6f}"):
        vals = []
        for c in cols:
            v = summary[c].get(name, None)
            if v is None:
                vals.append(f"{'N/A':^12}")
            else:
                vals.append(fmt.format(v))
        return f"{name:<20} | " + " | ".join(vals)

    print("\n" + "=" * len(header))
    print(" " * max(0, (len(header) - len(title)) // 2) + title)
    print("=" * len(header))
    print(header)
    print(line)
    print(row("Loss (Model Only)"))
    print(row("MSE"))
    print(row("MAE"))
    print(row("Masked MSE"))
    print(row("Masked MAE"))
    print(row("Volume Rel Err (%)", fmt="{:12.2f}"))
    print(line)
    print(f"Metrics for threshold {threshold}:")
    print(row(f"HIT (%) @ {threshold}", fmt="{:12.2f}"))
    print(row(f"FAR (%) @ {threshold}", fmt="{:12.2f}"))
    print(row(f"CSI (%) @ {threshold}", fmt="{:12.2f}"))
    print("=" * len(header) + "\n")
