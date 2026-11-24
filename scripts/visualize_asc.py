#!/usr/bin/env python3
"""Visualiza archivos .asc (ASCII grid) y genera GIFs de la evolución.

Uso básico:
  python scripts/visualize_asc.py --input outputs --output outputs_visualized --step 3

El script busca recursivamente archivos .asc cuyo nombre contenga el patrón (por defecto 'H'),
selecciona 1 de cada `step` y guarda imágenes y un GIF por cada simulación.
"""
import argparse
import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image


def read_asc(path):
    """Leer un archivo ESRI ASCII grid (.asc) sin depender de rasterio.
    Devuelve (data 2D numpy array, header dict)
    """
    header = {}
    header_keys = ['ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'NODATA_value']
    # Open with a permissive encoding to handle non-UTF8 .out files
    with open(path, 'r', encoding='latin-1', errors='replace') as f:
        # Read header lines (first 6 lines normalmente)
        # Some files may have fewer; read until we've got ncols and nrows and then data
        head_lines = []
        while len(head_lines) < 6:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].lower() in [k.lower() for k in header_keys]:
                head_lines.append(line)
            else:
                # maybe reached data; go back
                f.seek(pos)
                break
        for hl in head_lines:
            parts = hl.strip().split()
            key = parts[0].lower()
            val = parts[1]
            try:
                if key in ['ncols','nrows']:
                    header[key] = int(val)
                else:
                    header[key] = float(val)
            except Exception:
                header[key] = float('nan')
        # Read remaining as data (use StringIO to control parsing)
        from io import StringIO
        rest = f.read()
        # remove null bytes which break loadtxt
        rest = rest.replace('\x00', ' ')
        if not rest.strip():
            raise ValueError('No numeric data found')
        try:
            data = np.loadtxt(StringIO(rest))
        except Exception as e:
            raise ValueError(f'Could not parse numeric grid: {e}')
    # If header has nrows/ncols and data shape mismatch, try to reshape
    if 'nrows' in header and 'ncols' in header:
        expected = (header['nrows'], header['ncols'])
        if data.size == expected[0] * expected[1] and data.shape != expected:
            data = data.reshape(expected)
    return data, header


def find_simulations(root, pattern='H'):
    """Buscar carpetas de simulación que contengan archivos .asc con el patrón."""
    sims = {}
    for dirpath, dirnames, filenames in os.walk(root):
        if any(part.lower() == 'bin' for part in dirpath.split(os.sep)):
            continue
        asc_files = [os.path.join(dirpath, f) for f in filenames if f.lower().endswith(('.asc', '.out')) and pattern in f]
        if asc_files:
            # group by the simulation root: take the path up to '/output/asc' or the parent dir
            sims.setdefault(dirpath, []).extend(sorted(asc_files))
    return sims


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_and_save(data, header, outpath, vmin=None, vmax=None, cmap='viridis', contours=False, dpi=150, threshold=0.0):
    fig, ax = plt.subplots(figsize=(6,5))
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)
    extent = None
    if header:
        try:
            x0 = header.get('xllcorner', 0)
            y0 = header.get('yllcorner', 0)
            cs = header.get('cellsize', 1)
            ncols = int(header.get('ncols', data.shape[1]))
            nrows = int(header.get('nrows', data.shape[0]))
            extent = (x0, x0 + ncols*cs, y0, y0 + nrows*cs)
        except Exception:
            extent = None
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad('white')
    masked = np.ma.masked_invalid(data)
    if threshold > 0:
        masked = np.ma.masked_less(masked, threshold)
    im = ax.imshow(masked, origin='upper', cmap=cmap_obj, vmin=vmin, vmax=vmax, extent=extent)
    ax.set_xticks([])
    ax.set_yticks([])
    if contours:
        levels = np.linspace(vmin, vmax, 6)
        ax.contour(masked, levels=levels, colors='k', linewidths=0.4, extent=extent)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def make_gif(image_paths, out_gif, duration=0.5):
    imgs = []
    for p in image_paths:
        imgs.append(imageio.v2.imread(p))
    imageio.mimsave(out_gif, imgs, duration=duration)


def process_simulation(asc_list, outroot, relpath, step=3, cmap='viridis', contours=False, duration=0.5, threshold=0.0):
    # Select 1 of each `step`
    selected = asc_list[::step]
    if not selected:
        return
    # Determine vmin/vmax across selected to keep consistent scale
    vmin = float('inf')
    vmax = float('-inf')
    headers = {}
    datas = {}
    for p in selected:
        try:
            data, header = read_asc(p)
        except Exception as e:
            print(f'Skipping file (not a valid ASCII grid): {p} -> {e}')
            continue
        # mask nodata
        nod = header.get('NODATA_value', None) or header.get('nodata_value', None)
        if nod is not None:
            data = np.where(data == nod, np.nan, data)
        if threshold > 0:
            data = np.where(data < threshold, np.nan, data)
        datas[p] = data
        headers[p] = header
        if np.all(np.isnan(data)):
            continue
        mi = np.nanmin(data)
        ma = np.nanmax(data)
        if mi < vmin: vmin = mi
        if ma > vmax: vmax = ma
    # Prepare output dirs
    images_dir = os.path.join(outroot, relpath, 'images')
    gifs_dir = os.path.join(outroot, relpath, 'gifs')
    ensure_dir(images_dir)
    ensure_dir(gifs_dir)
    image_paths = []
    for i, p in enumerate(selected):
        if p not in datas:
            continue
        fname = os.path.splitext(os.path.basename(p))[0]
        outimg = os.path.join(images_dir, f"{i:03d}_{fname}.png")
        plot_and_save(datas[p], headers[p], outimg, vmin=vmin, vmax=vmax, cmap=cmap, contours=contours, threshold=threshold)
        image_paths.append(outimg)
    if not image_paths:
        print('No valid frames found for', relpath)
        return
    # Make GIF
    gif_name = os.path.join(gifs_dir, f"evolution_{os.path.basename(os.path.normpath(relpath))}.gif")
    make_gif(image_paths, gif_name, duration=duration)


def main():
    parser = argparse.ArgumentParser(description='Visualizar .asc y generar GIFs de H (1 de cada N).')
    parser.add_argument('--input', '-i', default='outputs', help='Carpeta raíz con outputs')
    parser.add_argument('--output', '-o', default='outputs_visualized', help='Carpeta raíz de salida')
    parser.add_argument('--pattern', '-p', default='H', help="Patrón a buscar en nombres de archivo (por defecto 'H')")
    parser.add_argument('--step', '-s', type=int, default=3, help='Selecciona 1 de cada `step` timesteps')
    parser.add_argument('--threshold', type=float, default=0.01, help='Oculta valores por debajo de este umbral (por defecto 0.01 = 1cm)')
    parser.add_argument('--cmap', default='viridis', help='Paleta de colores matplotlib')
    parser.add_argument('--contours', action='store_true', help='Dibujar contornos encima del raster')
    parser.add_argument('--duration', type=float, default=0.5, help='Duración de cada frame en el GIF (segundos)')
    args = parser.parse_args()

    sims = find_simulations(args.input, pattern=args.pattern)
    if not sims:
        print('No se encontraron archivos .asc con el patrón en', args.input)
        return
    print(f'Encontradas {len(sims)} carpetas con archivos .asc. Procesando...')
    for dirpath, asc_list in sims.items():
        # create a relative path from input root to dirpath to mirror structure
        rel = os.path.relpath(dirpath, args.input)
        print('Procesando', rel)
        process_simulation(asc_list, args.output, rel, step=args.step, cmap=args.cmap, contours=args.contours, duration=args.duration, threshold=args.threshold)
    print('Listo. Salidas en:', args.output)


if __name__ == '__main__':
    main()
