#!/usr/bin/env python3
"""
Aplica una máscara espacial basada en el shapefile del dominio a resultados TXT
de la U-Net y genera:
  - TXT enmascarado (fuera del dominio -> nan)
  - PNG enmascarado

Pensado para matrices de 10, 20 y 40 m con el mismo origen espacial.
Los originales no se modifican.
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

try:
    import shapefile as pyshp
except ImportError as exc:
    pyshp = None
    _PYSHAPE_IMPORT_ERROR = exc
else:
    _PYSHAPE_IMPORT_ERROR = None


# Alineación validada contra DEM_BM_10m.txt
X_MIN = 674583.0
Y_MAX = 4610227.0
NODATA_DEM = -9999.0
RESOLUTIONS = {
    (548, 292): 10.0,
    (274, 146): 20.0,
    (137, 73): 40.0,
}


def _ring_area(points):
    if len(points) < 3:
        return 0.0

    x_coords = np.asarray([point[0] for point in points], dtype=float)
    y_coords = np.asarray([point[1] for point in points], dtype=float)

    if np.allclose(points[0], points[-1]):
        x_coords = x_coords[:-1]
        y_coords = y_coords[:-1]

    if len(x_coords) < 3:
        return 0.0

    return 0.5 * abs(
        np.dot(x_coords, np.roll(y_coords, -1))
        - np.dot(y_coords, np.roll(x_coords, -1))
    )


def _shape_area(shape):
    if not shape.points:
        return 0.0

    if not shape.parts:
        return _ring_area(shape.points)

    part_starts = list(shape.parts) + [len(shape.points)]
    return sum(
        _ring_area(shape.points[start:end])
        for start, end in zip(part_starts[:-1], part_starts[1:])
    )


def _shape_to_mask(shape, out_shape, cell_size):
    if not shape.parts:
        ring_points = shape.points
    else:
        ring_end = shape.parts[1] if len(shape.parts) > 1 else len(shape.points)
        ring_points = shape.points[:ring_end]

    if len(ring_points) < 3:
        return np.zeros(out_shape, dtype=bool)

    row_count, col_count = out_shape
    x_coords = X_MIN + cell_size * (0.5 + np.arange(col_count, dtype=float))
    y_coords = Y_MAX - cell_size * (0.5 + np.arange(row_count, dtype=float))
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    path = MplPath(np.asarray(ring_points, dtype=float), closed=True)
    return path.contains_points(points).reshape(out_shape)


def build_masks(shapefile: Path, dem_files_by_shape: dict[tuple[int, int], Path | None]):
    """Construye máscaras por resolución a partir del mismo shapefile principal."""
    if pyshp is None:
        raise ImportError(
            "No se pudo importar 'shapefile'. Instala el paquete 'pyshp' en el entorno activo."
        ) from _PYSHAPE_IMPORT_ERROR

    reader = pyshp.Reader(str(shapefile))
    shapes = reader.shapes()

    if not shapes:
        raise ValueError("El shapefile no contiene geometrías.")

    # Usamos el polígono de mayor área como dominio exterior principal.
    main_geom = max(shapes, key=_shape_area)

    masks_by_shape = {}

    for shape, cell_size in RESOLUTIONS.items():
        shape_mask = _shape_to_mask(main_geom, shape, cell_size)

        dem_file = dem_files_by_shape.get(shape)
        dem_mask = None
        final_mask = shape_mask

        if dem_file is not None:
            if not dem_file.exists():
                print(
                    f"[WARN] DEM no encontrado para {int(cell_size)} m: {dem_file}. "
                    "Se usará solo la máscara del shapefile."
                )
            else:
                dem = np.loadtxt(dem_file)

                if dem.shape != shape:
                    print(
                        f"[WARN] DEM {dem_file.name} con forma {dem.shape} no coincide con {shape}. "
                        "Se usará solo la máscara del shapefile."
                    )
                else:
                    dem_mask = dem != NODATA_DEM
                    final_mask = shape_mask & dem_mask

        masks_by_shape[shape] = {
            "cell_size": cell_size,
            "shape_mask": shape_mask,
            "dem_mask": dem_mask,
            "final_mask": final_mask,
        }

    return masks_by_shape


def mask_txt(input_txt: Path, output_txt: Path, mask: np.ndarray):
    """Carga un TXT, pone NaN fuera de la máscara y guarda el resultado."""
    data = np.loadtxt(input_txt)

    if data.shape != mask.shape:
        raise ValueError(
            f"{input_txt.name}: forma {data.shape}; "
            f"se esperaba {mask.shape}"
        )

    masked = data.copy()
    masked[~mask] = np.nan

    np.savetxt(output_txt, masked, fmt="%.6f")
    return masked


def save_png(data: np.ndarray, output_png: Path, title: str):
    """Genera el PNG a partir del TXT ya enmascarado."""
    cmap = plt.cm.winter.copy()
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(8, 11))
    im = ax.imshow(data, cmap=cmap, origin="upper")

    ax.set_title(title)
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Prediction")

    fig.tight_layout()
    fig.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close(fig)


def process_file(input_txt: Path, output_dir: Path, masks_by_shape: dict[tuple[int, int], dict]):
    """Procesa un único TXT."""
    stem = input_txt.stem

    data = np.loadtxt(input_txt)
    shape_info = masks_by_shape.get(data.shape)

    if shape_info is None:
        print(f"[OMITIDO] {input_txt.name}: shape {data.shape} no reconocida")
        return False

    output_txt = output_dir / f"{stem}_masked.txt"
    output_png = output_dir / f"{stem}_masked.png"

    mask = shape_info["final_mask"]

    if data.shape != mask.shape:
        print(f"[OMITIDO] {input_txt.name}: shape {data.shape} no reconocida")
        return False

    masked = data.copy()
    masked[~mask] = np.nan

    np.savetxt(output_txt, masked, fmt="%.6f")

    removed = np.sum(np.isfinite(data) & ~mask)

    save_png(
        masked,
        output_png,
        title=f"{stem} (masked)"
    )

    print(
        f"[OK] {input_txt.name} | resolución {int(shape_info['cell_size'])} m | "
        f"shape {data.shape} | valores eliminados: {removed}"
    )

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Enmascara espacialmente resultados TXT de la U-Net."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="TXT individual o carpeta con TXT originales."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Carpeta donde guardar TXT y PNG procesados."
    )

    parser.add_argument(
        "--shapefile",
        required=True,
        help="Ruta a dominio_barranco.shp"
    )

    parser.add_argument(
        "--dem",
        required=False,
        help="Alias legacy de --dem10"
    )

    parser.add_argument(
        "--dem10",
        required=False,
        help="Ruta a DEM_BM_10m.txt"
    )

    parser.add_argument(
        "--dem20",
        required=False,
        help="Ruta a DEM_BM_20m.txt"
    )

    parser.add_argument(
        "--dem40",
        required=False,
        help="Ruta a DEM_BM_40m.txt"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    shapefile = Path(args.shapefile)
    dem10 = Path(args.dem10 or args.dem) if (args.dem10 or args.dem) else None
    dem20 = Path(args.dem20) if args.dem20 else None
    dem40 = Path(args.dem40) if args.dem40 else None

    output_dir.mkdir(parents=True, exist_ok=True)

    masks_by_shape = build_masks(
        shapefile,
        {
            (548, 292): dem10,
            (274, 146): dem20,
            (137, 73): dem40,
        },
    )

    for shape, info in masks_by_shape.items():
        if info["dem_mask"] is None:
            print(
                f"[WARN] {int(info['cell_size'])} m: usando solo la máscara del shapefile"
            )
        else:
            inter = np.logical_and(info["shape_mask"], info["dem_mask"]).sum()
            union = np.logical_or(info["shape_mask"], info["dem_mask"]).sum()
            iou = inter / union if union else 0.0
            differences = np.logical_xor(info["shape_mask"], info["dem_mask"]).sum()

            print(f"[INFO] {int(info['cell_size'])} m | shape {shape}")
            print(f"       celdas máscara shapefile: {info['shape_mask'].sum()}")
            print(f"       celdas válidas DEM: {info['dem_mask'].sum()}")
            print(f"       diferencias shapefile/DEM: {differences}")
            print(f"       IoU shapefile/DEM: {iou:.8f}")
    print()

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob("*.txt"))
        # Evitar reprocesar resultados ya enmascarados
        files = [f for f in files if not f.stem.endswith("_masked")]
    else:
        raise FileNotFoundError(f"No existe: {input_path}")

    if not files:
        raise FileNotFoundError("No se encontraron TXT para procesar.")

    for txt in files:
        process_file(txt, output_dir, masks_by_shape)


if __name__ == "__main__":
    main()
