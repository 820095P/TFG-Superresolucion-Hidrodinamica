#!/usr/bin/env python3
"""Genera comparativas 20/40 m -> 10 m con la mascara espacial original."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, PowerNorm
import numpy as np

from aplicar_mascara_tfg import (
    NODATA_DEM,
    RESOLUTIONS,
    X_MIN,
    Y_MAX,
    build_masks,
)


ROLE_PATTERNS = {
    "coarse": re.compile(r"^Unet_.+_coarse_cropped_.+\.txt$", re.IGNORECASE),
    "gt": re.compile(r"^Unet_.+_GT_.+\.txt$", re.IGNORECASE),
    "pred": re.compile(r"^Unet_.+_pred_.+\.txt$", re.IGNORECASE),
    "error": re.compile(r"^Unet_.+_abs_err_.+\.txt$", re.IGNORECASE),
}


def discover_input_files(
    input_dir: Path,
    explicit_files: dict[str, Path | None] | None = None,
) -> dict[str, Path]:
    """Encuentra los cuatro TXT de un experimento sin fijar nombres concretos."""
    explicit_files = explicit_files or {}
    candidates = {
        role: sorted(
            path for path in input_dir.glob("*.txt")
            if pattern.match(path.name) and not path.stem.endswith("_masked")
        )
        for role, pattern in ROLE_PATTERNS.items()
    }
    files = {
        role: path.resolve()
        for role, path in explicit_files.items()
        if path is not None
    }
    missing = [role for role, paths in candidates.items() if not paths]
    missing = [role for role in missing if role not in files]
    ambiguous = {
        role: paths for role, paths in candidates.items()
        if len(paths) > 1 and role not in files
    }
    if missing:
        raise FileNotFoundError(
            f"No se encontró el TXT de {', '.join(missing)} en {input_dir}. "
            "Se esperan los sufijos *_coarse_cropped_*.txt, *_GT_*.txt, "
            "*_pred_*.txt y *_abs_err_*.txt."
        )
    if ambiguous:
        details = "; ".join(
            f"{role}: {', '.join(path.name for path in paths)}"
            for role, paths in ambiguous.items()
        )
        raise ValueError(
            f"Hay varios candidatos para un mismo papel en {input_dir}: {details}. "
            "Usa --coarse, --gt, --pred y --error para desambiguarlos."
        )
    files.update({role: paths[0].resolve() for role, paths in candidates.items() if role not in files})
    print("[INPUT] Archivos detectados automáticamente:")
    for role, path in files.items():
        print(f"        {role}: {path.name}")
    return files


def experiment_stem(files: dict[str, Path]) -> str:
    """Obtiene una etiqueta estable a partir del prefijo común de los cuatro TXT."""
    names = [path.stem for path in files.values()]
    common = names[0]
    for name in names[1:]:
        index = 0
        while index < min(len(common), len(name)) and common[index] == name[index]:
            index += 1
        common = common[:index]
    common = common.rstrip("_-")
    return common.removeprefix("Unet_") or "experimento"


def infer_coarse_resolution(shape: tuple[int, int]) -> int:
    resolutions = {shape: int(cell_size) for shape, cell_size in RESOLUTIONS.items()}
    if shape not in resolutions or resolutions[shape] == 10:
        raise ValueError(f"La forma del coarse {shape} no corresponde a 20 m o 40 m.")
    return resolutions[shape]


def load_dem(path: Path | None) -> np.ndarray | None:
    """Carga DEM TXT o ESRI ASCII Grid sin alterar sus valores."""
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"No existe el DEM: {path}")
    skiprows = 6 if path.suffix.lower() == ".asc" else 0
    dem = np.loadtxt(path, skiprows=skiprows)
    return dem


def validate_dem_header(path: Path | None, cell_size: float) -> None:
    if path is None or path.suffix.lower() != ".asc":
        return
    header = {}
    with path.open(encoding="ascii") as stream:
        for _ in range(6):
            key, value = stream.readline().split()[:2]
            header[key.lower()] = float(value)
    if header.get("cellsize") != cell_size:
        raise ValueError(
            f"{path.name}: cellsize={header.get('cellsize')} y no {cell_size}."
        )
    expected_xll = X_MIN
    expected_yll = Y_MAX - cell_size * next(
        shape[0] for shape, resolution in RESOLUTIONS.items() if resolution == cell_size
    )
    mismatch = []
    if not np.isclose(header.get("xllcorner", np.nan), expected_xll):
        mismatch.append(f"xllcorner={header.get('xllcorner')} (esperado {expected_xll})")
    if not np.isclose(header.get("yllcorner", np.nan), expected_yll):
        mismatch.append(f"yllcorner={header.get('yllcorner')} (esperado {expected_yll})")
    if mismatch:
        print(
            f"[WARN] {path.name}: origen ASC distinto al origen fijo del script original: "
            + "; ".join(mismatch)
            + ". Se conserva el indexado original por resolucion."
        )


def load_and_check(path: Path, expected_shape: tuple[int, int] | None) -> np.ndarray:
    data = np.loadtxt(path)
    print(f"[INFO] {path.name}: shape={data.shape}, finitas={np.isfinite(data).sum()}")
    if expected_shape is not None and data.shape != expected_shape:
        raise ValueError(f"{path.name}: shape={data.shape}; esperado {expected_shape}")
    return data


def apply_mask(name: str, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    before = np.isfinite(data)
    masked = data.copy()
    masked[~mask] = np.nan
    after = np.isfinite(masked)
    print(
        f"[MASK] {name}: validas antes={before.sum()}, despues={after.sum()}, "
        f"eliminadas={np.logical_and(before, ~mask).sum()}, "
        f"NaN conservados={np.logical_and(~before, mask).sum()}"
    )
    return masked


def write_masked_txt(path: Path, data: np.ndarray) -> None:
    np.savetxt(path, data, fmt="%.6f")


def select_error(gt: np.ndarray, pred: np.ndarray, exported_error: np.ndarray,
                 source: str = "auto") -> tuple[np.ndarray, str]:
    """Valida el error exportado. Si no corresponde, evita dibujarlo como error
    de la predicción actual; en modo auto usa GT-pred sobre el soporte común.
    Nunca rellena NaN del error exportado ni modifica los TXT originales.
    """
    shared = np.isfinite(gt) & np.isfinite(pred) & np.isfinite(exported_error)
    if not shared.any():
        raise ValueError("No hay celdas compartidas para validar los errores.")
    actual = gt - pred
    diff = exported_error[shared] - actual[shared]
    max_diff = float(np.max(np.abs(diff)))
    mean_diff = float(np.mean(np.abs(diff)))
    consistent = np.allclose(exported_error[shared], actual[shared], rtol=2e-5, atol=2e-5)
    if consistent:
        print(f"[OK] Error exportado validado (diferencia máxima {max_diff:.2e} m).")
    else:
        print(f"[AVISO] Error exportado INCONSISTENTE con GT-pred: "
              f"diferencia máxima {max_diff:.6g} m; media {mean_diff:.6g} m.")
        print("[AVISO] Comprobar que GT, pred y error correspondan a la misma muestra/ejecución.")

    if source == "exported":
        if not consistent:
            raise ValueError("No se puede usar --error-source exported con un TXT de error inconsistente.")
        return exported_error, "exportado y validado"

    if source == "auto" and consistent:
        return exported_error, "exportado y validado"

    # En modo auto, si no coincide, el error verdadero de ESTOS mapas procede
    # de GT y pred. Se respeta el soporte original: no inventamos valores allí
    # donde el TXT de error marcaba NaN.
    result = np.full_like(exported_error, np.nan, dtype=float)
    result[shared] = actual[shared]
    print("[AVISO] Se usa GT-pred calculado desde los TXT actuales, SOLO en "
          "las celdas donde el error exportado también era válido.")
    return result, "recalculado GT-pred (soporte del TXT exportado)"


def save_individual_png(path: Path, data: np.ndarray, title: str, cmap: str) -> None:
    figure, axis = plt.subplots(figsize=(6, 8), constrained_layout=True)
    masked = np.ma.masked_invalid(data)
    image = axis.imshow(masked, cmap=cmap, origin="upper", interpolation="none")
    axis.set_title(title)
    axis.axis("off")
    figure.colorbar(image, ax=axis, label="Profundidad (m)")
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_comparison(
    output_png: Path,
    output_pdf: Path | None,
    maps: list[np.ndarray],
    signed_error: bool,
    depth_vmax: float | None = None,
    error_vmax: float | None = None,
) -> None:
    """Una fila de cuatro mapas; escalas y barras compartidas sin solapamiento.

    Mantiene la máscara y los NaN de cada TXT. No interpola el raster coarse:
    imshow usa el mismo extent geográfico para las diferentes resoluciones.
    """
    depth_values = np.concatenate([a[np.isfinite(a)] for a in maps[:3]])
    if depth_values.size == 0:
        raise ValueError("No hay profundidades válidas para representar.")
    measured_depth_max = float(np.max(depth_values))
    vmax = measured_depth_max if depth_vmax is None else depth_vmax
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("El máximo de profundidad debe ser positivo.")
    if vmax < measured_depth_max:
        print(f"[WARN] vmax profundidad={vmax:g} m: valores superiores quedarán saturados.")

    error_values = maps[3][np.isfinite(maps[3])]
    if error_values.size == 0:
        raise ValueError("El error enmascarado no contiene celdas válidas.")
    measured_error_max = float(np.max(np.abs(error_values)))
    error_limit = measured_error_max if error_vmax is None else error_vmax
    if not np.isfinite(error_limit) or error_limit <= 0:
        raise ValueError("El máximo de error debe ser positivo.")
    if error_limit < measured_error_max:
        print(f"[WARN] vmax error={error_limit:g} m: valores superiores quedarán saturados.")

    # La paleta original 'winter' muestra las profundidades pequeñas en azul
    # intenso; Blues las mostraba casi blancas y hacía invisibles los cauces.
    cmap_depth = plt.get_cmap("winter").copy()
    cmap_depth.set_bad("white")
    cmap_error = plt.get_cmap("RdBu_r" if signed_error else "Reds").copy()
    cmap_error.set_bad("white")
    depth_norm = Normalize(vmin=0, vmax=vmax)
    error_norm = (
        TwoSlopeNorm(vmin=-error_limit, vcenter=0, vmax=error_limit)
        if signed_error
        else PowerNorm(gamma=0.5, vmin=0, vmax=error_limit)
    )

    extent = [X_MIN, X_MIN + 292 * 10.0, Y_MAX - 548 * 10.0, Y_MAX]

    # Columnas: coarse, GT, pred, barra profundidades, error, barra error.
    # La barra de profundidades va entre (c) y (d), NO encima de (c).
    figure = plt.figure(figsize=(13.5, 4.8), facecolor="white")
    gs = figure.add_gridspec(
        1, 6, width_ratios=[1, 1, 1, .055, 1, .055],
        left=.016, right=.925, top=.975, bottom=.025, wspace=.12
    )
    axes = [figure.add_subplot(gs[0, i]) for i in (0, 1, 2, 4)]
    cax_depth = figure.add_subplot(gs[0, 3])
    cax_error = figure.add_subplot(gs[0, 5])

    for ax, data, label, cmap, norm in zip(
        axes, maps, ("(a)", "(b)", "(c)", "(d)"),
        (cmap_depth, cmap_depth, cmap_depth, cmap_error),
        (depth_norm, depth_norm, depth_norm, error_norm)
    ):
        ax.imshow(
            np.ma.masked_invalid(data), cmap=cmap, norm=norm,
            origin="upper", extent=extent, interpolation="nearest", aspect="equal"
        )
        ax.set_axis_off()
        ax.text(.02, .98, label, transform=ax.transAxes,
                va="top", ha="left", fontsize=10, color="black")

    from matplotlib.cm import ScalarMappable
    dbar = figure.colorbar(ScalarMappable(norm=depth_norm, cmap=cmap_depth), cax=cax_depth)
    dbar.set_label("Profundidad (m)", fontsize=9)
    dbar.ax.tick_params(labelsize=8)
    ebar = figure.colorbar(ScalarMappable(norm=error_norm, cmap=cmap_error), cax=cax_error)
    ebar.set_label("Error con signo (m)" if signed_error else "Error absoluto (m)", fontsize=9)
    ebar.ax.tick_params(labelsize=8)
    if not signed_error:
        # La normalización por potencia hace visibles los errores pequeños,
        # pero las marcas siguen expresadas en metros reales.
        ticks = [0.0, 0.02, 0.05, 0.10, 0.20, error_limit]
        ticks = sorted(set(t for t in ticks if t <= error_limit))
        ebar.set_ticks(ticks)

    figure.savefig(output_png, dpi=350, facecolor="white")
    if output_pdf is not None:
        figure.savefig(output_pdf, facecolor="white")
    plt.close(figure)
    print(f"[OK] Comparativa PNG: {output_png}")
    if output_pdf is not None:
        print(f"[OK] Comparativa PDF: {output_pdf}")
    print(f"[CHECK] Profundidades con escala compartida: [0, {vmax:.6g}] m")
    print(f"[CHECK] Error {'con signo' if signed_error else 'absoluto'}: "
          f"{'[-' if signed_error else '[0, '}{error_limit:.6g}] m")
    print("[CHECK] Resoluciones nativas, extent común, sin interpolación numérica")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--shapefile", type=Path)
    parser.add_argument("--dem10", type=Path)
    parser.add_argument("--dem20", type=Path)
    parser.add_argument("--dem40", type=Path)
    parser.add_argument("--coarse-resolution", choices=(20, 40), type=int,
                        help="Comprueba la resolucion; si se omite se infiere por la forma.")
    parser.add_argument("--coarse", type=Path)
    parser.add_argument("--gt", type=Path)
    parser.add_argument("--pred", type=Path)
    parser.add_argument("--error", type=Path)
    parser.add_argument("--signed-error", action="store_true")
    parser.add_argument("--error-source", choices=("auto", "exported", "recomputed"),
                        default="auto", help="auto: usar exportado si coincide, "
                        "si no recalcular GT-pred conservando el soporte del TXT de error.")
    parser.add_argument("--save-pdf", action="store_true",
                        help="Exportar además el PDF (por defecto solo PNG).")
    parser.add_argument("--save-individual", action="store_true",
                        help="Exporta también cuatro PNG individuales (opcional).")
    parser.add_argument("--save-masked-txt", action="store_true",
                        help="Exporta también los cuatro TXT enmascarados (opcional).")
    parser.add_argument("--depth-vmax", type=float,
                        help="Máximo común opcional de profundidad para varias figuras.")
    parser.add_argument("--error-vmax", type=float,
                        help="Máximo opcional de error para varias figuras.")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    project_dir = Path(__file__).resolve().parents[1]
    output_dir = (args.output_dir or input_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = discover_input_files(
        input_dir,
        {role: getattr(args, role) for role in ROLE_PATTERNS},
    )

    for path in files.values():
        if not path.exists():
            raise FileNotFoundError(f"No existe el TXT: {path}")
    shapefile = (args.shapefile or project_dir / "dominio_barranco.shp").resolve()
    dem_paths = {
        10: (args.dem10 or project_dir / "TFG" / "DEM_BM_10m.asc").resolve(),
        20: (args.dem20 or project_dir / "TFG" / "DEM_BM_20m.asc").resolve(),
        40: (args.dem40 or project_dir / "TFG" / "DEM_BM_40m.asc").resolve(),
    }
    if not shapefile.exists():
        raise FileNotFoundError(f"No existe el shapefile: {shapefile}")

    for resolution, dem_path in dem_paths.items():
        if not dem_path.exists():
            raise FileNotFoundError(f"No existe el DEM de {resolution} m: {dem_path}")
        validate_dem_header(dem_path, float(resolution))

    dem_arrays = {resolution: load_dem(path) for resolution, path in dem_paths.items()}
    masks = build_masks(
        shapefile,
        {
            shape: None
            for shape, resolution in [
                ((548, 292), 10),
                ((274, 146), 20),
                ((137, 73), 40),
            ]
        },
    )
    # build_masks carga TXT directamente; para ASC se sustituyen sus validaciones
    # con la misma interseccion original, conservando el poligono y la orientacion.
    for shape, resolution in [((548, 292), 10), ((274, 146), 20), ((137, 73), 40)]:
        if dem_arrays[resolution] is not None:
            if dem_arrays[resolution].shape != shape:
                raise ValueError(
                    f"DEM de {resolution} m con forma {dem_arrays[resolution].shape}; esperada {shape}"
                )
            masks[shape]["dem_mask"] = dem_arrays[resolution] != NODATA_DEM
            masks[shape]["final_mask"] = masks[shape]["shape_mask"] & masks[shape]["dem_mask"]
        info = masks[shape]
        print(
            f"[MASK] {resolution} m: shape={shape}, poligono={info['shape_mask'].sum()}, "
            f"DEM={info['dem_mask'].sum() if info['dem_mask'] is not None else 'no disponible'}, "
            f"final={info['final_mask'].sum()}"
        )

    coarse = load_and_check(files["coarse"], None)
    detected_resolution = infer_coarse_resolution(coarse.shape)
    if args.coarse_resolution is not None and args.coarse_resolution != detected_resolution:
        raise ValueError(
            f"--coarse-resolution={args.coarse_resolution} no coincide con la forma "
            f"detectada {coarse.shape} ({detected_resolution} m)."
        )
    coarse_resolution = detected_resolution
    coarse_shape = coarse.shape
    gt = load_and_check(files["gt"], (548, 292))
    pred = load_and_check(files["pred"], (548, 292))
    error = load_and_check(files["error"], (548, 292))

    coarse_mask = masks[coarse_shape]["final_mask"]
    fine_mask = masks[(548, 292)]["final_mask"]
    maps = [
        apply_mask("coarse", coarse, coarse_mask),
        apply_mask("GT", gt, fine_mask),
        apply_mask("pred", pred, fine_mask),
        apply_mask("error", error, fine_mask),
    ]
    error_plot, error_origin = select_error(maps[1], maps[2], maps[3], args.error_source)

    stem = f"comparativa_{experiment_stem(files)}"
    for key, data in zip(("coarse", "gt", "pred", "error"), maps):
        if args.save_masked_txt:
            write_masked_txt(output_dir / f"{files[key].stem}_masked.txt", data)
        if args.save_individual:
            # Mantener el modo individual accesible; visualizar su escala original.
            save_individual_png(output_dir / f"{files[key].stem}_masked.png", data,
                                key, "winter")

    error_for_plot = np.abs(error_plot) if not args.signed_error else error_plot
    print(f"[INFO] Fuente del panel de error: {error_origin}")
    save_comparison(output_dir / f"{stem}.png",
                    output_dir / f"{stem}.pdf" if args.save_pdf else None,
                    [*maps[:3], error_for_plot], args.signed_error,
                    args.depth_vmax, args.error_vmax)


if __name__ == "__main__":
    main()
