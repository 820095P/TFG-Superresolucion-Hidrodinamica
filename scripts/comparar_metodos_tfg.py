"""Paneles U-Net / bicubica / subgrid para el TFG, con mascara real del dominio.

Guardar en: ProyectoTFG/scripts_mascara_figuras_comparaciones/
Junto a:    aplicar_mascara_tfg.py

Ejecutar desde VS Code:
    python comparar_metodos_tfg_enmascarado.py
    python comparar_metodos_tfg_enmascarado.py --instante 80

Reutiliza EXACTAMENTE la mascara espacial de aplicar_mascara_tfg.py:
shapefile del Barranco + celdas validas del DEM, a su resolucion nativa.
No reescribe los .npz, no modifica metricas ni entrena modelos.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
import numpy as np

# Este modulo ya existe en la misma carpeta del proyecto.
from aplicar_mascara_tfg import (
    RESOLUTIONS, NODATA_DEM, X_MIN, Y_MAX, build_masks,
)

# ----------------------- RUTAS / CONFIGURACION -----------------------
CARPETA_RAIZ = Path(
    r"C:\Users\PC\Desktop\TFG\ProyectoTFG\resultados\resultados_sel"
)
DIRECTORIO_PROYECTO = Path(__file__).resolve().parent.parent
INSTANTE = 89

CASOS = (
    {"carpeta": "20to10", "modelo": "20to10_exp5", "resolucion": 20,
     "titulo": "20 m a 10 m", "salida": "comparacion_metodos_20to10.png"},
    {"carpeta": "40to10", "modelo": "40to10_exp1", "resolucion": 40,
     "titulo": "40 m a 10 m", "salida": "comparacion_metodos_40to10.png"},
)

# La normalizacion PowerNorm y el percentil SOLO alteran los colores,
# nunca los datos fisicos, los errores almacenados o las metricas del TFG.
ERROR_PERCENTIL = 99.0
ERROR_GAMMA = 0.45
UMBRAL_AGUA_M = 0.01
# ---------------------------------------------------------------------


def localizar_npz(carpeta: Path, modelo: str, instante: int) -> Path:
    if not carpeta.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {carpeta}")
    encontrados = sorted(
        archivo for archivo in carpeta.glob(f"*_comparacion_t{instante:02d}.npz")
        if f"{modelo}_RealCase_" in archivo.name
    )
    if len(encontrados) != 1:
        disponibles = '\n  '.join(p.name for p in carpeta.glob('*_comparacion_t*.npz'))
        raise FileNotFoundError(
            f"Esperaba un unico .npz de {modelo} para el instante {instante} "
            f"en {carpeta}; encontrados: {len(encontrados)}.\n"
            f"Archivos existentes:\n  {disponibles or '(ninguno)'}"
        )
    return encontrados[0]


def leer_npz(archivo: Path, resolucion: int) -> dict[str, np.ndarray]:
    nombres = ("coarse", "gt", "unet", "bicubic", "subgrid")
    with np.load(archivo, allow_pickle=False) as z:
        if not set(nombres).issubset(z.files):
            raise ValueError(f"{archivo.name}: faltan {set(nombres) - set(z.files)}")
        datos = {nombre: np.asarray(z[nombre], dtype=np.float64)
                 for nombre in nombres}

    shape_coarse = next(forma for forma, res in RESOLUTIONS.items()
                        if int(res) == resolucion)
    shape_fine = next(forma for forma, res in RESOLUTIONS.items()
                      if int(res) == 10)
    for nombre, matriz in datos.items():
        esperado = shape_coarse if nombre == "coarse" else shape_fine
        if matriz.shape != esperado:
            raise ValueError(
                f"{archivo.name}: {nombre} tiene forma {matriz.shape}, "
                f"pero la mascara georreferenciada espera {esperado}. "
                "No aplico una mascara desalineada."
            )
        if not np.all(np.isfinite(matriz)):
            raise ValueError(f"{archivo.name}: {nombre} ya contiene NaN/inf")
    return datos


def cargar_dem(ruta: Path, forma: tuple[int, int]) -> np.ndarray:
    if not ruta.is_file():
        raise FileNotFoundError(
            f"Falta el DEM: {ruta}. Indica su ubicacion con --dem10, --dem20 o --dem40."
        )
    if ruta.suffix.lower() == ".asc":
        # ESRI ASCII Grid: 6 lineas de cabecera y luego la matriz.
        with ruta.open(encoding="utf-8") as archivo:
            cabecera = {}
            for _ in range(6):
                trozos = archivo.readline().strip().split()
                if len(trozos) >= 2:
                    cabecera[trozos[0].lower()] = trozos[1]
        if "nrows" in cabecera and int(cabecera["nrows"]) != forma[0]:
            raise ValueError(f"{ruta}: cabecera nrows no coincide con {forma}")
        if "ncols" in cabecera and int(cabecera["ncols"]) != forma[1]:
            raise ValueError(f"{ruta}: cabecera ncols no coincide con {forma}")
        if "cellsize" in cabecera:
            resolucion_esperada = RESOLUTIONS[forma]
            if not np.isclose(float(cabecera["cellsize"]), resolucion_esperada):
                raise ValueError(f"{ruta}: cellsize distinto de {resolucion_esperada}")
        nodata = float(cabecera.get("nodata_value", NODATA_DEM))
        dem = np.loadtxt(ruta, skiprows=6)
    else:
        nodata = NODATA_DEM
        dem = np.loadtxt(ruta)
    if dem.shape != forma:
        raise ValueError(f"{ruta.name}: DEM con forma {dem.shape}, esperada {forma}")
    return np.isfinite(dem) & (dem != nodata) & (dem != NODATA_DEM)


def preparar_mascaras(shapefile: Path, dems: dict[int, Path]) -> dict[tuple[int,int], np.ndarray]:
    if not shapefile.exists():
        raise FileNotFoundError(
            f"No existe el shapefile: {shapefile}. Usa --shapefile para indicar la ruta."
        )
    # Mismo poligono principal y mismos centros de celda que aplicar_mascara_tfg.py.
    info = build_masks(shapefile, {forma: None for forma in RESOLUTIONS})
    mascaras = {}
    for forma, resolucion in RESOLUTIONS.items():
        validas_dem = cargar_dem(dems[int(resolucion)], forma)
        mascara = info[forma]["shape_mask"] & validas_dem
        if not np.any(mascara):
            raise ValueError(f"Mascara vacia para {int(resolucion)} m. Revisar shapefile/DEM.")
        mascaras[forma] = mascara
        print(f"[MASCARA] {int(resolucion)} m: {mascara.sum():,} celdas dentro del dominio")
    return mascaras


def aplicar_mascaras(datos: dict[str,np.ndarray], resolucion: int,
                     mascaras: dict[tuple[int,int],np.ndarray]) -> dict[str,np.ndarray]:
    forma_coarse = next(forma for forma,res in RESOLUTIONS.items() if int(res)==resolucion)
    forma_fine = next(forma for forma,res in RESOLUTIONS.items() if int(res)==10)
    salida = {}
    for nombre, matriz in datos.items():
        mascara = mascaras[forma_coarse if nombre == "coarse" else forma_fine]
        resultado = matriz.copy()
        resultado[~mascara] = np.nan
        salida[nombre] = resultado
    return salida


def percentil_robusto(errores: dict[str,np.ndarray], p: float) -> float:
    # Calcula percentil sobre errores no nulos, evitando que los pixeles
    # secos de todo el dominio hundan el maximo de la barra de color.
    valores = [arr[np.isfinite(arr) & (arr > 0.1)] for arr in errores.values()]
    positivos = [a for a in valores if a.size]
    if not positivos:
        return 1.0
    percentil = float(np.percentile(np.concatenate(positivos), p))
    return max(percentil, 1e-6)


def panel(datos: dict[str,np.ndarray], caso: dict, instante: int,
          destino: Path, percentil: float, gamma: float,
          error_max_manual: float | None, umbral_agua: float,
          guardar_pdf: bool) -> None:
    gt = datos["gt"]
    metodos = ("unet", "bicubic", "subgrid")
    errores = {nombre: 1000 * np.abs(datos[nombre] - gt) for nombre in metodos}

    valores_agua = np.concatenate([
        a[np.isfinite(a)] for a in datos.values()
    ])
    if not valores_agua.size:
        raise ValueError("No hay profundidades validas dentro de la mascara")
    max_agua = max(float(np.max(valores_agua)), 1e-6)
    vmax_error = (error_max_manual if error_max_manual is not None
                  else percentil_robusto(errores, percentil))
    if vmax_error <= 0:
        raise ValueError("El maximo de error debe ser positivo")

    cmap_agua = plt.get_cmap("viridis").copy()
    cmap_agua.set_bad("white")
    cmap_error = plt.get_cmap("Reds").copy()
    cmap_error.set_bad("white")
    norma_agua = Normalize(vmin=0, vmax=max_agua)
    norma_error = PowerNorm(gamma=gamma, vmin=0, vmax=vmax_error, clip=True)

    fig, axs = plt.subplots(2, 4, figsize=(15, 7.1))
    fig.subplots_adjust(left=.03, right=.865, bottom=.06, top=.87,
                        wspace=.09, hspace=.29)
    extent = (X_MIN, X_MIN + 2920.0, Y_MAX - 5480.0, Y_MAX)
    etiquetas_agua = (
        ("coarse", "Entrada de baja resolucion"),
        ("unet", "U-Net"),
        ("bicubic", "Interpolacion bicubica"),
        ("subgrid", "Subgrid"),
    )
    for col, (nombre, etiqueta) in enumerate(etiquetas_agua):
        matriz = datos[nombre]
        visible = np.ma.masked_invalid(matriz)
        visible = np.ma.masked_where(matriz <= umbral_agua, visible)
        axs[0,col].imshow(visible, cmap=cmap_agua, norm=norma_agua,
                          extent=extent, origin="upper", interpolation="nearest", aspect="equal")
        axs[0,col].set_axis_off()

    axs[1,0].set_axis_off()
    for col,(nombre,etiqueta) in enumerate((
        ("unet", "U-Net"), ("bicubic", "bicubica"), ("subgrid", "subgrid")
    ), start=1):
        imagen = np.ma.masked_invalid(errores[nombre])
        imagen = np.ma.masked_less(imagen, 10.0)
        axs[1,col].imshow(imagen, cmap=cmap_error, norm=norma_error,
                          extent=extent, origin="upper", interpolation="nearest", aspect="equal")
        axs[1,col].set_axis_off()

    cax_agua = fig.add_axes([.89, .55, .016, .30])
    cb_agua = fig.colorbar(plt.cm.ScalarMappable(norm=norma_agua, cmap=cmap_agua), cax=cax_agua)
    cb_agua.set_label("Profundidad de agua (m)", fontsize=14)
    cb_agua.ax.tick_params(labelsize=13)
    cax_error = fig.add_axes([.89, .105, .016, .30])
    cb_error = fig.colorbar(plt.cm.ScalarMappable(norm=norma_error, cmap=cmap_error), cax=cax_error)
    cb_error.set_label("Error absoluto (mm)", fontsize=14)
    cb_error.ax.tick_params(labelsize=13)

    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=320, bbox_inches="tight", facecolor="white")
    if guardar_pdf:
        fig.savefig(destino.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[OK] {destino}")
    print(f"     Profundidades: 0 a {max_agua:.3f} m")
    print(f"     Color error: 0 a {vmax_error:.2f} mm "
          f"({'manual' if error_max_manual is not None else f'percentil {percentil:g} de errores >0.1 mm'}; gamma={gamma:g})")
    for nombre, valores in errores.items():
        validos = valores[np.isfinite(valores)]
        print(f"     {nombre}: maximo real={validos.max():.1f} mm, "
              f"saturados={100*np.mean(validos>vmax_error):.2f}% de celdas validas")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carpeta", type=Path, default=CARPETA_RAIZ,
                        help="Carpeta resultados_sel con subcarpetas 20to10/40to10")
    parser.add_argument("--instante", type=int, default=INSTANTE)
    parser.add_argument("--shapefile", type=Path,
                        default=DIRECTORIO_PROYECTO / "dominio_barranco.shp")
    parser.add_argument("--dem10", type=Path, default=DIRECTORIO_PROYECTO / "TFG" / "DEM_BM_10m.asc")
    parser.add_argument("--dem20", type=Path, default=DIRECTORIO_PROYECTO / "TFG" / "DEM_BM_20m.asc")
    parser.add_argument("--dem40", type=Path, default=DIRECTORIO_PROYECTO / "TFG" / "DEM_BM_40m.asc")
    parser.add_argument("--error-percentil", type=float, default=ERROR_PERCENTIL)
    parser.add_argument("--error-gamma", type=float, default=ERROR_GAMMA)
    parser.add_argument("--error-max-mm", type=float, default=None,
                        help="Maximo manual de la barra de error (mm)")
    parser.add_argument("--agua-min-m", type=float, default=UMBRAL_AGUA_M,
                        help="Umbral de visualizacion de agua; no cambia los errores")
    parser.add_argument("--pdf", action="store_true", help="Guardar tambien PDF vectorial")
    args = parser.parse_args()

    if not 0 < args.error_percentil <= 100:
        parser.error("--error-percentil debe pertenecer a (0, 100]")
    if not 0 < args.error_gamma <= 1:
        parser.error("--error-gamma debe pertenecer a (0, 1]")

    mascaras = preparar_mascaras(args.shapefile, {10:args.dem10, 20:args.dem20, 40:args.dem40})
    for caso in CASOS:
        archivo = localizar_npz(args.carpeta / caso["carpeta"], caso["modelo"], args.instante)
        print(f"[NPZ] {archivo}")
        datos = leer_npz(archivo, caso["resolucion"])
        datos = aplicar_mascaras(datos, caso["resolucion"], mascaras)
        destino = args.carpeta / "comparativas" / caso["salida"]
        panel(datos, caso, args.instante, destino, args.error_percentil,
              args.error_gamma, args.error_max_mm, args.agua_min_m, args.pdf)
    print("Listo. Los archivos .npz originales no se han modificado.")

if __name__ == "__main__":
    main()
