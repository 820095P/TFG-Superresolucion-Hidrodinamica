"""Crea una copia de profundidades con cero fuera del DEM. Python 3 + numpy."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from datetime import datetime, timezone

FOLDERS = {'h_coarse20': 20, 'h_coarse20_real_case': 20,
           'h_coarse40': 40, 'h_coarse40_real_case': 40,
           'h_fine10': 10, 'h_fine10_real_case': 10}
DEFAULT_SOURCE = r'C:\Users\PC\Desktop\TFG\ProyectoTFG\UNet_data_txt'

def load_dem(path):
    import numpy as np
    with path.open(encoding='utf-8-sig') as f:
        header = dict(line.split() for line in [next(f) for _ in range(6)])
    header = {k.lower(): float(v) for k, v in header.items()}
    dem = np.loadtxt(path, skiprows=6, ndmin=2)
    if dem.shape != (int(header['nrows']), int(header['ncols'])):
        raise ValueError(f'Dimensiones incorrectas en {path}')
    mask = np.isfinite(dem) & (dem != header['nodata_value'])
    if not mask.any() or mask.all():
        raise ValueError(f'Mascara vacia o sin exterior: {path}')
    return mask, header

def process_file(src, dst, mask, cellsize):
    import numpy as np
    raw = src.read_bytes()
    text = raw.decode('utf-8-sig')
    a = np.loadtxt(src, encoding='utf-8-sig', ndmin=2)
    if a.shape != mask.shape:
        raise ValueError(f'{src}: matriz {a.shape}, DEM {mask.shape}. No se recorta ni transpone automaticamente.')
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError(f'{src}: profundidades negativas o no finitas; revisar antes de procesar.')
    # Mantiene literalmente numeros y separadores interiores; solo reemplaza
    # tokens numericos exteriores. No hay redondeo de las celdas validas.
    lines = text.splitlines(keepends=True)
    out, row = [], 0
    for line in lines:
        tokens = list(re.finditer(r'\S+', line))
        if not tokens:
            out.append(line)
            continue
        if row >= mask.shape[0] or len(tokens) != mask.shape[1]:
            raise ValueError(f'{src}: formato distinto de matriz TXT sin cabecera.')
        col = iter(range(mask.shape[1]))
        def replace(match):
            c = next(col)
            return match.group() if mask[row, c] else '0.000000'
        out.append(re.sub(r'\S+', replace, line))
        row += 1
    if row != mask.shape[0]:
        raise ValueError(f'{src}: numero de filas incorrecto.')
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(''.join(out).encode('utf-8'))
    b = np.loadtxt(dst, ndmin=2)
    if not np.array_equal(a[mask], b[mask]) or np.any(b[~mask] != 0):
        raise ValueError(f'Fallo de verificacion en {dst}')
    return {'archivo': str(src), 'filas': a.shape[0], 'columnas': a.shape[1],
            'celdas_validas': int(mask.sum()),
            'celdas_exteriores_modificadas': int(np.count_nonzero(a[~mask])),
            'celdas_exteriores_humedas_001m': int(np.count_nonzero(a[~mask] >= .01)),
            'volumen_exterior_retirado_m3': float(a[~mask].sum() * cellsize**2),
            'sha256_original': hashlib.sha256(raw).hexdigest(),
            'sha256_salida': hashlib.sha256(dst.read_bytes()).hexdigest()}

def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entrada', default=DEFAULT_SOURCE)
    parser.add_argument('--salida', help='Por defecto: carpeta hermana con sufijo _postprocesados')
    parser.add_argument('--dem-dir', type=Path, default=Path(__file__).resolve().parent / 'dems')
    args = parser.parse_args()
    source = Path(args.entrada).resolve()
    target = Path(args.salida).resolve() if args.salida else source.with_name(source.name + '_postprocesados')
    if not source.is_dir():
        raise ValueError(f'No existe la carpeta de entrada: {source}')
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('Entrada y salida deben ser carpetas independientes, sin estar una dentro de otra.')
    if target.exists():
        raise ValueError(f'La salida ya existe: {target}. Usa otra ruta para no mezclar ejecuciones.')
    masks, headers = {}, {}
    for resolution in (10, 20, 40):
        masks[resolution], headers[resolution] = load_dem(args.dem_dir / f'DEM_BM_{resolution}m.asc')
        if headers[resolution]['cellsize'] != resolution:
            raise ValueError('Resolucion DEM incorrecta.')
        print(f'DEM {resolution} m: {masks[resolution].shape}, {int(masks[resolution].sum())} celdas validas')
    for folder in FOLDERS:
        if not (source / folder).is_dir() or not list((source / folder).rglob('*.txt')):
            raise ValueError(f'Falta carpeta o archivos TXT: {folder}')
    files = sorted(source.rglob('*'))
    if any(p.is_symlink() for p in files):
        raise ValueError('Hay enlaces simbolicos en la entrada; revisar estructura.')
    target.mkdir()
    audit = target / '_postprocesado'
    audit.mkdir()
    metadata = {'estado': 'EN_CURSO', 'inicio_utc': datetime.now(timezone.utc).isoformat(),
                'entrada': str(source), 'salida': str(target),
                'operacion': 'Profundidad cero fuera del DEM propio de cada resolucion; interior sin cambios.',
                'nota': 'Postprocesamiento de salidas. No modifica la simulacion ni valida su fisica.',
                'dems': {str(r): {'cabecera': headers[r], 'celdas_validas': int(masks[r].sum()),
                     'sha256': hashlib.sha256((args.dem_dir / f'DEM_BM_{r}m.asc').read_bytes()).hexdigest()}
                     for r in masks}}
    def save_metadata():
        (audit / 'resumen.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    save_metadata()
    processed = copied = 0
    try:
        for r in masks:
            shutil.copy2(args.dem_dir / f'DEM_BM_{r}m.asc', audit / f'DEM_BM_{r}m.asc')
        with (audit / 'archivos.csv').open('w', newline='', encoding='utf-8-sig') as log:
            writer = None
            for path in files:
                rel = path.relative_to(source)
                dest = target / rel
                if path.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                res = FOLDERS.get(rel.parts[0])
                if res and path.suffix.lower() == '.txt':
                    record = process_file(path, dest, masks[res], res)
                    record['archivo'] = rel.as_posix()
                    if writer is None:
                        writer = csv.DictWriter(log, fieldnames=list(record))
                        writer.writeheader()
                    writer.writerow(record)
                    log.flush()
                    processed += 1
                    if processed % 50 == 0:
                        print(f'{processed} campos procesados y verificados...', flush=True)
                else:
                    if path.suffix.lower() == '.txt' and re.search(r'H_\d+', path.name, re.I):
                        raise ValueError(f'Campo H en carpeta no reconocida: {rel}')
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)
                    copied += 1
        metadata.update(estado='COMPLETADO', archivos_procesados=processed, archivos_copiados=copied)
        print(f'COMPLETADO: {processed} campos verificados. Salida: {target}')
    except Exception as exc:
        metadata.update(estado='ERROR_INCOMPLETO', error=str(exc))
        raise
    finally:
        save_metadata()

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
