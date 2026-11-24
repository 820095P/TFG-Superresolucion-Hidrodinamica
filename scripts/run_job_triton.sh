#!/bin/bash

# GPU a usar (0–3)
GPU_ID=2

# Directorios
BASE_DIR="/home/colabghc/pbenito"
TRITON_EXE="${BASE_DIR}/triton/build/triton.exe" # Quitamos el .exe si el archivo se llama solo triton

# Argumento: CFG (si no se da, falla)
if [ -z "$1" ]; then
    echo "Error: Debes pasar un fichero .cfg como argumento"
    exit 1
fi

CFG_FILE=$(readlink -f "$1") # Convierte a ruta absoluta real

# Comprobaciones
if [ ! -f "$TRITON_EXE" ]; then
    # Intenta con .exe por si acaso
    TRITON_EXE="${TRITON_EXE}.exe"
    if [ ! -f "$TRITON_EXE" ]; then
        echo "Error: No se encuentra el ejecutable triton en build/"
        exit 1
    fi
fi

if [ ! -f "$CFG_FILE" ]; then
    echo "Error: No existe el CFG $CFG_FILE"
    exit 1
fi

# Log con ruta fija para que no se pierda
LOG_FILE="${BASE_DIR}/triton_gpu${GPU_ID}_$(date +%Y%m%d_%H%M).log"
TIME_FILE="${BASE_DIR}/triton_gpu${GPU_ID}_$(date +%Y%m%d_%H%M)_time.txt"

echo "=================================="
echo "Lanzando Triton"
echo "GPU: $GPU_ID | CFG: $CFG_FILE"
echo "Log: $LOG_FILE"
echo "Tiempo: $TIME_FILE"
echo "=================================="

# Forzar GPU y ejecutar
export CUDA_VISIBLE_DEVICES=$GPU_ID

/usr/bin/time -v -o "$TIME_FILE" \
mpirun -n 8 -x CUDA_VISIBLE_DEVICES=$GPU_ID "$TRITON_EXE" "$CFG_FILE" \
> "$LOG_FILE" 2>&1 &

JOB_PID=$!
echo "Trabajo lanzado con PID $JOB_PID"
exit 0


