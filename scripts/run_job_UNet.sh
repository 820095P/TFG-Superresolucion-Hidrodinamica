#!/bin/bash
set -euo pipefail # evita que el script continue si hay errores, variables no definidas o fallos en pipes
 
############################
# CONFIGURACIÓN FIJA
############################

# GPU a usar (0–3)
GPU_ID=2

# Directorio base de código de la UNet
UNET_CODE_DIR="/home/colabghc/pbenito/UNet"

# Directorio base de datos de la UNet
UNET_DATA_DIR="/working_disk/pbenito"

# main
UNET_MAIN="main.py"

# CFG por defecto
DEFAULT_CFG="${UNET_DATA_DIR}/cfg/Unet_prueba_1.txt"

############################
# ARGUMENTOS
############################

# CFG opcional por línea de comandos
CFG_FILE=${1:-$DEFAULT_CFG}

############################
# COMPROBACIONES
############################

if [ ! -d "$UNET_CODE_DIR" ]; then
    echo "Error: no existe $UNET_CODE_DIR"
    exit 1
fi

if [ ! -f "${UNET_CODE_DIR}/${UNET_MAIN}" ]; then
    echo "Error: main.py no existe"
    exit 1
fi

if [ ! -f "$CFG_FILE" ]; then
    echo "Error: no existe el fichero CFG $CFG_FILE"
    exit 1
fi

# ¿Está libre la GPU?
GPU_JOBS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU_ID)
if [ -n "$GPU_JOBS" ]; then
    echo "Error: la GPU $GPU_ID está ocupada"
    echo "PIDs activos: $GPU_JOBS"
    exit 1
fi

############################
# EJECUCIÓN
############################
LOG_FILE="${UNET_DATA_DIR}/logs/UNet_gpu${GPU_ID}_$(date +%Y%m%d_%H%M).log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "=================================="
echo "Lanzando UNet"
echo "GPU física: $GPU_ID"
echo "CFG: $CFG_FILE"
echo "Log: $LOG_FILE"
echo "=================================="

cd "$UNET_CODE_DIR" || exit 1

# Forzar GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Activar entorno virtual con librerías necesasrias
VENV_PATH="${UNET_CODE_DIR}/venv_Unet_pytorch/bin/activate"
if [ ! -f "$VENV_PATH" ]; then
  echo "Error: no se encuentra el venv en $VENV_PATH"
  exit 1
fi
source "$VENV_PATH"

# Ejecutar en background (nohup para que siga corriendo si cerramos la terminal, -u para que la salida sea no-buffered)
MASTER_PORT=$((29500+GPU_ID))
nohup python -u main.py --config "$CFG_FILE" --master_addr 127.0.0.1 --master_port "$MASTER_PORT" > "$LOG_FILE" 2>&1 &

JOB_PID=$!
echo "Trabajo lanzado con PID $JOB_PID"
echo "Para matarlo: kill -9 $JOB_PID"

exit 0

