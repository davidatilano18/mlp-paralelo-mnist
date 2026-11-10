# Imagen reproducible del proyecto (solo CPU).
#
# ¿Por qué solo CPU? Darle la GPU a un contenedor requiere el NVIDIA
# Container Toolkit (en Windows, además, WSL2 con soporte CUDA). Para la
# parte de GPU usamos el notebook de Google Colab (notebooks/), que es
# más sencillo de reproducir para cualquiera. Dentro del contenedor el
# backend "gpu" simplemente no aparece y sus tests se saltan.
#
# Construir:   docker build -t mlp-paralelo .
# Tests:       docker run --rm mlp-paralelo
# Ver README.md para los demás comandos.

# Imagen base oficial de Python, variante "slim" (Debian mínimo): pesa
# poco y tiene las bibliotecas de C que necesitan NumPy y Numba.
FROM python:3.13-slim

# - PYTHONDONTWRITEBYTECODE: no ensuciar la imagen con archivos .pyc.
# - PYTHONUNBUFFERED: que los print salgan de inmediato en `docker run`.
# - MPLBACKEND=Agg: matplotlib genera PNG sin intentar abrir ventanas.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Primero SOLO requirements.txt e instalar: Docker guarda cada paso en
# caché, así que si después cambiamos el código (pero no las dependencias)
# no vuelve a descargar todo.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ahora sí, el código del proyecto.
COPY . .

# Por defecto: correr los tests de correctitud.
CMD ["python", "-m", "pytest", "-q"]
