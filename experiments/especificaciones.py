"""Registra las especificaciones del equipo donde se corren los experimentos.

Sin esto, los tiempos no se pueden interpretar ni comparar: el mismo
código tarda distinto en otra CPU, con otra cantidad de núcleos o RAM.

Uso:    python -m experiments.especificaciones
Salida: results/especificaciones.json (y se imprime en pantalla)
"""
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numba
import numpy as np
import psutil

DIR_RESULTADOS = Path(__file__).resolve().parent.parent / "results"


def _ejecutar(cmd):
    """Corre un comando del sistema y devuelve su salida, o None si falla."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20,
                              check=True).stdout.strip()
    except Exception:
        return None


def nombre_cpu():
    """platform.processor() da textos genéricos en Windows ("Intel64
    Family 6..."), así que pedimos el nombre comercial a cada sistema."""
    sistema = platform.system()
    if sistema == "Windows":
        nombre = _ejecutar(["powershell", "-NoProfile", "-Command",
                            "(Get-CimInstance Win32_Processor).Name"])
    elif sistema == "Linux":
        try:
            lineas = Path("/proc/cpuinfo").read_text().splitlines()
            nombre = next(l.split(":", 1)[1].strip() for l in lineas if l.startswith("model name"))
        except Exception:
            nombre = None
    elif sistema == "Darwin":
        nombre = _ejecutar(["sysctl", "-n", "machdep.cpu.brand_string"])
    else:
        nombre = None
    return nombre or platform.processor() or "desconocido"


def info_gpu():
    """Datos de la GPU NVIDIA: primero con nvidia-smi y, si no existe,
    con lo que reporte numba.cuda."""
    gpus = []
    if shutil.which("nvidia-smi"):
        salida = _ejecutar(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                            "--format=csv,noheader"])
        for linea in (salida or "").splitlines():
            nombre, memoria, driver = [x.strip() for x in linea.split(",")]
            gpus.append({"nombre": nombre, "memoria": memoria, "driver": driver})
    try:
        from src.matmul_gpu import GPU_DISPONIBLE
        if GPU_DISPONIBLE:
            from numba import cuda
            disp = cuda.get_current_device()
            cc = disp.compute_capability
            n = disp.name
            datos_numba = {"nombre_numba": n.decode() if isinstance(n, bytes) else n,
                           "compute_capability": f"{cc[0]}.{cc[1]}",
                           "multiprocesadores": disp.MULTIPROCESSOR_COUNT}
            if gpus:
                gpus[0].update(datos_numba)
            else:
                gpus.append(datos_numba)
    except Exception:
        pass
    return gpus or None


def especificaciones():
    frecuencia = psutil.cpu_freq()
    from src.matmul_gpu import GPU_DISPONIBLE
    return {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "sistema_operativo": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": nombre_cpu(),
        # Núcleos físicos = unidades de cómputo reales.
        # Lógicos = físicos x hilos por núcleo (Hyper-Threading). Dos hilos
        # en el mismo núcleo físico comparten sus unidades de cálculo, así
        # que NO duplican el rendimiento en código numérico intensivo.
        "nucleos_fisicos": psutil.cpu_count(logical=False),
        "nucleos_logicos": psutil.cpu_count(logical=True),
        "frecuencia_max_mhz": round(frecuencia.max) if frecuencia else None,
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "gpu": info_gpu(),
        "gpu_usable_por_numba": GPU_DISPONIBLE,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "numba": numba.__version__,
        "numba_hilos_max": numba.config.NUMBA_NUM_THREADS,
        "numba_capa_hilos": numba.threading_layer() if _capa_inicializada() else "sin inicializar",
    }


def _capa_inicializada():
    """numba.threading_layer() solo funciona después de ejecutar algo en
    paralelo; corremos una función paralela mínima para inicializarla."""
    from src.matmul import matmul_bloques
    try:
        matmul_bloques(np.ones((4, 4)), np.ones((4, 4)))
        return True
    except Exception:
        return False


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    specs = especificaciones()
    DIR_RESULTADOS.mkdir(exist_ok=True)
    salida = DIR_RESULTADOS / "especificaciones.json"
    salida.write_text(json.dumps(specs, indent=2, ensure_ascii=False), encoding="utf-8")
    for clave, valor in specs.items():
        print(f"{clave:22s} {valor}")
    print(f"\nGuardado en {salida.relative_to(DIR_RESULTADOS.parent)}")


if __name__ == "__main__":
    main()
