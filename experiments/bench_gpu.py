"""Benchmark de la versión GPU: separa el cómputo de las copias CPU<->GPU.

Para cada tamaño n se mide (matrices n x n, float32):
    total   = copiar A y B a la GPU + kernel + copiar C de regreso
    kernel  = solo el kernel, con los datos ya en la GPU
    copias  = total - kernel
    numpy   = A @ B en CPU (BLAS), como referencia

Uso:  python -m experiments.bench_gpu [--reps 5] [--tamanos 64 256 1024]
Salida: results/bench_gpu.csv
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from src.matmul import matmul_numpy
from src.matmul_gpu import GPU_DISPONIBLE, _lanzar, calentar, matmul_gpu

DIR_RESULTADOS = Path(__file__).resolve().parent.parent / "results"


def medir(f, reps):
    """Media y desviación estándar en milisegundos de `reps` ejecuciones.
    Hace una llamada previa de calentamiento que NO se cuenta."""
    f()
    tiempos = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        tiempos.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(tiempos)), float(np.std(tiempos, ddof=1))  # desv. muestral


def llevar_a_reloj_de_trabajo(cuda, segundos=1.5):
    """Calentamiento SOSTENIDO de la GPU antes de medir.

    Una GPU en reposo baja su reloj para ahorrar energía y tarda en volver
    a su frecuencia de trabajo. En una laptop, además, CPU y GPU comparten
    el límite de potencia: justo después de minutos de CPU al 100 % la GPU
    arranca lenta. Lo observamos: en la corrida completa el kernel salió
    hasta 5x más lento que medido por separado. Con ~1.5 s de trabajo
    continuo la GPU sube su reloj y las mediciones se estabilizan.
    """
    n = 1024
    dA = cuda.to_device(np.ones((n, n), np.float32))
    dC = cuda.device_array((n, n), np.float32)
    fin = time.perf_counter() + segundos
    while time.perf_counter() < fin:
        _lanzar(dA, dA, dC)
        cuda.synchronize()


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--tamanos", type=int, nargs="+", default=[64, 256, 512, 1024, 2048, 4096])
    args = p.parse_args(argv)

    if not GPU_DISPONIBLE:
        sys.exit("No hay GPU CUDA disponible.")
    from numba import cuda

    calentar()  # compila los kernels antes de medir (excluye el JIT)
    llevar_a_reloj_de_trabajo(cuda)
    nombre = cuda.get_current_device().name
    # Según la versión de numba-cuda el nombre llega como bytes o como str
    print(f"GPU: {nombre.decode() if isinstance(nombre, bytes) else nombre}")
    rng = np.random.default_rng(0)
    filas = []
    for n in args.tamanos:
        A = rng.standard_normal((n, n)).astype(np.float32)
        B = rng.standard_normal((n, n)).astype(np.float32)
        dA, dB = cuda.to_device(A), cuda.to_device(B)
        dC = cuda.device_array((n, n), dtype=np.float32)

        def solo_kernel():
            _lanzar(dA, dB, dC)
            # El lanzamiento es asíncrono: sin synchronize() mediríamos solo
            # lo que tarda la CPU en "encargar" el trabajo, no el trabajo.
            cuda.synchronize()

        total, total_std = medir(lambda: matmul_gpu(A, B), args.reps)
        kernel, kernel_std = medir(solo_kernel, args.reps)
        ref, ref_std = medir(lambda: matmul_numpy(A, B), args.reps)
        # GFLOP/s: una matmul n x n x n hace 2·n³ operaciones (mult. + suma)
        gflops = 2 * n**3 / (kernel / 1e3) / 1e9
        filas.append(dict(n=n, total_ms=total, total_std=total_std,
                          kernel_ms=kernel, kernel_std=kernel_std,
                          copias_ms=total - kernel, numpy_ms=ref, numpy_std=ref_std,
                          kernel_gflops=gflops))
        print(f"n={n:5d}  total {total:9.2f} ± {total_std:6.2f} ms | kernel {kernel:9.2f} ms "
              f"({gflops:7.1f} GFLOP/s) | copias {total - kernel:8.2f} ms | numpy {ref:9.2f} ms")

    DIR_RESULTADOS.mkdir(exist_ok=True)
    salida = DIR_RESULTADOS / "bench_gpu.csv"
    with open(salida, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"Guardado en {salida.relative_to(DIR_RESULTADOS.parent)}")


if __name__ == "__main__":
    main()
