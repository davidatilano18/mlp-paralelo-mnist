"""Reto extra: ingenua vs. por bloques con UN SOLO HILO.

Con un hilo no hay paralelismo, así que cualquier diferencia de tiempo
viene de CÓMO se recorre la memoria. Se comparan tres variantes:

    ingenua : i-j-k. Lee B por columnas (saltos de q elementos).
    ikj     : i-k-j SIN bloques. Lee B y C por filas (contiguo, SIMD),
              pero para cada fila de C recorre toda B: si B no cabe en
              caché, se vuelve a traer de la RAM una y otra vez.
    bloques : i-k-j CON bloques. Además reutiliza cada bloque mientras
              está en caché. Se prueban varios tamaños de bloque.

    ingenua -> ikj      = efecto del orden de acceso (líneas de caché + SIMD)
    ikj     -> bloques  = efecto de la reutilización (localidad temporal)

Se reporta GFLOP/s = 2·n³ / tiempo, que normaliza por el tamaño: si el
rendimiento cae al crecer n, es porque los datos dejaron de caber en
alguna caché (L1 ~48 KB, L2 ~1-2 MB, L3 ~12 MB en este tipo de CPU).

Uso:    python -m experiments.bench_cache [--reps 3] [--tamanos ...] [--bloques ...]
Salida: results/cache.csv
"""
import argparse
import sys
from pathlib import Path

import numba
import numpy as np
import pandas as pd

from experiments.medicion import cronometrar
from src.matmul import matmul_bloques, matmul_ikj, matmul_ingenua

DIR_RESULTADOS = Path(__file__).resolve().parent.parent / "results"


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reps", type=int, default=3)
    # 768 no es potencia de 2 a propósito: con potencias de 2, las columnas
    # de B caen en el mismo "conjunto" de la caché y chocan entre sí
    # (conflictos de asociatividad), lo que castiga extra a la ingenua.
    p.add_argument("--tamanos", type=int, nargs="+", default=[128, 256, 512, 768, 1024])
    p.add_argument("--bloques", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    args = p.parse_args(argv)

    numba.set_num_threads(1)  # ¡UN SOLO HILO en todo el experimento!
    rng = np.random.default_rng(0)
    filas = []

    def registrar(variante, n, bs, tiempos):
        media = float(np.mean(tiempos))
        filas.append(dict(variante=variante, n=n, bloque=bs, media_s=media,
                          std_s=float(np.std(tiempos, ddof=1)) if len(tiempos) > 1 else 0.0,
                          gflops=2 * n**3 / media / 1e9))
        etiqueta = f"{variante}" + (f" bs={bs}" if bs else "")
        print(f"  n={n:5d}  {etiqueta:14s} {media * 1e3:10.2f} ms  {filas[-1]['gflops']:6.2f} GFLOP/s",
              flush=True)

    for n in args.tamanos:
        A = rng.standard_normal((n, n)).astype(np.float32)
        B = rng.standard_normal((n, n)).astype(np.float32)
        registrar("ingenua", n, None, cronometrar(lambda: matmul_ingenua(A, B), args.reps))
        registrar("ikj", n, None, cronometrar(lambda: matmul_ikj(A, B), args.reps))
        for bs in args.bloques:
            registrar("bloques", n, bs, cronometrar(lambda: matmul_bloques(A, B, bs), args.reps))

    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    DIR_RESULTADOS.mkdir(exist_ok=True)
    pd.DataFrame(filas).to_csv(DIR_RESULTADOS / "cache.csv", index=False)
    print("\nGuardado en results/cache.csv")


if __name__ == "__main__":
    main()
