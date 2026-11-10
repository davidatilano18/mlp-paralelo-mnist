"""Experimento principal: escalabilidad con 1, 2, 4 y 8 hilos.

Dos cargas de trabajo:
  matmul : una multiplicación n x n (por defecto 1024) con la versión por
           bloques. Es casi 100 % paralelizable: muestra el "techo" del
           algoritmo.
  mlp    : UNA época completa de entrenamiento en MNIST con el backend por
           bloques. Además del matmul (paralelo) incluye partes que NO
           paralelizamos: ReLU, softmax, actualización de pesos con NumPy,
           barajado y el propio ciclo de Python. Aquí aparece Amdahl.

Para cada carga se mide la versión ingenua (línea base secuencial) y la
de bloques con cada número de hilos. Como referencia también NumPy y
GPU (si hay), que no dependen de numba.set_num_threads.

Métricas (con las medias):
    speedup relativo   S(p) = T_bloques(1) / T_bloques(p)   -> Amdahl
    eficiencia         E(p) = S(p) / p
    speedup absoluto   T_ingenua / T_bloques(p)   (algoritmo + paralelismo)
    f                  fracción paralela (Ley de Amdahl), puntual y ajustada

¿Por qué Amdahl usa el speedup RELATIVO? Amdahl describe cuánto mejora UN
MISMO programa al darle más procesadores. Comparar contra la ingenua
mezcla dos cosas (mejor algoritmo y más hilos) y daría f sin sentido.

Uso:    python -m experiments.bench_hilos [--reps 3] [--hilos 1 2 4 8] [--n 1024]
Salida: results/hilos_crudo.csv, results/hilos_resumen.csv, results/amdahl.csv
"""
import argparse
import sys
from pathlib import Path

import numba
import numpy as np
import pandas as pd

from experiments.medicion import amdahl_f_ajuste, amdahl_f_puntual, cronometrar
from src import datos
from src.matmul import BACKENDS, crear_matmul_bloques, matmul_bloques, matmul_ingenua
from src.mlp import MLP

DIR_RESULTADOS = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# Cargas de trabajo: cada una devuelve una función sin argumentos que
# ejecuta exactamente el mismo trabajo cada vez que se llama.
# ---------------------------------------------------------------------------
def carga_matmul(matmul, n):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n)).astype(np.float32)
    B = rng.standard_normal((n, n)).astype(np.float32)
    return lambda: matmul(A, B)


def carga_mlp(matmul, X, y, tam_lote, muestras_calentamiento=640):
    """Una época de entrenamiento. Cada llamada crea una red NUEVA con la
    misma semilla: así todas las repeticiones hacen el mismo trabajo (mismos
    pesos iniciales, mismo orden de lotes)."""
    def una_epoca():
        red = MLP(matmul=matmul, semilla=0)
        red.entrenar(X, y, epocas=1, tam_lote=tam_lote, tasa=0.1, verbose=False)

    # Calentamiento corto (10 lotes) en lugar de una época entera: basta
    # para compilar todas las formas de matriz que aparecen en la red.
    MLP(matmul=matmul, semilla=0).entrenar(X[:muestras_calentamiento], y[:muestras_calentamiento],
                                           epocas=1, tam_lote=tam_lote, verbose=False)
    return una_epoca


def medir(carga, backend, hilos, f, reps, calentar, filas):
    numba.set_num_threads(hilos if hilos else numba.config.NUMBA_NUM_THREADS)
    if backend == "gpu":
        # Tras minutos de CPU al 100 % la GPU de una laptop arranca con el
        # reloj bajo; ver experiments/bench_gpu.py:llevar_a_reloj_de_trabajo.
        from numba import cuda

        from experiments.bench_gpu import llevar_a_reloj_de_trabajo
        llevar_a_reloj_de_trabajo(cuda)
    tiempos = cronometrar(f, reps=reps, calentar=calentar)
    etiqueta = f"{hilos} hilos" if hilos else "—"
    # Desviación estándar MUESTRAL (ddof=1), la misma que usa pandas en el
    # resumen: con pocas repeticiones es la estimación correcta.
    print(f"  {carga:6s} {backend:8s} {etiqueta:8s}  {np.mean(tiempos):8.4f} ± "
          f"{np.std(tiempos, ddof=1):.4f} s", flush=True)
    for r, t in enumerate(tiempos):
        filas.append(dict(carga=carga, backend=backend, hilos=hilos, rep=r, tiempo_s=t))


def resumir(crudo):
    """De los tiempos crudos a la tabla de métricas por (carga, hilos)."""
    g = (crudo.groupby(["carga", "backend", "hilos"], dropna=False)["tiempo_s"]
         .agg(media="mean", std="std", reps="count").reset_index())
    filas, amdahl = [], []
    for carga, d in g.groupby("carga"):
        ingenua = d.loc[d.backend == "ingenua", "media"].iloc[0]
        par = d[d.backend == "bloques"].sort_values("hilos")
        T1 = par.loc[par.hilos == 1, "media"].iloc[0]
        for _, fila in d.iterrows():
            fila = fila.to_dict()
            if fila["backend"] == "bloques":
                p = int(fila["hilos"])
                S = T1 / fila["media"]
                fila.update(speedup=S, eficiencia=S / p, f_puntual=amdahl_f_puntual(S, p))
            fila["speedup_vs_ingenua"] = ingenua / fila["media"]
            filas.append(fila)
        ps = par.hilos.to_numpy(int)
        Ss = T1 / par.media.to_numpy()
        if (ps > 1).any():
            f = amdahl_f_ajuste(ps, Ss)
            amdahl.append(dict(carga=carga, f_ajuste=f,
                               speedup_maximo=1 / (1 - f) if f < 1 else np.inf))
    return pd.DataFrame(filas), pd.DataFrame(amdahl)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reps", type=int, default=3, help="repeticiones (mínimo 3)")
    p.add_argument("--hilos", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--n", type=int, default=1024, help="tamaño de la matmul")
    p.add_argument("--bloque", type=int, default=32)
    p.add_argument("--lote", type=int, default=64)
    p.add_argument("--muestras", type=int, default=None,
                   help="usar solo las primeras N muestras de MNIST (pruebas rápidas)")
    p.add_argument("--cargas", nargs="+", choices=["matmul", "mlp"], default=["matmul", "mlp"])
    args = p.parse_args(argv)

    maximo = numba.config.NUMBA_NUM_THREADS
    if max(args.hilos) > maximo:
        sys.exit(f"Este equipo solo permite hasta {maximo} hilos de Numba.")
    if 1 not in args.hilos:
        sys.exit("Se necesita la medición con 1 hilo para calcular el speedup.")

    bloques = crear_matmul_bloques(args.bloque)
    referencias = {k: v for k, v in BACKENDS.items() if k in ("numpy", "gpu")}
    filas = []

    if "matmul" in args.cargas:
        print(f"\nCarga matmul: {args.n} x {args.n}, float32, bloque {args.bloque}")
        medir("matmul", "ingenua", 1, carga_matmul(matmul_ingenua, args.n), args.reps, True, filas)
        for h in args.hilos:
            medir("matmul", "bloques", h, carga_matmul(bloques, args.n), args.reps, True, filas)
        for nombre, f in referencias.items():
            medir("matmul", nombre, None, carga_matmul(f, args.n), args.reps, True, filas)

    if "mlp" in args.cargas:
        X, y, _, _ = datos.cargar("mnist")
        if args.muestras:
            X, y = X[:args.muestras], y[:args.muestras]
        print(f"\nCarga mlp: 1 época, {X.shape[0]} muestras, lote {args.lote}, bloque {args.bloque}")
        medir("mlp", "ingenua", 1, carga_mlp(matmul_ingenua, X, y, args.lote), args.reps, False, filas)
        for h in args.hilos:
            numba.set_num_threads(h)
            medir("mlp", "bloques", h, carga_mlp(bloques, X, y, args.lote), args.reps, False, filas)
        for nombre, f in referencias.items():
            medir("mlp", nombre, None, carga_mlp(f, X, y, args.lote), args.reps, False, filas)

    numba.set_num_threads(maximo)
    crudo = pd.DataFrame(filas)
    resumen, amdahl = resumir(crudo)

    DIR_RESULTADOS.mkdir(exist_ok=True)
    crudo.to_csv(DIR_RESULTADOS / "hilos_crudo.csv", index=False)
    resumen.to_csv(DIR_RESULTADOS / "hilos_resumen.csv", index=False)
    amdahl.to_csv(DIR_RESULTADOS / "amdahl.csv", index=False)

    pd.set_option("display.width", 140)
    print("\n" + resumen.round(4).to_string(index=False))
    print("\n" + amdahl.round(4).to_string(index=False))
    print("\nGuardado en results/hilos_crudo.csv, hilos_resumen.csv y amdahl.csv")


if __name__ == "__main__":
    main()
