"""Entrena el MLP con el backend elegido y reporta precisión y tiempo.

Uso (desde la raíz del repo):
    python -m experiments.entrenar                        # MNIST, backend numpy
    python -m experiments.entrenar --backend ingenua
    python -m experiments.entrenar --dataset fashion --epocas 15

Guarda un resumen en results/entrenamiento_<dataset>_<backend>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numba
import numpy as np

from src import datos
from src.matmul import (BACKENDS, TAM_BLOQUE_DEFECTO, crear_matmul_bloques,
                        obtener_matmul)
from src.mlp import MLP

DIR_RESULTADOS = Path(__file__).resolve().parent.parent / "results"


def main():
    # La consola de Windows no usa UTF-8 por defecto y los acentos salen mal.
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=list(datos.CONJUNTOS), default="mnist")
    p.add_argument("--backend", choices=list(BACKENDS), default="numpy")
    p.add_argument("--epocas", type=int, default=10)
    p.add_argument("--lote", type=int, default=64)
    p.add_argument("--tasa", type=float, default=0.1, help="tasa de aprendizaje")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--bloque", type=int, default=TAM_BLOQUE_DEFECTO,
                   help="tamaño de bloque (solo backend 'bloques')")
    p.add_argument("--hilos", type=int, default=numba.config.NUMBA_NUM_THREADS,
                   help="hilos de Numba (solo backend 'bloques')")
    args = p.parse_args()
    numba.set_num_threads(args.hilos)

    X, y, X_prueba, y_prueba = datos.cargar(args.dataset)
    print(f"{args.dataset}: {X.shape[0]} entrenamiento, {X_prueba.shape[0]} prueba")

    if args.backend == "bloques":
        matmul = crear_matmul_bloques(args.bloque)
        print(f"Backend bloques: tamaño de bloque {args.bloque}, {args.hilos} hilos")
    else:
        matmul = obtener_matmul(args.backend)
    # Calentamiento: la primera llamada a una función de Numba la compila.
    # La hacemos aquí para que el tiempo de compilación no se cuente como
    # tiempo de entrenamiento.
    matmul(np.ones((2, 2), np.float32), np.ones((2, 2), np.float32))

    red = MLP(matmul=matmul, semilla=args.semilla)
    t0 = time.perf_counter()
    hist = red.entrenar(X, y, epocas=args.epocas, tam_lote=args.lote, tasa=args.tasa,
                        X_val=X_prueba, y_val=y_prueba)
    t_total = time.perf_counter() - t0
    # Nota: este tiempo incluye evaluar en el conjunto de prueba al final de
    # cada época. Las mediciones formales de rendimiento (etapa 5) se harán
    # aparte y sin esa evaluación.

    acc = red.precision(X_prueba, y_prueba)
    print(f"\nPrecisión final en prueba: {acc:.2%}   tiempo total: {t_total:.1f} s "
          f"({t_total / args.epocas:.1f} s/época)")

    DIR_RESULTADOS.mkdir(exist_ok=True)
    salida = DIR_RESULTADOS / f"entrenamiento_{args.dataset}_{args.backend}.json"
    salida.write_text(json.dumps({**vars(args), "precision_prueba": acc,
                                  "tiempo_total_s": t_total, **hist}, indent=2))
    print(f"Resumen guardado en {salida.relative_to(DIR_RESULTADOS.parent)}")


if __name__ == "__main__":
    main()
