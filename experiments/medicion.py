"""Utilidades de medición y métricas compartidas por todos los benchmarks.

Reglas de medición (las mismas en todos los experimentos):
    1. Una llamada de CALENTAMIENTO antes de medir: la primera llamada a
       una función de Numba la compila (JIT) y eso no es rendimiento del
       algoritmo. También "calienta" cachés y el reparto de hilos.
    2. Cronómetro: time.perf_counter, el reloj de mayor resolución
       disponible y que no salta si cambia la hora del sistema.
    3. Varias repeticiones (mínimo 3); se reporta media y desviación
       estándar para saber qué tan confiable es cada número.
"""
import time

import numpy as np


def cronometrar(f, reps=3, calentar=True):
    """Ejecuta f() `reps` veces y devuelve la lista de tiempos en segundos."""
    if calentar:
        f()
    tiempos = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        tiempos.append(time.perf_counter() - t0)
    return tiempos


def amdahl_f_puntual(S, p):
    """Fracción paralela f despejada de la Ley de Amdahl para UNA medición.

    Amdahl:   S(p) = 1 / ((1 - f) + f / p)
    Despeje:  1/S = 1 - f + f/p = 1 - f (1 - 1/p)
              f = (1 - 1/S) / (1 - 1/p)

    Con p = 1 no está definida (0/0): con un solo hilo no se puede
    distinguir la parte paralela de la serial.
    """
    if p == 1:
        return np.nan
    return (1 - 1 / S) / (1 - 1 / p)


def amdahl_f_ajuste(ps, Ss):
    """Mejor f para TODAS las mediciones juntas (mínimos cuadrados).

    Reescribiendo Amdahl:  1/S - 1 = f · (1/p - 1)
    es una recta que pasa por el origen con x = 1/p - 1, y = 1/S - 1,
    cuya pendiente es f. La pendiente de mínimos cuadrados por el origen
    es  f = sum(x·y) / sum(x²).  Un solo f para todos los p es más robusto
    que los valores puntuales, que dependen del ruido de cada medición.
    """
    ps, Ss = np.asarray(ps, float), np.asarray(Ss, float)
    x = 1 / ps - 1
    y = 1 / Ss - 1
    mascara = ps > 1
    return float(np.sum(x[mascara] * y[mascara]) / np.sum(x[mascara] ** 2))


def speedup_amdahl(f, p):
    """Speedup que predice Amdahl para una fracción paralela f con p hilos."""
    p = np.asarray(p, float)
    return 1 / ((1 - f) + f / p)
