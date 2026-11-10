"""Tests de las fórmulas de speedup y Ley de Amdahl (experiments/medicion.py).

Si estas fórmulas tuvieran un error, todas las conclusiones del reporte
estarían mal, así que también se prueban: generamos speedups "perfectos"
con una f conocida y verificamos que las funciones la recuperen.
"""
import numpy as np
import pytest

from experiments.medicion import (amdahl_f_ajuste, amdahl_f_puntual, cronometrar,
                                  speedup_amdahl)

HILOS = np.array([1, 2, 4, 8])


@pytest.mark.parametrize("f", [0.0, 0.5, 0.9, 0.99, 1.0])
def test_recupera_f_conocida(f):
    S = speedup_amdahl(f, HILOS)
    for p, s in zip(HILOS[1:], S[1:]):
        assert amdahl_f_puntual(s, p) == pytest.approx(f)
    assert amdahl_f_ajuste(HILOS, S) == pytest.approx(f)


def test_casos_limite_de_amdahl():
    # f = 1 (todo paralelo): speedup ideal S = p
    assert np.allclose(speedup_amdahl(1.0, HILOS), HILOS)
    # f = 0 (todo serial): no hay mejora, S = 1
    assert np.allclose(speedup_amdahl(0.0, HILOS), 1.0)
    # Con infinitos hilos el speedup se satura en 1 / (1 - f)
    assert speedup_amdahl(0.9, 1e12) == pytest.approx(10.0)


def test_f_puntual_indefinida_con_un_hilo():
    assert np.isnan(amdahl_f_puntual(1.0, 1))


def test_cronometrar_calienta_y_repite():
    llamadas = []
    tiempos = cronometrar(lambda: llamadas.append(1), reps=3)
    assert len(tiempos) == 3
    assert len(llamadas) == 4  # 1 de calentamiento + 3 medidas
