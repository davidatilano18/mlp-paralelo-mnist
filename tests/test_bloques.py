"""Tests específicos de la versión por bloques en paralelo.

Los tests generales de test_matmul.py ya prueban "bloques" con el tamaño
por defecto (porque recorren BACKENDS). Aquí probamos lo que es propio de
esta versión: que el resultado NO dependa ni del tamaño de bloque ni del
número de hilos. Si dependiera, habría un error en el manejo de los
bordes o una condición de carrera entre hilos.
"""
import numba
import numpy as np
import pytest

from src.matmul import crear_matmul_bloques, matmul_bloques


@pytest.fixture
def matrices():
    rng = np.random.default_rng(0)
    # Tamaños primos a propósito: ningún tamaño de bloque los divide
    # exacto, así que siempre se ejercitan los bloques incompletos del borde.
    A = rng.standard_normal((67, 131))
    B = rng.standard_normal((131, 53))
    return A, B, A @ B


@pytest.mark.parametrize("bs", [1, 2, 7, 16, 32, 64, 500])
def test_cualquier_tamano_de_bloque(matrices, bs):
    """bs = 1 es el caso extremo (un elemento por bloque) y bs = 500 es
    mayor que todas las dimensiones (un solo bloque)."""
    A, B, esperado = matrices
    assert np.allclose(matmul_bloques(A, B, bs), esperado)


@pytest.mark.parametrize("hilos", [1, 2, 4, 8])
def test_cualquier_numero_de_hilos(matrices, hilos):
    """Con más hilos los mosaicos se reparten distinto, pero cada mosaico
    se calcula siempre igual, así que el resultado debe ser IDÉNTICO."""
    hilos = min(hilos, numba.config.NUMBA_NUM_THREADS)  # por si el equipo tiene menos
    A, B, _ = matrices
    original = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        referencia = matmul_bloques(A, B)
        numba.set_num_threads(hilos)
        assert np.array_equal(matmul_bloques(A, B), referencia)
    finally:
        numba.set_num_threads(original)


def test_fabrica_fija_el_tamano(matrices):
    A, B, esperado = matrices
    f = crear_matmul_bloques(16)
    assert f.__name__ == "matmul_bloques_16"
    assert np.allclose(f(A, B), esperado)


def test_variante_ikj_del_experimento_de_cache(matrices):
    """matmul_ikj no está en BACKENDS (no es una versión oficial), así que
    la probamos aparte: se usa en el reto extra y también debe ser correcta."""
    from src.matmul import matmul_ikj
    A, B, esperado = matrices
    assert np.allclose(matmul_ikj(A, B), esperado)


def test_tamano_invalido(matrices):
    A, B, _ = matrices
    with pytest.raises(ValueError):
        matmul_bloques(A, B, 0)
