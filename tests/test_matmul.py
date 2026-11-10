"""Tests de correctitud de las implementaciones de matmul.

Idea central: NumPy (A @ B) es nuestra "respuesta correcta". Cada backend
debe dar el mismo resultado salvo errores de redondeo, por eso comparamos
con np.allclose y una tolerancia, NUNCA con ==.

¿Por qué no son idénticos los resultados? La suma en punto flotante no es
asociativa: (a + b) + c puede diferir de a + (b + c) en el último bit.
BLAS suma en otro orden (bloques, SIMD) que nuestro triple ciclo, así que
pequeñas diferencias son normales y esperadas.

Los tests se parametrizan sobre TODOS los backends registrados en
src.matmul.BACKENDS: cuando agreguemos la versión por bloques o la de GPU,
quedarán probadas automáticamente sin escribir tests nuevos.
"""
import numpy as np
import pytest

from src.matmul import BACKENDS, obtener_matmul

# Tolerancias por tipo de dato. float32 tiene ~7 dígitos decimales de
# precisión y float64 ~16, así que float32 necesita una tolerancia mayor.
TOLERANCIAS = {
    np.float32: dict(rtol=1e-4, atol=1e-4),
    np.float64: dict(rtol=1e-10, atol=1e-10),
}

# Formas (n, m, q) para A (n x m) y B (m x q). Incluimos casos "incómodos"
# a propósito: vectores, matrices no cuadradas, tamaños primos y tamaños
# que no son múltiplos de un tamaño de bloque típico (útil en la etapa 3),
# y las formas reales que usará el MLP (lote de 64 x 784 por 784 x 128, etc.).
FORMAS = [
    (1, 1, 1),
    (1, 5, 1),     # producto punto: fila por columna
    (5, 1, 5),     # producto exterior: columna por fila
    (3, 4, 5),
    (17, 31, 13),  # primos: no dividen a ningún tamaño de bloque
    (64, 64, 64),
    (100, 70, 33),
    (64, 784, 128),  # capa 1 del MLP: lote x entrada x ocultas
    (64, 128, 64),   # capa 2
    (64, 64, 10),    # capa 3 (salida)
]


@pytest.fixture
def rng():
    """Generador con semilla fija: los tests son reproducibles."""
    return np.random.default_rng(42)


@pytest.mark.parametrize("nombre", list(BACKENDS))
@pytest.mark.parametrize("tipo", [np.float32, np.float64])
@pytest.mark.parametrize("n,m,q", FORMAS)
def test_coincide_con_numpy(nombre, tipo, n, m, q, rng):
    """Cada backend debe coincidir con NumPy para cualquier forma y tipo."""
    matmul = obtener_matmul(nombre)
    A = rng.standard_normal((n, m)).astype(tipo)
    B = rng.standard_normal((m, q)).astype(tipo)

    C = matmul(A, B)
    esperado = A @ B

    assert C.shape == (n, q)
    assert C.dtype == tipo, "el resultado debe conservar el tipo de entrada"
    assert np.allclose(C, esperado, **TOLERANCIAS[tipo])


@pytest.mark.parametrize("nombre", list(BACKENDS))
def test_transpuestas_no_contiguas(nombre, rng):
    """En el backpropagation multiplicamos por X.T y W.T, que son VISTAS
    no contiguas en memoria. El backend debe manejarlas correctamente."""
    matmul = obtener_matmul(nombre)
    X = rng.standard_normal((64, 30))
    W = rng.standard_normal((20, 30))
    assert not W.T.flags["C_CONTIGUOUS"]
    assert np.allclose(matmul(X, W.T), X @ W.T)
    assert np.allclose(matmul(X.T, X), X.T @ X)


@pytest.mark.parametrize("nombre", list(BACKENDS))
def test_identidad_y_ceros(nombre, rng):
    """Propiedades algebraicas simples: A·I = A y A·0 = 0."""
    matmul = obtener_matmul(nombre)
    A = rng.standard_normal((8, 6))
    assert np.allclose(matmul(A, np.eye(6)), A)
    assert np.allclose(matmul(A, np.zeros((6, 3))), 0.0)


@pytest.mark.parametrize("nombre", list(BACKENDS))
def test_no_modifica_entradas(nombre, rng):
    """El MLP reutiliza sus pesos después de multiplicar: la función no
    debe alterar A ni B (efectos secundarios ocultos = bugs difíciles)."""
    matmul = obtener_matmul(nombre)
    A = rng.standard_normal((10, 7))
    B = rng.standard_normal((7, 4))
    A0, B0 = A.copy(), B.copy()
    matmul(A, B)
    assert np.array_equal(A, A0) and np.array_equal(B, B0)


@pytest.mark.parametrize("nombre", list(BACKENDS))
def test_dimensiones_incompatibles(nombre):
    """Si las columnas de A no coinciden con las filas de B debe fallar con
    un error claro en lugar de leer memoria fuera del arreglo."""
    matmul = obtener_matmul(nombre)
    with pytest.raises(ValueError):
        matmul(np.ones((3, 4)), np.ones((5, 2)))
    with pytest.raises(ValueError):
        matmul(np.ones(3), np.ones((3, 2)))  # A no es 2D


def test_backend_inexistente():
    with pytest.raises(ValueError):
        obtener_matmul("cuántica")
