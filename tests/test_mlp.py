"""Tests del MLP.

1. Verificación de gradientes: comprueba que nuestro backpropagation
   calcula las derivadas correctas comparándolo con una aproximación
   numérica. Es LA prueba estándar para saber si un backprop está bien.
2. Equivalencia entre backends: entrenar con cada matmul, misma semilla,
   debe dar los mismos pesos (salvo redondeo) y las mismas predicciones.
3. Sanidad: softmax estable y que la pérdida baje al entrenar.

Se usan datos sintéticos pequeños para que los tests no dependan de
descargar MNIST y corran en segundos.
"""
import numpy as np
import pytest

from src.matmul import BACKENDS, obtener_matmul
from src.mlp import MLP, entropia_cruzada, softmax


def datos_sinteticos(n=200, d=20, clases=4, semilla=0, dtype=np.float32):
    """Clases separables: cada clase es una nube alrededor de un centro."""
    rng = np.random.default_rng(semilla)
    centros = rng.standard_normal((clases, d)) * 3
    y = rng.integers(0, clases, n)
    X = centros[y] + rng.standard_normal((n, d))
    return X.astype(dtype), y


# ---------------------------------------------------------------------------
# 1. Verificación numérica de gradientes
# ---------------------------------------------------------------------------
def test_gradientes_coinciden_con_diferencias_finitas():
    """La derivada se puede aproximar con diferencias centrales:

        dL/dw  ~  (L(w + h) - L(w - h)) / (2h)

    Es lentísimo (dos forwards por cada peso), pero no tiene forma de
    equivocarse en la regla de la cadena. Si coincide con el backprop,
    el backprop está bien. Usamos float64 y una red diminuta para que
    la aproximación sea precisa y la prueba rápida.
    """
    X, y = datos_sinteticos(n=8, d=5, clases=3, dtype=np.float64)
    red = MLP(capas=(5, 4, 3, 3), semilla=1, dtype=np.float64)

    P = red.forward(X)
    gW, gb = red.backward(P, y)

    h = 1e-6
    for l in range(len(red.W)):
        for param, grad in ((red.W[l], gW[l]), (red.b[l], gb[l])):
            numerico = np.zeros_like(param)
            for idx in np.ndindex(param.shape):
                original = param[idx]
                param[idx] = original + h
                mas = entropia_cruzada(red.forward(X), y)
                param[idx] = original - h
                menos = entropia_cruzada(red.forward(X), y)
                param[idx] = original
                numerico[idx] = (mas - menos) / (2 * h)
            # Error relativo: robusto tanto para gradientes grandes como chicos
            err = np.abs(grad - numerico) / np.maximum(1e-8, np.abs(grad) + np.abs(numerico))
            assert err.max() < 1e-5, f"capa {l}: error relativo {err.max():.2e}"


# ---------------------------------------------------------------------------
# 2. Equivalencia entre backends
# ---------------------------------------------------------------------------
def entrenar_con(nombre, X, y):
    red = MLP(capas=(X.shape[1], 16, 8, 4), matmul=obtener_matmul(nombre), semilla=7)
    hist = red.entrenar(X, y, epocas=3, tam_lote=32, tasa=0.1, verbose=False)
    return red, hist


@pytest.mark.parametrize("nombre", [n for n in BACKENDS if n != "numpy"])
def test_backend_equivalente_a_numpy(nombre):
    """Misma semilla => mismos pesos iniciales y mismo orden de lotes.
    La única diferencia es el backend, así que el resultado debe coincidir
    salvo el redondeo de punto flotante (que se acumula un poco a lo largo
    de los pasos de entrenamiento, por eso la tolerancia es de 1e-3)."""
    X, y = datos_sinteticos()
    ref, hist_ref = entrenar_con("numpy", X, y)
    red, hist = entrenar_con(nombre, X, y)

    for W_ref, W in zip(ref.W, red.W):
        assert np.allclose(W, W_ref, rtol=1e-3, atol=1e-3)
    assert np.allclose(hist["perdida"], hist_ref["perdida"], rtol=1e-3)
    assert np.array_equal(red.predecir(X), ref.predecir(X))


# ---------------------------------------------------------------------------
# 3. Sanidad
# ---------------------------------------------------------------------------
def test_softmax_estable_y_normalizado():
    Z = np.array([[1000.0, 1001.0, 1002.0], [-5.0, 0.0, 5.0]])
    P = softmax(Z)
    assert np.all(np.isfinite(P)), "sin el truco de restar el máximo, e^1000 = inf"
    assert np.allclose(P.sum(axis=1), 1.0)


def test_la_perdida_baja_y_aprende():
    X, y = datos_sinteticos()
    red, hist = entrenar_con("numpy", X, y)
    assert hist["perdida"][-1] < hist["perdida"][0]
    assert red.precision(X, y) > 0.9


def test_misma_semilla_mismo_resultado():
    """Reproducibilidad: dos corridas idénticas dan pesos idénticos."""
    X, y = datos_sinteticos()
    a, _ = entrenar_con("numpy", X, y)
    b, _ = entrenar_con("numpy", X, y)
    for Wa, Wb in zip(a.W, b.W):
        assert np.array_equal(Wa, Wb)
