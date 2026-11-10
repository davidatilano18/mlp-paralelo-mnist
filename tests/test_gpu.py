"""Tests específicos de la versión GPU.

Los tests generales (test_matmul.py, test_mlp.py) ya incluyen "gpu"
automáticamente cuando hay GPU. Aquí probamos detalles propios de CUDA:
matrices más grandes que un mosaico en cada dimensión, con bordes
incompletos, y el lanzamiento directo sobre arreglos que ya viven en la
GPU. En un equipo sin GPU estos tests se SALTAN (no fallan).
"""
import numpy as np
import pytest

from src.matmul_gpu import GPU_DISPONIBLE, TPB

pytestmark = pytest.mark.skipif(not GPU_DISPONIBLE, reason="no hay GPU CUDA")


@pytest.mark.parametrize("n,m,q", [
    (TPB, TPB, TPB),              # exactamente un mosaico
    (TPB + 1, TPB - 1, TPB + 3),  # un mosaico y un pedacito en cada borde
    (300, 200, 170),              # varios mosaicos, ninguno múltiplo de 16
])
def test_bordes_de_mosaico(n, m, q):
    from src.matmul_gpu import matmul_gpu
    rng = np.random.default_rng(3)
    A = rng.standard_normal((n, m)).astype(np.float32)
    B = rng.standard_normal((m, q)).astype(np.float32)
    assert np.allclose(matmul_gpu(A, B), A @ B, rtol=1e-4, atol=1e-4)


def test_lanzar_con_datos_en_la_gpu():
    """_lanzar trabaja sobre arreglos que ya están en la VRAM: es lo que
    usaremos para medir el kernel sin el costo de las copias."""
    from numba import cuda

    from src.matmul_gpu import _lanzar
    rng = np.random.default_rng(4)
    A = rng.standard_normal((50, 40)).astype(np.float32)
    B = rng.standard_normal((40, 30)).astype(np.float32)
    dC = cuda.device_array((50, 30), dtype=np.float32)
    _lanzar(cuda.to_device(A), cuda.to_device(B), dC)
    assert np.allclose(dC.copy_to_host(), A @ B, rtol=1e-4, atol=1e-4)
