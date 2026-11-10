"""Validación y preparación de entradas, común a TODAS las versiones de matmul
(CPU y GPU). Vive en su propio módulo para que src/matmul.py y
src/matmul_gpu.py puedan usarla sin importarse uno al otro.
"""
import numpy as np


def _validar(A, B):
    """Revisa que A y B se puedan multiplicar y devuelve sus dimensiones.

    ¿Por qué validar en Python y no dentro de la función de Numba?
    Porque si las formas no coinciden, el código compilado leería fuera de
    los límites del arreglo (Numba no revisa índices por defecto) y daría
    basura o tronaría sin un mensaje claro. Es mejor fallar pronto y con
    un error entendible.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"Se esperaban matrices 2D, llegaron {A.ndim}D y {B.ndim}D")
    n, m = A.shape
    m2, q = B.shape
    if m != m2:
        raise ValueError(
            f"Dimensiones incompatibles: A es {A.shape} y B es {B.shape} "
            f"(las columnas de A deben ser iguales a las filas de B)"
        )
    return n, m, q


def _preparar(A, B):
    """Convierte A y B a un tipo de dato común y a memoria contigua (C-order).

    - Tipo común: si una es float32 y otra float64, Numba compilaría una
      versión distinta para cada combinación; además el resultado debe
      tener una precisión bien definida. Usamos la regla de NumPy.
    - Contigua en orden C (fila por fila): los ciclos recorren la memoria
      asumiendo ese orden. Una vista transpuesta (p. ej. W.T) NO es
      contigua, y en el backpropagation usaremos muchas transpuestas.
    """
    tipo = np.result_type(A.dtype, B.dtype, np.float32)  # al menos float32
    A = np.ascontiguousarray(A, dtype=tipo)
    B = np.ascontiguousarray(B, dtype=tipo)
    return A, B
