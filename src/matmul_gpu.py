"""Versión GPU de la multiplicación de matrices con numba.cuda.

Se usa igual que las demás: matmul_gpu(A, B) -> C (arreglos de NumPy).
Si no hay GPU NVIDIA o no está instalado numba-cuda, GPU_DISPONIBLE es
False y el resto del proyecto sigue funcionando solo con CPU.

MODELO DE EJECUCIÓN DE CUDA (lo mínimo para entender el kernel)
    - Un "kernel" es una función que se ejecuta en la GPU por MILES de
      hilos a la vez; cada hilo calcula UN elemento C[fila, col].
    - Los hilos se agrupan en BLOQUES (aquí de 16 x 16 = 256 hilos) y los
      bloques forman una MALLA (grid) que cubre toda la matriz C.
    - Los hilos de un mismo bloque comparten una memoria pequeña y muy
      rápida (memoria compartida, ~48 KB) y se pueden sincronizar entre
      sí con cuda.syncthreads(). Hilos de bloques distintos no.
    - La GPU tiene su propia memoria (VRAM). Los datos deben copiarse de
      la RAM a la VRAM y de regreso, a través del bus PCIe, que es lento.
"""
import math
import warnings

import numpy as np

from src.validacion import _preparar, _validar

# Lado del mosaico: bloques de 16 x 16 = 256 hilos. Es un valor clásico:
# múltiplo de 32 hilos (un "warp", la unidad que la GPU ejecuta en
# paralelo) y deja suficiente memoria compartida para varios bloques por
# multiprocesador.
TPB = 16

try:
    import numba
    from numba import cuda, float32, float64

    GPU_DISPONIBLE = cuda.is_available()
    # Numba avisa cuando la malla es pequeña (p. ej. C de 64 x 10 usa solo
    # 4 bloques). En el MLP eso pasa siempre y llenaría la consola.
    numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
except Exception:  # numba-cuda no instalado o sin controlador NVIDIA
    GPU_DISPONIBLE = False


def _crear_kernel(tipo):
    """Crea el kernel para un tipo de dato (float32 o float64).

    ¿Por qué una "fábrica"? La memoria compartida se reserva con un tamaño
    y un tipo FIJOS al compilar; no puede depender del tipo de los datos
    en tiempo de ejecución. Así que compilamos un kernel por cada tipo.
    """

    @cuda.jit
    def kernel(A, B, C):
        """Multiplicación por mosaicos con memoria compartida.

        Es la MISMA idea que la versión por bloques del CPU, pero aquí la
        "caché" la manejamos nosotros a mano: la memoria compartida.

        Cada bloque de 16 x 16 hilos calcula un mosaico 16 x 16 de C.
        Recorre la dimensión común en pasos de 16:
          1. Cada hilo copia UN elemento de A y UNO de B de la memoria
             global (lenta) a la memoria compartida (rápida). Entre los
             256 hilos cargan dos mosaicos completos de 16 x 16.
          2. syncthreads(): esperar a que TODOS terminen de copiar; si no,
             alguno leería un dato que todavía no se ha cargado.
          3. Cada hilo hace 16 multiplicaciones leyendo solo de la memoria
             compartida. Cada dato cargado se reutiliza 16 veces.
          4. syncthreads() otra vez antes de sobrescribir los mosaicos en
             la siguiente vuelta.
        Sin memoria compartida, cada hilo leería 2·m datos de la memoria
        global; con ella, 16 veces menos.
        """
        sA = cuda.shared.array((TPB, TPB), tipo)
        sB = cuda.shared.array((TPB, TPB), tipo)

        tx = cuda.threadIdx.x   # posición del hilo dentro del bloque
        ty = cuda.threadIdx.y
        # x recorre COLUMNAS: hilos vecinos (tx, tx+1) leen direcciones de
        # memoria vecinas. La GPU junta esas lecturas en una sola
        # transacción ("acceso coalescente"), lo cual es mucho más rápido.
        col = cuda.blockIdx.x * TPB + tx
        fila = cuda.blockIdx.y * TPB + ty

        n, m = A.shape
        q = B.shape[1]
        acumulado = tipo(0)

        for t in range((m + TPB - 1) // TPB):
            # Paso 1: carga. Fuera de los límites se carga 0, que no altera
            # la suma (bordes cuando n, m o q no son múltiplos de 16).
            k_a = t * TPB + tx
            k_b = t * TPB + ty
            sA[ty, tx] = A[fila, k_a] if (fila < n and k_a < m) else tipo(0)
            sB[ty, tx] = B[k_b, col] if (k_b < m and col < q) else tipo(0)
            cuda.syncthreads()                      # Paso 2
            for kk in range(TPB):                   # Paso 3
                acumulado += sA[ty, kk] * sB[kk, tx]
            cuda.syncthreads()                      # Paso 4

        # Los hilos que caen fuera de C (malla redondeada hacia arriba)
        # participaron en las cargas, pero no escriben.
        if fila < n and col < q:
            C[fila, col] = acumulado

    return kernel


if GPU_DISPONIBLE:
    _KERNELS = {np.dtype(np.float32): _crear_kernel(float32),
                np.dtype(np.float64): _crear_kernel(float64)}


def _lanzar(dA, dB, dC):
    """Lanza el kernel sobre arreglos que YA están en la GPU.

    La malla tiene ceil(q/16) x ceil(n/16) bloques para cubrir todo C.
    Se separa de matmul_gpu para poder medir el cómputo sin las copias.
    """
    n, q = dC.shape
    malla = (math.ceil(q / TPB), math.ceil(n / TPB))
    _KERNELS[np.dtype(dA.dtype)][malla, (TPB, TPB)](dA, dB, dC)


def matmul_gpu(A, B):
    """C = A @ B en la GPU, con la misma firma que las versiones de CPU.

    El costo total tiene TRES partes, y conviene tenerlas en mente al
    interpretar las mediciones:
        1. copiar A y B de la RAM a la GPU   (cuda.to_device)
        2. ejecutar el kernel                (muy rápido)
        3. copiar C de regreso a la RAM      (copy_to_host)
    Para matrices pequeñas, como las del MLP con lotes de 64, las copias
    pueden costar más que el cálculo mismo.
    """
    if not GPU_DISPONIBLE:
        raise RuntimeError("No hay GPU CUDA disponible (o falta instalar numba-cuda)")
    n, _, q = _validar(A, B)
    A, B = _preparar(A, B)
    dA = cuda.to_device(A)
    dB = cuda.to_device(B)
    dC = cuda.device_array((n, q), dtype=A.dtype)  # se reserva, no se copia
    _lanzar(dA, dB, dC)
    # copy_to_host espera a que termine el kernel (los lanzamientos son
    # asíncronos: la CPU sigue avanzando mientras la GPU trabaja).
    return dC.copy_to_host()


def calentar():
    """Compila los kernels con una llamada pequeña (excluye el JIT de las
    mediciones) y silencia avisos de rendimiento de matrices diminutas."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tipo in (np.float32, np.float64):
            matmul_gpu(np.ones((2, 2), tipo), np.ones((2, 2), tipo))
