"""Implementaciones de multiplicación de matrices C = A @ B.

Todas las versiones comparten la MISMA firma:

    matmul(A, B) -> C

donde A tiene forma (n, m), B tiene forma (m, q) y C tiene forma (n, q).
Gracias a eso el MLP puede recibir cualquiera de ellas como parámetro y
cambiar de "backend" sin modificar una sola línea de la red.

Versiones (se irán agregando por etapas):
    1. matmul_ingenua  -> triple ciclo secuencial con Numba (@njit).  [Etapa 1]
    2. matmul_bloques  -> tiling + paralelismo con prange.            [Etapa 3]
                          (tamaño de bloque configurable)
    3. matmul_gpu      -> kernel CUDA con numba.cuda.                 [Etapa 4]
                          (en src/matmul_gpu.py; solo si hay GPU)
    4. matmul_numpy    -> A @ B (BLAS), referencia "de producción".   [Etapa 1]
"""
import numpy as np
from numba import njit, prange

from src.matmul_gpu import GPU_DISPONIBLE, matmul_gpu
from src.validacion import _preparar, _validar

# Tamaño de bloque por defecto (ver explicación en _matmul_bloques_kernel).
TAM_BLOQUE_DEFECTO = 32


# ---------------------------------------------------------------------------
# 1. Versión ingenua secuencial (línea base de todas las mediciones)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _matmul_ingenua_kernel(A, B):
    """Triple ciclo clásico i-j-k, exactamente la definición matemática:

        C[i, j] = sum_k A[i, k] * B[k, j]

    @njit compila esta función a código máquina (vía LLVM) la primera vez
    que se llama con ciertos tipos. Así comparamos ALGORITMOS y no
    "Python lento vs. C rápido": sin Numba, un triple ciclo en Python puro
    sería ~100x más lento y cualquier speedup posterior sería engañoso.

    cache=True guarda el código compilado en disco (__pycache__) para que
    la siguiente ejecución del programa no tenga que recompilar.

    Nota sobre la caché del procesador (importante para el reto extra):
    en el ciclo más interno k avanza, así que A[i, k] se lee por FILA
    (posiciones contiguas en memoria, bien) pero B[k, j] se lee por
    COLUMNA (saltando q elementos cada vez, mal). Para matrices grandes
    cada acceso a B puede ser un fallo de caché. La versión por bloques
    de la etapa 3 ataca justamente este problema.
    """
    n, m = A.shape
    q = B.shape[1]
    C = np.zeros((n, q), dtype=A.dtype)
    for i in range(n):
        for j in range(q):
            # Acumulamos en una variable local (vive en un registro del CPU)
            # en lugar de escribir C[i, j] en cada vuelta: evita m escrituras
            # a memoria por cada elemento de C.
            # Se inicializa con C[i, j] (que vale 0) para que el acumulador
            # tenga el MISMO tipo que la matriz (float32 o float64). Si
            # escribiéramos 0.0, Numba lo haría float64 siempre.
            acumulado = C[i, j]
            for k in range(m):
                acumulado += A[i, k] * B[k, j]
            C[i, j] = acumulado
    return C


def matmul_ingenua(A, B):
    """C = A @ B con el triple ciclo secuencial compilado por Numba."""
    _validar(A, B)
    A, B = _preparar(A, B)
    return _matmul_ingenua_kernel(A, B)


# ---------------------------------------------------------------------------
# Variante de apoyo: i-k-j SIN bloques (solo para el experimento de caché)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _matmul_ikj_kernel(A, B):
    """Igual que la ingenua pero con los ciclos en orden i-k-j y SIN bloques.

    No es uno de los 4 backends oficiales: existe para el reto extra. La
    versión por bloques cambia DOS cosas respecto a la ingenua:
        (a) el orden de los ciclos (i-k-j: acceso por filas + SIMD), y
        (b) el recorrido por bloques (reutilización de datos en caché).
    Midiendo ingenua -> i-k-j -> bloques podemos separar cuánto aporta
    cada cambio por sí solo, en lugar de atribuirle todo a "la caché".
    """
    n, m = A.shape
    q = B.shape[1]
    C = np.zeros((n, q), dtype=A.dtype)
    for i in range(n):
        C_fila = C[i]
        for k in range(m):
            a_ik = A[i, k]
            B_fila = B[k]
            for j in range(q):
                C_fila[j] += a_ik * B_fila[j]
    return C


def matmul_ikj(A, B):
    """C = A @ B con orden i-k-j sin bloques (secuencial)."""
    _validar(A, B)
    A, B = _preparar(A, B)
    return _matmul_ikj_kernel(A, B)


# ---------------------------------------------------------------------------
# 2. Versión por bloques (tiling) en paralelo con prange
# ---------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _matmul_bloques_kernel(A, B, bs):
    """C = A @ B dividiendo C en "mosaicos" (tiles) de bs x bs.

    IDEA 1 — BLOQUES PARA APROVECHAR LA CACHÉ
    La caché del CPU es una memoria pequeña (L1 ~48 KB por núcleo) pero
    mucho más rápida que la RAM. En la versión ingenua, para calcular una
    fila de C se recorre TODA la matriz B; si B no cabe en caché, cada
    fila de C vuelve a traerla desde la RAM.
    Por bloques, trabajamos con un pedazo de A (bs x bs), uno de B
    (bs x bs) y uno de C (bs x bs) a la vez. Con bs = 32 y float32 son
    3 * 32 * 32 * 4 B = 12 KB: los tres caben juntos en L1 y cada número
    que se trae de la RAM se reutiliza ~bs veces antes de salir de caché.

    IDEA 2 — ORDEN i-k-j DENTRO DEL BLOQUE
    En el ciclo más interno j avanza, así que B[k, j] y C[i, j] se leen
    por FILA, es decir, en posiciones contiguas de memoria. Eso aprovecha
    cada línea de caché completa (64 bytes = 16 float32) y además permite
    que el compilador use instrucciones SIMD (varias multiplicaciones en
    una sola instrucción). La ingenua (i-j-k) lee B por columna.

    IDEA 3 — PARALELISMO SIN CONDICIONES DE CARRERA
    Cada iteración del prange calcula UN mosaico completo de C. Ningún
    otro hilo escribe en ese mosaico, así que no hacen falta candados
    (locks) ni operaciones atómicas: los hilos son independientes.
    ¿Por qué repartir mosaicos de C y no solo bloques de filas? En el MLP
    muchas multiplicaciones tienen pocas filas (el lote es de 64). Con
    bloques de 32 filas solo habría 2 tareas para 8 hilos. Al numerar los
    mosaicos (fila de bloques, columna de bloques) hay más tareas que
    repartir: un C de 64 x 128 tiene 2 x 4 = 8 mosaicos.

    prange (en vez de range) le indica a Numba que las iteraciones son
    independientes y se pueden repartir entre los hilos. El número de
    hilos se controla con numba.set_num_threads(p).
    """
    n, m = A.shape
    q = B.shape[1]
    C = np.zeros((n, q), dtype=A.dtype)

    bloques_i = (n + bs - 1) // bs   # división hacia arriba: el último
    bloques_j = (q + bs - 1) // bs   # bloque puede quedar incompleto
    for t in prange(bloques_i * bloques_j):
        # Convertimos el número de mosaico t en (fila, columna) de bloque
        i0 = (t // bloques_j) * bs
        j0 = (t % bloques_j) * bs
        i1 = min(i0 + bs, n)          # min(): bordes cuando n o q no son
        j1 = min(j0 + bs, q)          # múltiplos de bs
        ancho = j1 - j0
        # Recorremos la dimensión común (k) también por bloques
        for k0 in range(0, m, bs):
            k1 = min(k0 + bs, m)
            for i in range(i0, i1):
                # Rebanadas 1D de la fila: son VISTAS (no copian datos).
                # ¿Por qué no escribir directo C[i, j] += A[i, k] * B[k, j]?
                # Lo medimos: con índices 2D, LLVM no logró vectorizar este
                # ciclo y era ~7x más lento. Con vectores 1D contiguos y un
                # índice que va de 0 a ancho, el compilador sí genera
                # instrucciones SIMD (AVX2: 8 float32 por instrucción).
                C_fila = C[i, j0:j1]
                for k in range(k0, k1):
                    a_ik = A[i, k]           # se queda en un registro
                    B_fila = B[k, j0:j1]
                    for j in range(ancho):
                        C_fila[j] += a_ik * B_fila[j]
    return C


def matmul_bloques(A, B, tam_bloque=TAM_BLOQUE_DEFECTO):
    """C = A @ B por bloques y en paralelo.

    tam_bloque es opcional, así que la firma básica matmul(A, B) se
    conserva. Para fijar otro tamaño y pasarlo al MLP usa crear_matmul_bloques.
    """
    if tam_bloque < 1:
        raise ValueError("tam_bloque debe ser un entero positivo")
    _validar(A, B)
    A, B = _preparar(A, B)
    return _matmul_bloques_kernel(A, B, int(tam_bloque))


def crear_matmul_bloques(tam_bloque):
    """Devuelve una función matmul(A, B) con el tamaño de bloque fijo.

    Ejemplo:  MLP(matmul=crear_matmul_bloques(64))
    Una "fábrica" de funciones (closure): la función interna recuerda el
    valor de tam_bloque con el que fue creada.
    """
    def matmul(A, B):
        return matmul_bloques(A, B, tam_bloque)
    matmul.__name__ = f"matmul_bloques_{tam_bloque}"
    return matmul


# ---------------------------------------------------------------------------
# 4. Referencia NumPy (BLAS)
# ---------------------------------------------------------------------------
def matmul_numpy(A, B):
    """C = A @ B usando NumPy, que delega en una biblioteca BLAS optimizada
    (OpenBLAS o MKL): usa instrucciones vectoriales SIMD, bloques ajustados
    a la caché y varios hilos. Sirve como "techo" realista de rendimiento y
    como respuesta correcta contra la cual verificamos las demás versiones.
    """
    _validar(A, B)
    A, B = _preparar(A, B)
    return A @ B


# ---------------------------------------------------------------------------
# Registro de backends
# ---------------------------------------------------------------------------
# Diccionario nombre -> función. Los tests, el MLP y los benchmarks lo usan
# para recorrer todas las versiones sin escribir un if por cada una.
# Cada etapa agregará aquí su nueva versión.
BACKENDS = {
    "ingenua": matmul_ingenua,
    "bloques": matmul_bloques,
    "numpy": matmul_numpy,
}

# La versión GPU vive en src/matmul_gpu.py y solo se registra si hay una
# GPU CUDA utilizable. Así el proyecto funciona igual en equipos sin GPU
# (por ejemplo, dentro de Docker), donde simplemente no aparece "gpu".
if GPU_DISPONIBLE:
    BACKENDS["gpu"] = matmul_gpu


def obtener_matmul(nombre):
    """Devuelve la función matmul registrada con ese nombre."""
    try:
        return BACKENDS[nombre]
    except KeyError:
        disponibles = ", ".join(BACKENDS)
        raise ValueError(f"Backend '{nombre}' no existe. Disponibles: {disponibles}") from None
