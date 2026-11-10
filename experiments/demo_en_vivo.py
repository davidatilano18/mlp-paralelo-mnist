"""Demostración EN VIVO para la presentación, en 4 partes.

Se detiene al final de cada parte hasta que se presiona Enter, para que
cada quien explique mientras la pantalla muestra los resultados.

Uso:
    python -m experiments.demo_en_vivo              # presentación completa
    python -m experiments.demo_en_vivo --parte 3    # ensayar solo una parte
    python -m experiments.demo_en_vivo --sin-pausas # probar que todo corre

Antes de empezar compila TODAS las funciones de Numba (calentamiento), así
nada se queda "pensando" frente al grupo por la compilación JIT.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numba
import numpy as np

from experiments.medicion import amdahl_f_ajuste, cronometrar
from src import datos
from src.matmul import (BACKENDS, crear_matmul_bloques, matmul_bloques, matmul_ikj,
                        matmul_ingenua, matmul_numpy)
from src.matmul_gpu import GPU_DISPONIBLE
from src.mlp import MLP

RESULTADOS = Path(__file__).resolve().parent.parent / "results"
PAUSAS = True


# ---------------------------------------------------------------------------
# Utilidades de presentación
# ---------------------------------------------------------------------------
def titulo(texto):
    print("\n" + "═" * 70)
    print(f"  {texto}")
    print("═" * 70 + "\n")


def pausa(mensaje="Enter para continuar"):
    if PAUSAS:
        input(f"\n   ▶ {mensaje}...")
    print()


def ms(t):
    return f"{t * 1e3:8.1f} ms" if t < 1 else f"{t:8.2f} s "


def barra(valor, maximo, ancho=40):
    """Barra de texto para comparar tiempos visualmente en la terminal."""
    n = max(1, round(ancho * valor / maximo))
    return "█" * n


def dibujar_digito(x):
    """Dibuja una imagen de 28x28 con caracteres (más oscuro = más tinta)."""
    tonos = " ·░▒▓█"
    img = x.reshape(28, 28)
    for fila in img[2:26]:  # recortamos el borde vacío
        print("      " + "".join(tonos[min(int(v * (len(tonos) - 1) + 0.5), len(tonos) - 1)] * 2
                                 for v in fila[2:26]))


def abrir(archivo):
    """Abre una gráfica con el visor de imágenes del sistema."""
    ruta = RESULTADOS / archivo
    if not ruta.exists():
        print(f"   (no existe {archivo}; corre primero experiments.correr_todo)")
        return
    try:
        if sys.platform == "win32":
            os.startfile(ruta)  # noqa: S606 - abrir un PNG local del proyecto
        elif sys.platform == "darwin":
            os.system(f'open "{ruta}"')
        else:
            os.system(f'xdg-open "{ruta}" >/dev/null 2>&1 &')
        print(f"   Abriendo results/{archivo}")
    except Exception:
        print(f"   Abre manualmente results/{archivo}")


# ---------------------------------------------------------------------------
# Calentamiento: compilar todo antes de presentar
# ---------------------------------------------------------------------------
def calentar_todo(X, y):
    print("Preparando la demostración (compilando funciones de Numba)...", flush=True)
    t0 = time.perf_counter()
    A = np.ones((40, 40), np.float32)
    for f in (matmul_ingenua, matmul_ikj, matmul_bloques, matmul_numpy):
        f(A, A)
    if GPU_DISPONIBLE:
        from numba import cuda

        from experiments.bench_gpu import llevar_a_reloj_de_trabajo
        from src.matmul_gpu import calentar
        calentar()
        llevar_a_reloj_de_trabajo(cuda, segundos=0.5)
    MLP(matmul=crear_matmul_bloques(32)).entrenar(X[:640], y[:640], epocas=1, verbose=False)
    print(f"Listo ({time.perf_counter() - t0:.1f} s).")


# ---------------------------------------------------------------------------
# Parte 1 — el problema y la red
# ---------------------------------------------------------------------------
def parte_1(X, y, Xp, yp):
    titulo("PARTE 1 · El problema y la red neuronal")

    print(f"MNIST: {X.shape[0]:,} imágenes de entrenamiento y {Xp.shape[0]:,} de prueba.")
    print(f"Cada imagen es un vector de {X.shape[1]} pixeles (28 × 28) con valores de 0 a 1.\n")
    print(f"   Ejemplo: la imagen #0 de prueba es un {yp[0]}:")
    dibujar_digito(Xp[0])
    pausa("Enter para entrenar la red")

    print("Red 784 → 128 → 64 → 10 · ReLU · softmax · SGD con lotes de 64")
    print("Entrenando 1 época (60 000 imágenes) con el matmul por bloques...\n")
    red = MLP(matmul=crear_matmul_bloques(32), semilla=0)
    t0 = time.perf_counter()
    red.entrenar(X, y, epocas=1, tam_lote=64, tasa=0.1, X_val=Xp, y_val=yp)
    print(f"\n   Tiempo: {time.perf_counter() - t0:.2f} s para 1 época")
    print("   Con 10 épocas llega a 97.91 % (ver README).")

    P = red.forward(Xp[:1])[0]
    print(f"\n   ¿Qué dice la red de la imagen #0?  → {int(np.argmax(P))}")
    for d in np.argsort(P)[::-1][:3]:
        print(f"      {d}: {P[d]:6.1%}  {barra(P[d], 1, 30)}")

    print("\n   CLAVE: cada paso hace 3 multiplicaciones de matrices por capa")
    print("          (A·W en el forward, Aᵀ·dZ y dZ·Wᵀ en el backward).")
    print("          → Ahí se va casi todo el tiempo. Eso es lo que paralelizamos.")
    pausa("Enter para pasar a la Parte 2")


# ---------------------------------------------------------------------------
# Parte 2 — matmul en CPU y caché
# ---------------------------------------------------------------------------
def parte_2():
    titulo("PARTE 2 · Matmul en CPU: ingenua vs. por bloques (1 solo hilo)")
    n = 1024
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n)).astype(np.float32)
    B = rng.standard_normal((n, n)).astype(np.float32)
    numba.set_num_threads(1)

    print(f"Multiplicando dos matrices de {n} × {n} con UN SOLO HILO.")
    print("Como no hay paralelismo, toda diferencia viene de cómo se usa la memoria.\n")
    variantes = [
        ("ingenua (i-j-k)", lambda: matmul_ingenua(A, B)),
        ("i-k-j sin bloques", lambda: matmul_ikj(A, B)),
        ("bloques bs=32", lambda: matmul_bloques(A, B, 32)),
        ("bloques bs=128", lambda: matmul_bloques(A, B, 128)),
    ]
    base = None
    for nombre, f in variantes:
        t = min(cronometrar(f, reps=2, calentar=False))
        comparacion = "  (referencia)" if base is None else f"{(base / t):6.1f}× más rápido"
        base = base or t  # la primera (ingenua) es la referencia
        print(f"   {nombre:20s} {ms(t)}  {comparacion:16s} {barra(t, base)}", flush=True)

    print("\n   ingenua → i-k-j : leer B por FILAS (contiguo, SIMD) en vez de por columnas")
    print("   i-k-j  → bloques: reutilizar cada bloque mientras está en caché")

    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    print("\n¿Es correcto? Comparamos contra NumPy con np.allclose:")
    for nombre, f in (("ingenua", matmul_ingenua), ("bloques", matmul_bloques)):
        ok = np.allclose(f(A[:200, :300], B[:300, :150]), A[:200, :300] @ B[:300, :150],
                         rtol=1e-4, atol=1e-4)
        print(f"   {nombre:10s} {'✔ coincide' if ok else '✘ NO coincide'}")
    pausa("Enter para pasar a la Parte 3")


# ---------------------------------------------------------------------------
# Parte 3 — GPU
# ---------------------------------------------------------------------------
def parte_3():
    titulo("PARTE 3 · La GPU: cómputo vs. copias")
    if not GPU_DISPONIBLE:
        print("Este equipo no tiene GPU CUDA. Mostramos la gráfica ya medida:")
        abrir("gpu.png")
        pausa("Enter para pasar a la Parte 4")
        return

    from numba import cuda

    from experiments.bench_gpu import llevar_a_reloj_de_trabajo
    from src.matmul_gpu import _lanzar, matmul_gpu
    nombre = cuda.get_current_device().name
    print(f"GPU: {nombre.decode() if isinstance(nombre, bytes) else nombre}\n")
    llevar_a_reloj_de_trabajo(cuda, segundos=1.0)

    rng = np.random.default_rng(0)
    print(f"   {'n':>6s} | {'GPU total':>11s} | {'solo kernel':>11s} | {'copias':>7s} | {'CPU NumPy':>11s}")
    print("   " + "-" * 62)
    for n in (64, 512, 2048):
        A = rng.standard_normal((n, n)).astype(np.float32)
        B = rng.standard_normal((n, n)).astype(np.float32)
        dA, dB = cuda.to_device(A), cuda.to_device(B)
        dC = cuda.device_array((n, n), np.float32)

        def kernel():
            _lanzar(dA, dB, dC)
            cuda.synchronize()

        total = np.mean(cronometrar(lambda: matmul_gpu(A, B), reps=5))
        k = np.mean(cronometrar(kernel, reps=5))
        cpu = np.mean(cronometrar(lambda: matmul_numpy(A, B), reps=5))
        print(f"   {n:6d} | {ms(total)}  | {ms(k)}  | {(total - k) / total:6.0%} | {ms(cpu)}")

    print("\n   n = 64 (como el lote del MLP): casi todo el tiempo es COPIAR por PCIe.")
    print("   n = 2048: el cálculo domina y la GPU le gana a la CPU.")
    print("   → Por eso entrenar el MLP en GPU (≈5 s/época) es más lento que en CPU (≈1 s).")
    pausa("Enter para pasar a la Parte 4")


# ---------------------------------------------------------------------------
# Parte 4 — speedup, eficiencia y Amdahl
# ---------------------------------------------------------------------------
def parte_4():
    titulo("PARTE 4 · Speedup, eficiencia y Ley de Amdahl")
    n = 1024
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n)).astype(np.float32)
    B = rng.standard_normal((n, n)).astype(np.float32)
    maximo = numba.config.NUMBA_NUM_THREADS
    hilos = [p for p in (1, 2, 4, 8) if p <= maximo]

    print(f"Matmul por bloques {n} × {n}, 3 repeticiones por configuración (en vivo):\n")
    print(f"   {'hilos':>5s} | {'tiempo':>16s} | {'S = T1/Tp':>9s} | {'E = S/p':>7s}")
    print("   " + "-" * 50)
    medias = []
    for p in hilos:
        numba.set_num_threads(p)
        t = cronometrar(lambda: matmul_bloques(A, B), reps=3)
        medias.append(np.mean(t))
        S = medias[0] / medias[-1]
        print(f"   {p:5d} | {ms(np.mean(t))} ± {np.std(t, ddof=1) * 1e3:4.1f} | "
              f"{S:9.2f} | {S / p:7.2f}  {barra(S, max(hilos), 24)}")
    numba.set_num_threads(maximo)

    S = medias[0] / np.array(medias)
    f = amdahl_f_ajuste(np.array(hilos), S)
    print(f"\n   Ley de Amdahl:  S = 1 / ((1 - f) + f/p)   →   f ≈ {f:.2f}")
    print(f"   Speedup máximo con infinitos hilos: 1 / (1 - f) ≈ {1 / (1 - f):.1f}×")
    # El comentario sobre 8 hilos se basa en lo que se ACABA de medir: en
    # vivo los números varían de una corrida a otra.
    if len(hilos) == 4:
        E8 = S[3] / 8
        if S[3] < S[2]:
            print(f"\n   Con 8 hilos fue MÁS LENTO que con 4 (eficiencia {E8:.2f}).")
        else:
            print(f"\n   De 4 a 8 hilos casi no mejora: el speedup sube solo de "
                  f"{S[2]:.2f} a {S[3]:.2f} (eficiencia {E8:.2f}).")
        print("   Causa: con 8 hilos entran los 4 núcleos de EFICIENCIA (CPU híbrida),")
        print("   más lentos; los núcleos rápidos terminan y esperan a los lentos.")
    print("\n   En el MLP completo f ≈ 0.53 → máximo ≈ 2.1× aunque hubiera infinitos")
    print("   hilos (ReLU, softmax, SGD y el ciclo de Python son seriales).")
    pausa("Enter para abrir las gráficas")
    for g in ("speedup.png", "eficiencia.png", "tiempo_vs_nucleos.png"):
        abrir(g)
    print("\n   Fin de la demostración. ¡Gracias!")


def main():
    global PAUSAS
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parte", type=int, choices=[1, 2, 3, 4], help="ejecutar solo una parte")
    p.add_argument("--sin-pausas", action="store_true", help="no esperar Enter (para probar)")
    args = p.parse_args()
    PAUSAS = not args.sin_pausas

    X, y, Xp, yp = datos.cargar("mnist")
    calentar_todo(X, y)
    print(f"Backends disponibles: {', '.join(BACKENDS)} · hilos máximos: "
          f"{numba.config.NUMBA_NUM_THREADS}")
    pausa("Enter para comenzar")

    partes = {1: lambda: parte_1(X, y, Xp, yp), 2: parte_2, 3: parte_3, 4: parte_4}
    for n in ([args.parte] if args.parte else [1, 2, 3, 4]):
        partes[n]()


if __name__ == "__main__":
    main()
