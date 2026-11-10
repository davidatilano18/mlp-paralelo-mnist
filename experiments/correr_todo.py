"""Corre TODOS los experimentos en orden y genera las gráficas.

    1. especificaciones del equipo
    2. escalabilidad con 1, 2, 4, 8 hilos (matmul y MLP)
    3. reto extra de caché (1 hilo)
    4. benchmark de GPU (si hay GPU)
    5. gráficas y resumen.md

Uso:
    python -m experiments.correr_todo            # completo (~15-25 min en una laptop)
    python -m experiments.correr_todo --rapido   # versión corta para probar que todo corre

Recomendaciones para medir bien en una laptop: conectada a la corriente,
modo de máximo rendimiento y sin otros programas pesados abiertos.
"""
import argparse
import sys
import time

from experiments import bench_cache, bench_hilos, especificaciones, graficas
from src.matmul_gpu import GPU_DISPONIBLE


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rapido", action="store_true", help="tamaños pequeños, solo para probar")
    p.add_argument("--reps", type=int, default=3)
    args = p.parse_args()
    reps = ["--reps", str(args.reps)]
    t0 = time.perf_counter()

    print("=" * 60 + "\n1. Especificaciones del equipo\n" + "=" * 60)
    especificaciones.main()

    print("\n" + "=" * 60 + "\n2. Escalabilidad con hilos\n" + "=" * 60)
    bench_hilos.main(reps + (["--n", "256", "--muestras", "3200"] if args.rapido else []))

    print("\n" + "=" * 60 + "\n3. Reto extra: caché con 1 hilo\n" + "=" * 60)
    bench_cache.main(reps + (["--tamanos", "128", "256", "--bloques", "16", "64"] if args.rapido else []))

    if GPU_DISPONIBLE:
        print("\n" + "=" * 60 + "\n4. GPU\n" + "=" * 60)
        from experiments import bench_gpu
        bench_gpu.main(["--reps", str(max(args.reps, 5))]
                       + (["--tamanos", "64", "256", "512"] if args.rapido else []))
    else:
        print("\n4. GPU: no disponible en este equipo, se omite.")

    print("\n" + "=" * 60 + "\n5. Gráficas\n" + "=" * 60)
    graficas.main()
    print(f"\nListo en {(time.perf_counter() - t0) / 60:.1f} min. Resultados en results/")


if __name__ == "__main__":
    main()
