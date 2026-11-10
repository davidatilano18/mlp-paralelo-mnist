# Resumen de resultados
Equipo: 13th Gen Intel(R) Core(TM) i5-13420H · 8 núcleos físicos / 12 lógicos

## matmul 1024×1024
| backend | hilos | media | std | speedup | eficiencia | f_puntual | speedup_vs_ingenua |
|---|---|---|---|---|---|---|---|
| bloques | 1 | 0.095 | 0.001 | 1.000 | 1.000 | — | 27.842 |
| bloques | 2 | 0.049 | 0.000 | 1.928 | 0.964 | 0.963 | 53.689 |
| bloques | 4 | 0.026 | 0.002 | 3.676 | 0.919 | 0.971 | 102.360 |
| bloques | 8 | 0.032 | 0.001 | 2.984 | 0.373 | 0.760 | 83.090 |
| gpu | — | 0.005 | 0.001 | — | — | — | 482.663 |
| ingenua | 1 | 2.649 | 0.027 | — | — | — | 1.000 |
| numpy | — | 0.008 | 0.002 | — | — | — | 343.436 |

## MLP: 1 época de MNIST
| backend | hilos | media | std | speedup | eficiencia | f_puntual | speedup_vs_ingenua |
|---|---|---|---|---|---|---|---|
| bloques | 1 | 1.801 | 0.017 | 1.000 | 1.000 | — | 3.057 |
| bloques | 2 | 1.184 | 0.003 | 1.521 | 0.761 | 0.685 | 4.650 |
| bloques | 4 | 1.036 | 0.025 | 1.739 | 0.435 | 0.567 | 5.317 |
| bloques | 8 | 1.076 | 0.023 | 1.675 | 0.209 | 0.461 | 5.120 |
| gpu | — | 5.191 | 0.381 | — | — | — | 1.061 |
| ingenua | 1 | 5.507 | 0.017 | — | — | — | 1.000 |
| numpy | — | 3.178 | 2.263 | — | — | — | 1.733 |

## Ley de Amdahl (ajuste por mínimos cuadrados)
| carga | f_ajuste | speedup_maximo |
|---|---|---|
| matmul | 0.867 | 7.529 |
| mlp | 0.534 | 2.146 |

## Reto extra: GFLOP/s con 1 hilo
| n | ingenua | ikj | bloques (mejor) | mejor bs |
|---|---|---|---|---|
| 128 | 4.80 | 40.26 | 35.62 | 256 |
| 256 | 4.18 | 51.33 | 39.50 | 256 |
| 512 | 4.13 | 47.23 | 42.31 | 256 |
| 768 | 3.52 | 33.82 | 39.30 | 256 |
| 1024 | 0.83 | 31.04 | 38.93 | 256 |
