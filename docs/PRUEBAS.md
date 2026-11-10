# Catálogo de pruebas

El proyecto tiene dos tipos de pruebas:

1. **Pruebas de correctitud** (130 tests automáticos con `pytest`). Responden
   la pregunta *¿da el resultado correcto?*
2. **Pruebas de rendimiento** (experimentos en `experiments/`). Responden
   *¿qué tan rápido es y por qué?*

```bash
python -m pytest -q          # corre los 130 tests (~5 s)
python -m pytest -v          # muestra el nombre de cada test
python -m pytest tests/test_mlp.py -v   # solo un archivo
```

---

## 1. Pruebas de correctitud (130 tests)

| Archivo | Tests | Qué garantiza |
|---|---|---|
| `test_matmul.py` | 97 | Cada versión de matmul da lo mismo que NumPy |
| `test_bloques.py` | 14 | La versión por bloques no depende del tamaño de bloque ni del número de hilos |
| `test_gpu.py` | 4 | Detalles propios del kernel CUDA |
| `test_mlp.py` | 7 | El backpropagation es correcto y todos los backends entrenan igual |
| `test_metricas.py` | 8 | Las fórmulas de speedup y Amdahl son correctas |

¿Por qué hay tantos? Muchos tests están **parametrizados**: el mismo test se
repite con cada backend (ingenua, bloques, gpu y numpy), con cada tipo de dato
(float32 y float64) y con varias formas de matriz. Por ejemplo,
`test_coincide_con_numpy` se ejecuta 4 backends × 2 tipos × 10 formas = **80 veces**.

### [`tests/test_matmul.py`](../tests/test_matmul.py): todas las versiones contra NumPy

| Test | Veces | Qué comprueba | Por qué importa |
|---|---|---|---|
| `test_coincide_con_numpy` | 80 | `np.allclose(matmul(A, B), A @ B)` para 10 formas, float32 y float64 | Es la prueba principal de correctitud. Incluye formas "incómodas" (vectores, tamaños primos) y las formas reales del MLP (64×784 · 784×128) |
| `test_transpuestas_no_contiguas` | 4 | Funciona con `W.T` y `X.T` | En el backprop se multiplica por transpuestas, que son vistas no contiguas en memoria |
| `test_identidad_y_ceros` | 4 | `A·I = A` y `A·0 = 0` | Propiedades algebraicas básicas |
| `test_no_modifica_entradas` | 4 | A y B quedan intactas | La red reutiliza sus pesos después de multiplicar |
| `test_dimensiones_incompatibles` | 4 | Da `ValueError` si las formas no cuadran | Sin esto, Numba leería memoria fuera del arreglo sin avisar |
| `test_backend_inexistente` | 1 | Error claro con un nombre inválido | Mensajes útiles al usar la línea de comandos |

**¿Por qué `allclose` y no `==`?** La suma en punto flotante no es asociativa:
`(a+b)+c` puede diferir de `a+(b+c)` en el último bit. Cada versión suma en
otro orden, así que se usa una tolerancia: `1e-4` para float32 y `1e-10` para
float64.

### [`tests/test_bloques.py`](../tests/test_bloques.py): versión por bloques en paralelo

| Test | Veces | Qué comprueba | Por qué importa |
|---|---|---|---|
| `test_cualquier_tamano_de_bloque` | 7 | bs = 1, 2, 7, 16, 32, 64 y 500 con matrices de tamaños primos | Los bloques del borde quedan incompletos; bs = 500 es mayor que la matriz |
| `test_cualquier_numero_de_hilos` | 4 | Con 1, 2, 4 y 8 hilos el resultado es **idéntico bit a bit** | Si hubiera una condición de carrera entre hilos, el resultado cambiaría |
| `test_fabrica_fija_el_tamano` | 1 | `crear_matmul_bloques(16)` funciona | Es lo que recibe el MLP |
| `test_variante_ikj_del_experimento_de_cache` | 1 | La variante i-k-j es correcta | Se usa en el reto extra y no está en `BACKENDS` |
| `test_tamano_invalido` | 1 | bs = 0 da error | Validación de entrada |

### [`tests/test_gpu.py`](../tests/test_gpu.py): kernel CUDA (se saltan sin GPU)

| Test | Veces | Qué comprueba |
|---|---|---|
| `test_bordes_de_mosaico` | 3 | Matrices de exactamente un mosaico de 16×16, de un mosaico más un pedazo, y de 300×200×170 |
| `test_lanzar_con_datos_en_la_gpu` | 1 | El kernel funciona con datos que ya están en la GPU (es lo que se usa para medir sin las copias) |

Además, cuando hay GPU, el backend `gpu` entra automáticamente en **todos** los
tests de `test_matmul.py` y en la equivalencia de entrenamiento de `test_mlp.py`.

### [`tests/test_mlp.py`](../tests/test_mlp.py): la red neuronal

| Test | Qué comprueba | Por qué importa |
|---|---|---|
| `test_gradientes_coinciden_con_diferencias_finitas` | Cada derivada del backprop coincide con `(L(w+h) − L(w−h)) / 2h`, con error relativo menor a 1e-5 | **Es la prueba estándar de que el backpropagation está bien.** La aproximación numérica no puede equivocarse en la regla de la cadena |
| `test_backend_equivalente_a_numpy` (×3) | Entrenar con ingenua, bloques o gpu da los mismos pesos, pérdidas y predicciones que con NumPy (misma semilla) | Requisito del proyecto: cambiar el backend no cambia lo que aprende la red |
| `test_softmax_estable_y_normalizado` | Con logits de 1000 no aparece `inf` y las probabilidades suman 1 | Comprueba el truco de restar el máximo |
| `test_la_perdida_baja_y_aprende` | La pérdida baja y la precisión supera el 90 % en datos sintéticos | Prueba de sanidad del entrenamiento |
| `test_misma_semilla_mismo_resultado` | Dos entrenamientos idénticos dan pesos idénticos | Reproducibilidad |

Estos tests usan **datos sintéticos** pequeños, no MNIST, para correr en
segundos sin descargar nada.

### [`tests/test_metricas.py`](../tests/test_metricas.py): fórmulas de rendimiento

| Test | Qué comprueba |
|---|---|
| `test_recupera_f_conocida` (×5) | Se generan speedups "perfectos" con f = 0, 0.5, 0.9, 0.99 y 1, y se verifica que el despeje puntual y el ajuste recuperen esa f |
| `test_casos_limite_de_amdahl` | f = 1 da S = p; f = 0 da S = 1; con infinitos hilos S tiende a 1/(1−f) |
| `test_f_puntual_indefinida_con_un_hilo` | Con p = 1, f no está definida (resultado NaN) |
| `test_cronometrar_calienta_y_repite` | Hace 1 llamada de calentamiento más N medidas |

Si estas fórmulas estuvieran mal, todas las conclusiones del reporte lo
estarían, y por eso también se prueban.

---

## 2. Pruebas de rendimiento (experimentos)

Todas siguen las mismas reglas
([`experiments/medicion.py`](../experiments/medicion.py)):
- **calentamiento** antes de medir, para excluir la compilación JIT;
- cronómetro **`time.perf_counter`**;
- **5 repeticiones** por configuración, con **media y desviación estándar
  muestral**.

| Experimento | Comando | Qué mide | Resultado en `results/` |
|---|---|---|---|
| Especificaciones | `python -m experiments.especificaciones` | CPU, núcleos, RAM, GPU y versiones | `especificaciones.json` |
| Escalabilidad | `python -m experiments.bench_hilos --reps 5` | 1, 2, 4 y 8 hilos en una matmul de 1024×1024 y en una época del MLP: tiempo, S, E y f | `hilos_crudo.csv`, `hilos_resumen.csv`, `amdahl.csv` |
| Reto de caché | `python -m experiments.bench_cache` | 1 hilo: ingenua, i-k-j y bloques (bs de 8 a 256) con n de 128 a 1024, en GFLOP/s | `cache.csv` |
| GPU | `python -m experiments.bench_gpu` | Kernel contra copias CPU↔GPU contra NumPy, con n de 64 a 4096 | `bench_gpu.csv` |
| Entrenamiento | `python -m experiments.entrenar --backend X` | Precisión y tiempo de 10 épocas | `entrenamiento_*.json` |
| Todo junto | `python -m experiments.correr_todo --reps 5` | Todo lo anterior y las gráficas | todos los PNG y `resumen.md` |

### Resultados obtenidos (i5-13420H, RTX 4050, Windows 11)

**Precisión** (10 épocas):

| Conjunto | Backends | Precisión |
|---|---|---|
| MNIST | ingenua, bloques, numpy | 97.91 % |
| MNIST | gpu | 97.99 % |
| Fashion-MNIST | numpy | 86.85 % |

**Escalabilidad** (versión por bloques):

| Hilos | matmul 1024: S | matmul: E | MLP: S | MLP: E |
|---|---|---|---|---|
| 2 | 1.93 | 0.96 | 1.52 | 0.76 |
| 4 | 3.68 | 0.92 | 1.74 | 0.43 |
| 8 | 2.98 | 0.37 | 1.67 | 0.21 |
| **f (Amdahl)** | **0.87** | | **0.53** | |

**Reto de caché** (1 hilo, n = 1024):

| Variante | GFLOP/s |
|---|---|
| ingenua | 0.83 |
| i-k-j sin bloques | 31.0 |
| bloques (mejor bs) | 38.9 |

**GPU:** el kernel llega a unos 900 GFLOP/s. Con n = 64 las copias son el 90 %
del tiempo.

Las tablas completas están en [`results/resumen.md`](../results/resumen.md) y
las gráficas en [`results/`](../results/).

### Pruebas manuales que se hicieron durante el desarrollo
- Entrenamiento completo con cada backend: ingenua, bloques, numpy y gpu.
- Construcción de la imagen de Docker; dentro del contenedor pasan 101 tests y
  se saltan los 4 de GPU, y `correr_todo --rapido` genera los resultados.
- Ejecución completa de la demostración en vivo (`demo_en_vivo --sin-pausas`).
- La gráfica del notebook de Colab se probó localmente. **El notebook completo
  falta probarlo en Colab con GPU T4.**
