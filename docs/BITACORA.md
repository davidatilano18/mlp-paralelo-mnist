# Bitácora del proyecto: qué se hizo, paso a paso

Este documento cuenta **en orden** cómo se construyó el proyecto en 6 etapas.
Cada etapa se revisó en equipo antes de pasar a la siguiente. Para cada una
se indica:

- el **objetivo**;
- los **archivos** que se crearon y las partes del código que conviene leer
  (los enlaces llevan a la línea exacta);
- los **problemas** que aparecieron y cómo se resolvieron;
- **cómo comprobarlo** uno mismo.

El catálogo detallado de todas las pruebas está en [PRUEBAS.md](PRUEBAS.md).

> **Sugerencia para estudiar:** abran el archivo enlazado y lean los
> comentarios del código. Cada función explica en español *por qué* se hace
> así, no solo *qué* hace.

---

## Etapa 1: estructura, matmul ingenua y tests

**Objetivo:** tener la multiplicación de matrices más simple posible, que
será la **línea base** de todas las mediciones, y la forma de comprobar que
cualquier versión futura da el resultado correcto.

**Qué se hizo**

| Archivo | Qué contiene |
|---|---|
| [`src/matmul.py`](../src/matmul.py#L32) | `_matmul_ingenua_kernel`: triple ciclo `i-j-k` compilado con `@njit` |
| [`src/validacion.py`](../src/validacion.py) | `_validar` (revisa dimensiones) y `_preparar` (tipo común y memoria contigua) |
| [`src/matmul.py`](../src/matmul.py#L211) | `matmul_numpy`: `A @ B`, la referencia "correcta" |
| [`src/matmul.py`](../src/matmul.py#L228) | `BACKENDS`: diccionario nombre → función |
| [`tests/test_matmul.py`](../tests/test_matmul.py) | comparación de cada backend contra NumPy |

**Decisiones de diseño**
- **Todas las versiones con la misma firma** `matmul(A, B) -> C`. Así la red
  puede recibir cualquiera sin cambiar su código.
- **Compilar también la versión secuencial** con Numba. Sin eso estaríamos
  comparando "Python lento contra C rápido", no algoritmos.
- **El registro `BACKENDS`:** los tests recorren ese diccionario, así que
  cada versión nueva que se registre queda probada automáticamente.
- **El acumulador del ciclo** se inicializa con `C[i, j]` y no con `0.0`. Así
  tiene el mismo tipo que la matriz (float32), y la comparación con la
  versión por bloques es justa.

**Cómo comprobarlo:** `python -m pytest tests/test_matmul.py -v`

---

## Etapa 2: MLP secuencial que supera el 95 % en MNIST

**Objetivo:** una red 784 → 128 → 64 → 10 escrita desde cero, que reciba la
función matmul como parámetro.

**Qué se hizo**

| Archivo / parte | Qué leer |
|---|---|
| [`src/datos.py`](../src/datos.py#L50) | `cargar()`: descarga con `fetch_openml`, guarda en `data/`, escala los pixeles a [0, 1] y separa 60 000 / 10 000 |
| [`src/mlp.py` · `softmax`](../src/mlp.py#L35) | resta del máximo para que `exp` no se desborde |
| [`src/mlp.py` · `__init__`](../src/mlp.py#L63) | inicialización de He, `√(2/n)` |
| [`src/mlp.py` · `forward`](../src/mlp.py#L91) | `Z = A·W + b` y luego ReLU (softmax en la salida); guarda A y Z para el backward |
| [`src/mlp.py` · `backward`](../src/mlp.py#L108) | **backpropagation**: `dZ = P − Y`, `dW = Aᵀ·dZ`, `dA = dZ·Wᵀ` y la derivada de ReLU |
| [`src/mlp.py` · `entrenar`](../src/mlp.py#L157) | SGD por mini-lotes, barajando los datos en cada época |
| [`experiments/entrenar.py`](../experiments/entrenar.py) | script de línea de comandos para entrenar con cualquier backend |
| [`tests/test_mlp.py`](../tests/test_mlp.py) | verificación de gradientes y equivalencia entre backends |

**Resultado:** **97.91 %** en MNIST, igual con la versión ingenua que con
NumPy. Fashion-MNIST: 86.85 %.

**Problema encontrado:** `fetch_openml` fallaba con pandas 3.
**Solución:** usar `parser="liac-arff"`, que no depende de pandas
([`src/datos.py`](../src/datos.py#L38)).

**Cómo comprobarlo**
```
python -m pytest tests/test_mlp.py -v
python -m experiments.entrenar --backend ingenua
```

---

## Etapa 3: matmul por bloques en paralelo

**Objetivo:** dividir la multiplicación en mosaicos que quepan en la caché y
repartirlos entre hilos con `prange`.

**Qué se hizo**

| Archivo / parte | Qué leer |
|---|---|
| [`src/matmul.py` · `_matmul_bloques_kernel`](../src/matmul.py#L115) | el kernel completo; el comentario explica las 3 ideas (caché, orden i-k-j y paralelismo sin condiciones de carrera) |
| [`src/matmul.py` · `crear_matmul_bloques`](../src/matmul.py#L195) | "fábrica" que fija el tamaño de bloque y conserva la firma `matmul(A, B)` |
| [`tests/test_bloques.py`](../tests/test_bloques.py) | resultado igual con cualquier tamaño de bloque y número de hilos |

**Decisiones de diseño**
- **Repartir mosaicos de C, no filas.** En el MLP las matrices tienen solo 64
  filas: repartiendo bloques de 32 filas habría 2 tareas para 8 hilos. Al
  numerar los mosaicos hay más trabajo que repartir.
- **Sin candados:** cada mosaico de C lo escribe un solo hilo.

**Problema encontrado (importante):** la primera versión casi no mejoraba con
un hilo, y el tamaño de bloque no cambiaba nada. Eso indicaba que el problema
no era la caché: con índices 2D (`C[i, j]`), el compilador **no vectorizaba**
el ciclo interno.
**Solución:** usar rebanadas 1D de cada fila (`C_fila[j] += a * B_fila[j]`).
Resultado: **unas 7 veces más rápido** con un hilo.

**Cómo comprobarlo**
```
python -m pytest tests/test_bloques.py -v
python -m experiments.entrenar --backend bloques --hilos 4
```

---

## Etapa 4: GPU con numba.cuda

**Objetivo:** la misma multiplicación en la GPU, usable en la RTX 4050 de la
laptop y en Google Colab.

**Qué se hizo**

| Archivo / parte | Qué leer |
|---|---|
| [`src/matmul_gpu.py` · `kernel`](../src/matmul_gpu.py#L52) | mosaicos de 16×16 en **memoria compartida**, `syncthreads()` y acceso coalescente |
| [`src/matmul_gpu.py` · `matmul_gpu`](../src/matmul_gpu.py#L123) | las 3 fases: copiar a la GPU, ejecutar el kernel y copiar de regreso |
| [`src/matmul_gpu.py` · `_lanzar`](../src/matmul_gpu.py#L112) | lanza el kernel sin copias (para medir solo el cómputo) |
| [`experiments/bench_gpu.py`](../experiments/bench_gpu.py) | separa el tiempo del kernel del de las copias |
| [`notebooks/mlp_gpu_colab.ipynb`](../notebooks/mlp_gpu_colab.ipynb) | versión para Colab |
| [`tests/test_gpu.py`](../tests/test_gpu.py) | bordes de mosaico y lanzamiento directo (se saltan si no hay GPU) |

**Problemas encontrados**
1. **Faltaba el compilador de CUDA (libNVVM).** Se instaló `numba-cuda[cu13]`,
   que lo trae por pip, así que no hace falta el CUDA Toolkit.
2. **`numba-cuda` no funcionaba con NumPy 2.5** porque usa `np.row_stack`,
   que se eliminó. Se fijó NumPy en 2.4.6.
3. **Import circular** entre `matmul.py` y `matmul_gpu.py`. Se resolvió
   moviendo la validación a [`src/validacion.py`](../src/validacion.py).

**Resultado clave:** el kernel llega a unos 900 GFLOP/s, pero entrenar el MLP
en GPU tarda unos 5 s por época, **más lento** que bloques en CPU (~1 s).
Con lotes de 64 las matrices son pequeñas y el 90 % del tiempo se va en
copiar datos por PCIe.

---

## Etapa 5: benchmarks y gráficas

**Objetivo:** medir con rigor (1, 2, 4 y 8 hilos, repeticiones, media y
desviación estándar), calcular speedup, eficiencia y Amdahl, y hacer el reto
extra de caché.

**Qué se hizo**

| Archivo | Qué hace |
|---|---|
| [`experiments/medicion.py`](../experiments/medicion.py) | `cronometrar` (calentamiento y repeticiones) y fórmulas de Amdahl (despeje puntual y ajuste por mínimos cuadrados) |
| [`experiments/especificaciones.py`](../experiments/especificaciones.py) | registra CPU, núcleos, RAM y GPU en `results/especificaciones.json` |
| [`experiments/bench_hilos.py`](../experiments/bench_hilos.py) | dos cargas: matmul 1024×1024 y una época del MLP |
| [`experiments/bench_cache.py`](../experiments/bench_cache.py) | reto extra con 1 hilo: ingenua, i-k-j sin bloques y bloques con varios tamaños |
| [`src/matmul.py` · `_matmul_ikj_kernel`](../src/matmul.py#L81) | variante de apoyo para separar el efecto del orden de acceso del efecto de los bloques |
| [`experiments/graficas.py`](../experiments/graficas.py) | genera los PNG y `results/resumen.md` |
| [`experiments/correr_todo.py`](../experiments/correr_todo.py) | corre todo en orden |
| [`tests/test_metricas.py`](../tests/test_metricas.py) | comprueba que las fórmulas de Amdahl recuperen una f conocida |

**Problemas encontrados**
1. **La GPU midió hasta 5 veces más lento en la corrida completa.** Después
   de minutos de CPU al 100 %, la GPU de la laptop arranca con el reloj bajo,
   porque comparten el límite de energía. **Solución:** un calentamiento
   sostenido de 1.5 s antes de medir (`llevar_a_reloj_de_trabajo` en
   [`bench_gpu.py`](../experiments/bench_gpu.py)).
2. **La referencia de NumPy en el MLP varía mucho**, porque usa hilos de
   OpenBLAS que no controlamos. Se documentó como limitación; no afecta las
   métricas de Amdahl.

**Resultados principales:** ver la sección "Resultados y discusión" del
[README](../README.md#resultados-y-discusión) y
[`results/resumen.md`](../results/resumen.md).

---

## Etapa 6: README y Docker

| Archivo | Qué contiene |
|---|---|
| [`README.md`](../README.md) | instalación, comandos, diseño, metodología, resultados y uso de IA |
| [`Dockerfile`](../Dockerfile) | imagen reproducible (solo CPU) |
| [`.dockerignore`](../.dockerignore) | evita copiar `.venv`, datos y cachés a la imagen |

**Verificación:** dentro del contenedor pasan 101 tests y se saltan los 4 de
GPU. `correr_todo --rapido` genera todos los resultados.

---

## Extra: demostración en vivo

[`experiments/demo_en_vivo.py`](../experiments/demo_en_vivo.py): un script
de 4 partes, una por integrante, que se detiene con Enter. Mide en vivo y
describe lo que acaba de medir, porque los números varían un poco de una
corrida a otra.
