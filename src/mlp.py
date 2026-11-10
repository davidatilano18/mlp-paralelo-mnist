"""Perceptrón multicapa (MLP) implementado desde cero.

Arquitectura por defecto: 784 -> 128 -> 64 -> 10
    - Capas ocultas con activación ReLU.
    - Capa de salida con softmax + entropía cruzada.
    - Entrenamiento con SGD por mini-lotes.

Punto clave del proyecto: TODAS las multiplicaciones de matrices (forward
y backward) pasan por self.matmul, una función que se recibe como
parámetro. Así podemos entrenar exactamente la misma red con la versión
ingenua, la de bloques, la de GPU o la de NumPy, y medir solo la
diferencia del backend.

Notación (para un mini-lote de B ejemplos):
    X      (B, 784)   entradas
    W[l]   (n_in, n_out) pesos de la capa l;  b[l] (n_out,) sesgos
    Z[l] = A[l-1] @ W[l] + b[l]     "pre-activación"
    A[l] = ReLU(Z[l])               (A[0] = X)
    P    = softmax(Z[última])       probabilidades por clase
"""
import numpy as np

from src.matmul import matmul_numpy


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------
def relu(Z):
    """ReLU(z) = max(0, z). Es barata y su derivada (0 o 1) no se "apaga"
    para valores grandes como la sigmoide, lo que facilita entrenar."""
    return np.maximum(Z, 0)


def softmax(Z):
    """Convierte puntuaciones (logits) en probabilidades que suman 1.

    Truco de estabilidad numérica: restar el máximo de cada fila antes del
    exponencial. Matemáticamente no cambia el resultado
    (e^(z-c) / sum e^(z-c) = e^z / sum e^z), pero evita que e^z se
    desborde a infinito cuando algún logit es grande (p. ej. e^100).
    """
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def entropia_cruzada(P, y):
    """Pérdida promedio: -log(probabilidad asignada a la clase correcta).

    Si la red está segura y acierta (P = 1), la pérdida es 0; si le da
    probabilidad casi 0 a la clase correcta, la pérdida crece mucho.
    El 1e-12 evita log(0) = -infinito.
    """
    B = y.shape[0]
    return float(-np.mean(np.log(P[np.arange(B), y] + 1e-12)))


# ---------------------------------------------------------------------------
# La red
# ---------------------------------------------------------------------------
class MLP:
    def __init__(self, capas=(784, 128, 64, 10), matmul=matmul_numpy,
                 semilla=0, dtype=np.float32):
        """
        capas:   tamaños de cada capa, incluida la entrada y la salida.
        matmul:  función (A, B) -> A @ B que se usará en toda la red.
        semilla: fija la inicialización y el barajado de los datos, para
                 que dos entrenamientos con distinto backend partan de
                 exactamente los mismos pesos y vean los lotes en el mismo
                 orden (requisito para comparar backends).
        """
        self.capas = tuple(capas)
        self.matmul = matmul
        self.dtype = dtype
        self.rng = np.random.default_rng(semilla)

        # Inicialización de He: W ~ Normal(0, sqrt(2 / n_entrada)).
        # ¿Por qué? Si los pesos son muy grandes, las activaciones crecen
        # capa tras capa y explotan; si son muy chicos, se desvanecen. Con
        # ReLU (que anula ~la mitad de las neuronas) el factor 2/n_entrada
        # mantiene la varianza de las activaciones estable entre capas.
        # Los sesgos empiezan en 0: no hay simetría que romper en ellos.
        self.W, self.b = [], []
        for n_in, n_out in zip(self.capas[:-1], self.capas[1:]):
            escala = np.sqrt(2.0 / n_in)
            self.W.append((self.rng.standard_normal((n_in, n_out)) * escala).astype(dtype))
            self.b.append(np.zeros(n_out, dtype=dtype))

    # ------------------------------------------------------------------ forward
    def forward(self, X):
        """Propagación hacia adelante. Devuelve las probabilidades P y guarda
        los valores intermedios (A y Z de cada capa) porque el backward los
        necesita para calcular los gradientes."""
        A = X
        self._A = [X]   # activaciones: entrada de cada capa
        self._Z = []    # pre-activaciones: para saber dónde ReLU fue > 0
        ultima = len(self.W) - 1
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            Z = self.matmul(A, W) + b   # b se "difunde" (broadcast) a cada fila
            self._Z.append(Z)
            A = softmax(Z) if l == ultima else relu(Z)
            if l != ultima:
                self._A.append(A)
        return A

    # ----------------------------------------------------------------- backward
    def backward(self, P, y):
        """Backpropagation: calcula dPérdida/dW y dPérdida/db de cada capa
        aplicando la regla de la cadena desde la salida hacia la entrada.

        Devuelve listas (gW, gb) alineadas con self.W y self.b.
        """
        B = y.shape[0]

        # 1) Gradiente en la salida. Softmax + entropía cruzada juntas tienen
        #    una derivada muy simple:  dL/dZ = P - Y   (Y = one-hot de y).
        #    Dividimos entre B porque la pérdida es un PROMEDIO del lote.
        dZ = P.copy()
        dZ[np.arange(B), y] -= 1
        dZ /= B

        gW = [None] * len(self.W)
        gb = [None] * len(self.W)
        for l in range(len(self.W) - 1, -1, -1):
            A_prev = self._A[l]
            # 2) Z = A_prev @ W + b  =>  dL/dW = A_prev^T @ dZ
            #    (cada peso W[i, j] contribuye con A_prev[:, i] * dZ[:, j],
            #    sumado sobre los ejemplos del lote: eso es un matmul).
            gW[l] = self.matmul(A_prev.T, dZ)
            #    dL/db = suma de dZ sobre el lote (b se sumó a cada fila).
            gb[l] = dZ.sum(axis=0)
            if l > 0:
                # 3) Propagamos el error a la capa anterior:
                #    dL/dA_prev = dZ @ W^T
                dA = self.matmul(dZ, self.W[l].T)
                # 4) A través de ReLU: la derivada es 1 donde Z > 0 y 0 en
                #    otro caso, así que solo pasa el gradiente de las
                #    neuronas que estaban activas.
                dZ = dA * (self._Z[l - 1] > 0)
        return gW, gb

    # ------------------------------------------------------------ entrenamiento
    def paso_sgd(self, X, y, tasa):
        """Un paso de descenso por gradiente con un mini-lote.
        Devuelve la pérdida del lote (antes de actualizar)."""
        P = self.forward(X)
        perdida = entropia_cruzada(P, y)
        gW, gb = self.backward(P, y)
        # Movernos un poco en la dirección contraria al gradiente,
        # que es la dirección en la que la pérdida baja más rápido.
        for l in range(len(self.W)):
            self.W[l] -= tasa * gW[l]
            self.b[l] -= tasa * gb[l]
        return perdida

    def entrenar(self, X, y, epocas=10, tam_lote=64, tasa=0.1,
                 X_val=None, y_val=None, verbose=True):
        """SGD por mini-lotes.

        ¿Por qué mini-lotes y no un ejemplo a la vez (SGD puro) ni todo el
        conjunto (descenso por gradiente completo)?
          - Un ejemplo: gradiente muy ruidoso y matmuls de 1 fila, que no
            aprovechan el paralelismo.
          - Todo el conjunto: un solo paso por época, muy lento para aprender.
          - Mini-lote (64): gradiente razonablemente estable y matrices lo
            bastante grandes para que paralelizar el matmul valga la pena.

        En cada época se barajan los datos para que los lotes sean
        distintos y la red no aprenda el orden de los ejemplos.
        """
        historial = {"perdida": [], "precision_val": []}
        n = X.shape[0]
        for epoca in range(1, epocas + 1):
            orden = self.rng.permutation(n)
            perdidas = []
            for inicio in range(0, n, tam_lote):
                idx = orden[inicio:inicio + tam_lote]
                perdidas.append(self.paso_sgd(X[idx], y[idx], tasa))
            historial["perdida"].append(float(np.mean(perdidas)))

            msg = f"Época {epoca:2d}/{epocas}  pérdida={historial['perdida'][-1]:.4f}"
            if X_val is not None:
                acc = self.precision(X_val, y_val)
                historial["precision_val"].append(acc)
                msg += f"  precisión_val={acc:.2%}"
            if verbose:
                print(msg, flush=True)
        return historial

    # --------------------------------------------------------------- evaluación
    def predecir(self, X, tam_lote=1000):
        """Clase más probable para cada fila de X. Se procesa por lotes para
        no crear de golpe matrices intermedias de 10 000 x 128."""
        salida = [np.argmax(self.forward(X[i:i + tam_lote]), axis=1)
                  for i in range(0, X.shape[0], tam_lote)]
        return np.concatenate(salida)

    def precision(self, X, y):
        """Fracción de ejemplos clasificados correctamente."""
        return float(np.mean(self.predecir(X) == y))
