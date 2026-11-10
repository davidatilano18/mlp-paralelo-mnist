"""Carga de MNIST y Fashion-MNIST.

sklearn se usa SOLO para descargar los datos (fetch_openml); la red está
implementada por nosotros. La primera descarga tarda (~50 MB); después
se guarda una copia comprimida en data/ y se carga en menos de un segundo.

Ambos conjuntos tienen el mismo formato: 70 000 imágenes de 28x28 en
escala de grises, aplanadas a vectores de 784 valores, con 10 clases.
Los primeros 60 000 son el conjunto de entrenamiento oficial y los
últimos 10 000 el de prueba; respetamos esa división para que nuestros
resultados sean comparables con los publicados.
"""
from pathlib import Path

import numpy as np

# Nombres de los conjuntos en OpenML
CONJUNTOS = {
    "mnist": "mnist_784",
    "fashion": "Fashion-MNIST",
}

# Etiquetas legibles de Fashion-MNIST (para gráficas y reportes)
CLASES_FASHION = [
    "Playera", "Pantalón", "Suéter", "Vestido", "Abrigo",
    "Sandalia", "Camisa", "Tenis", "Bolsa", "Botín",
]

DIR_DATOS = Path(__file__).resolve().parent.parent / "data"
N_ENTRENAMIENTO = 60_000


def _descargar(nombre):
    """Descarga desde OpenML y guarda en data/<nombre>.npz."""
    from sklearn.datasets import fetch_openml  # import tardío: solo si hace falta

    print(f"Descargando {CONJUNTOS[nombre]} desde OpenML (solo la primera vez)...")
    # parser="liac-arff": el parser por defecto usa pandas y falla con
    # pandas 3.x en esta versión de sklearn; este no depende de pandas.
    X, y = fetch_openml(CONJUNTOS[nombre], version=1, as_frame=False,
                        return_X_y=True, parser="liac-arff")
    # Guardamos como uint8 (0-255): ocupa 4 veces menos que float32.
    X = X.astype(np.uint8)
    y = y.astype(np.int64)  # OpenML entrega las etiquetas como texto "0".."9"
    DIR_DATOS.mkdir(exist_ok=True)
    np.savez_compressed(DIR_DATOS / f"{nombre}.npz", X=X, y=y)
    return X, y


def cargar(nombre="mnist", dtype=np.float32):
    """Devuelve (X_ent, y_ent, X_prueba, y_prueba).

    X: matriz (muestras, 784) con valores en [0, 1].
    y: vector de enteros 0..9.

    ¿Por qué dividir entre 255? Los pixeles van de 0 a 255. Con entradas
    tan grandes, las sumas ponderadas de la primera capa serían enormes,
    los gradientes también, y el SGD diverge o necesita una tasa de
    aprendizaje diminuta. Escalar a [0, 1] mantiene todo en un rango sano.
    """
    if nombre not in CONJUNTOS:
        raise ValueError(f"Conjunto '{nombre}' desconocido. Opciones: {list(CONJUNTOS)}")

    archivo = DIR_DATOS / f"{nombre}.npz"
    if archivo.exists():
        with np.load(archivo) as d:
            X, y = d["X"], d["y"]
    else:
        X, y = _descargar(nombre)

    X = X.astype(dtype) / dtype(255.0)
    n = N_ENTRENAMIENTO
    return X[:n], y[:n], X[n:], y[n:]
