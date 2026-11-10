"""Genera las gráficas y un resumen en Markdown a partir de los CSV de results/.

Gráficas:
    tiempo_vs_nucleos.png   tiempo (media ± desv. est.) vs. hilos
    speedup.png             speedup real vs. ideal (y curva de Amdahl ajustada)
    eficiencia.png          eficiencia E = S/p vs. hilos
    cache_variantes.png     reto extra: GFLOP/s de ingenua, ikj y bloques (1 hilo)
    cache_tam_bloque.png    reto extra: efecto del tamaño de bloque (1 hilo)
    gpu.png                 GPU: kernel vs. total con copias vs. NumPy
Resumen:
    resumen.md              tablas listas para pegar en el reporte

Uso:  python -m experiments.graficas
"""
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")  # genera archivos sin abrir ventanas (sirve en Docker)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.medicion import speedup_amdahl  # noqa: E402

DIR = Path(__file__).resolve().parent.parent / "results"

# Paleta categórica en orden FIJO (validada para daltonismo). Cada entidad
# conserva su color en todas las gráficas: matmul siempre azul, mlp
# siempre naranja, etc.
AZUL, NARANJA, AQUA, AMARILLO, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRIS_REF = "#8a8984"      # referencias "ideales": recesivas, en gris
TEXTO, TEXTO_2 = "#0b0b0b", "#52514e"
# Rampa secuencial de un solo tono (azul claro -> oscuro) para magnitudes
RAMPA_AZUL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

COLOR_CARGA = {"matmul": AZUL, "mlp": NARANJA}
NOMBRE_CARGA = {"matmul": "matmul 1024×1024", "mlp": "MLP: 1 época de MNIST"}
COLOR_BACKEND = {"ingenua": NARANJA, "numpy": AQUA, "gpu": AMARILLO}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.edgecolor": TEXTO_2, "axes.labelcolor": TEXTO, "text.color": TEXTO,
    "xtick.color": TEXTO_2, "ytick.color": TEXTO_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.8,
    "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


def _eje_hilos(ax, hilos):
    """Eje x en escala log2: 1, 2, 4, 8 quedan equiespaciados."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(hilos, [str(h) for h in hilos])
    ax.minorticks_off()
    ax.set_xlabel("hilos (p)")


def _seg(t):
    """Formato legible de tiempo: 0.0079 -> '7.9 ms', 2.59 -> '2.59 s'."""
    return f"{t * 1e3:.3g} ms" if t < 1 else f"{t:.3g} s"


def _titulo_equipo():
    """Subtítulo con el equipo: una gráfica sin contexto de hardware no se
    puede interpretar."""
    archivo = DIR / "especificaciones.json"
    if not archivo.exists():
        return ""
    e = json.loads(archivo.read_text(encoding="utf-8"))
    return f"{e['cpu']} · {e['nucleos_fisicos']} núcleos físicos / {e['nucleos_logicos']} lógicos"


def graficas_hilos():
    archivo = DIR / "hilos_resumen.csv"
    if not archivo.exists():
        print("  (sin hilos_resumen.csv: corre experiments.bench_hilos)")
        return
    r = pd.read_csv(archivo)
    amdahl = pd.read_csv(DIR / "amdahl.csv").set_index("carga")["f_ajuste"]
    cargas = [c for c in ("matmul", "mlp") if c in set(r.carga)]
    sub = _titulo_equipo()

    # --- 1. Tiempo vs. núcleos (un panel por carga: escalas distintas) ---
    # Aquí las series son BACKENDS, así que el color sigue al backend
    # (bloques siempre azul) y no a la carga.
    fig, axes = plt.subplots(1, len(cargas), figsize=(5.6 * len(cargas), 4.6), squeeze=False,
                             layout="constrained")
    for ax, carga in zip(axes[0], cargas):
        d = r[r.carga == carga]
        b = d[d.backend == "bloques"].sort_values("hilos")
        hilos = b.hilos.astype(int).tolist()
        ax.errorbar(hilos, b.media, yerr=b["std"], color=AZUL, marker="o", capsize=3,
                    label="bloques (paralelo)")
        for h, t in zip(hilos, b.media):  # etiqueta directa de cada medición
            ax.annotate(_seg(t), (h, t), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=7, color=TEXTO_2)
        # Referencias que no dependen de p: líneas horizontales; su valor
        # va en la leyenda para que las etiquetas no se encimen.
        for backend, estilo in (("ingenua", "--"), ("numpy", "-."), ("gpu", ":")):
            fila = d[d.backend == backend]
            if fila.empty:
                continue
            t = fila.media.iloc[0]
            tipo = "secuencial, 1 hilo" if backend == "ingenua" else "referencia"
            ax.axhline(t, color=COLOR_BACKEND[backend], linestyle=estilo, linewidth=1.5,
                       label=f"{backend} ({tipo}): {_seg(t)}")
        ax.set_yscale("log")
        formato = matplotlib.ticker.FuncFormatter(lambda v, _: _seg(v))
        ax.yaxis.set_major_formatter(formato)
        y0, y1 = ax.get_ylim()
        if y1 / y0 < 20:  # menos de ~1 década: rotular también marcas menores
            ax.yaxis.set_minor_formatter(formato)
            ax.tick_params(axis="y", which="minor", labelsize=8)
        _eje_hilos(ax, hilos)
        ax.set_ylabel("tiempo (escala log)")
        ax.set_title(NOMBRE_CARGA.get(carga, carga), loc="left", fontweight="bold", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.suptitle(f"Tiempo vs. número de hilos (media ± desv. est.) — {sub}", fontsize=10)
    fig.savefig(DIR / "tiempo_vs_nucleos.png")
    plt.close(fig)

    # --- 2. Speedup real vs. ideal ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    hilos_todos = sorted(r[r.backend == "bloques"].hilos.astype(int).unique())
    ax.plot(hilos_todos, hilos_todos, color=GRIS_REF, linestyle="--", linewidth=1.5,
            label="ideal (S = p)")
    p_continuo = np.geomspace(1, max(hilos_todos), 100)
    for carga in cargas:
        b = r[(r.carga == carga) & (r.backend == "bloques")].sort_values("hilos")
        c = COLOR_CARGA[carga]
        ax.plot(b.hilos, b.speedup, color=c, marker="o", label=f"{NOMBRE_CARGA[carga]} (real)")
        f = amdahl[carga]
        ax.plot(p_continuo, speedup_amdahl(f, p_continuo), color=c, linestyle=":", linewidth=1.5,
                label=f"Amdahl ajustado, f = {f:.2f}")
        ax.annotate(f"{b.speedup.iloc[-1]:.2f}×", (b.hilos.iloc[-1], b.speedup.iloc[-1]),
                    xytext=(5, -3), textcoords="offset points", fontsize=8, color=TEXTO_2)
    _eje_hilos(ax, hilos_todos)
    ax.set_yscale("log", base=2)
    ax.set_yticks(hilos_todos, [str(h) for h in hilos_todos])
    ax.set_ylabel("speedup  S = T₁ / Tₚ")
    ax.set_title(f"Speedup real vs. ideal\n{sub}", loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(DIR / "speedup.png")
    plt.close(fig)

    # --- 3. Eficiencia ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(1.0, color=GRIS_REF, linestyle="--", linewidth=1.5, label="ideal (E = 1)")
    for carga in cargas:
        b = r[(r.carga == carga) & (r.backend == "bloques")].sort_values("hilos")
        ax.plot(b.hilos, b.eficiencia, color=COLOR_CARGA[carga], marker="o", label=NOMBRE_CARGA[carga])
    _eje_hilos(ax, hilos_todos)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("eficiencia  E = S / p")
    ax.set_title(f"Eficiencia paralela\n{sub}", loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    fig.savefig(DIR / "eficiencia.png")
    plt.close(fig)
    print("  tiempo_vs_nucleos.png, speedup.png, eficiencia.png")


def graficas_cache():
    archivo = DIR / "cache.csv"
    if not archivo.exists():
        print("  (sin cache.csv: corre experiments.bench_cache)")
        return
    c = pd.read_csv(archivo)
    bl = c[c.variante == "bloques"]

    # --- 4. Variantes: GFLOP/s vs. n ---
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    series = [("ingenua (i-j-k)", c[c.variante == "ingenua"], NARANJA),
              ("i-k-j sin bloques", c[c.variante == "ikj"], AQUA),
              ("bloques, bs = 32 (defecto)", bl[bl.bloque == 32], AZUL)]
    mejor = bl.loc[bl.groupby("n").gflops.idxmax()]
    series.append(("bloques, mejor bs", mejor, MAGENTA))
    for nombre, d, color in series:
        d = d.sort_values("n")
        ax.plot(d.n, d.gflops, marker="o", color=color, label=nombre)
    for _, f in mejor.iterrows():  # etiqueta selectiva: qué bs ganó en cada n
        ax.annotate(f"bs={int(f.bloque)}", (f.n, f.gflops), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=7, color=TEXTO_2)
    ax.set_xlabel("n (matrices n × n, float32)")
    ax.set_ylabel("GFLOP/s (más es mejor)")
    ax.set_title("Reto extra: acceso a memoria con UN solo hilo", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(DIR / "cache_variantes.png")
    plt.close(fig)

    # --- 5. Tamaño de bloque: una línea por n (rampa secuencial) ---
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    tamanos = sorted(bl.n.unique())
    colores = RAMPA_AZUL[-len(tamanos):] if len(tamanos) <= len(RAMPA_AZUL) else None
    for i, n in enumerate(tamanos):
        d = bl[bl.n == n].sort_values("bloque")
        ax.plot(d.bloque, d.gflops, marker="o", color=colores[i] if colores else None, label=f"n = {n}")
    ax.set_xscale("log", base=2)
    bloques = sorted(bl.bloque.unique())
    ax.set_xticks(bloques, [str(int(b)) for b in bloques])
    ax.minorticks_off()
    ax.set_xlabel("tamaño de bloque (bs)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("Efecto del tamaño de bloque (1 hilo)", loc="left", fontsize=10)
    ax.legend(fontsize=8, title="tamaño de matriz", title_fontsize=8)
    fig.savefig(DIR / "cache_tam_bloque.png")
    plt.close(fig)
    print("  cache_variantes.png, cache_tam_bloque.png")


def grafica_gpu():
    archivo = DIR / "bench_gpu.csv"
    if not archivo.exists():
        print("  (sin bench_gpu.csv: corre experiments.bench_gpu)")
        return
    g = pd.read_csv(archivo)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(g.n, g.kernel_ms, marker="o", color=AZUL, label="GPU: solo kernel")
    ax.plot(g.n, g.total_ms, marker="o", color=NARANJA, label="GPU: total (con copias)")
    ax.plot(g.n, g.numpy_ms, marker="s", color=AQUA, linestyle="--", label="NumPy (CPU, BLAS)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(g.n, [str(n) for n in g.n])
    ax.minorticks_off()
    ax.set_xlabel("n (matrices n × n, float32)")
    ax.set_ylabel("tiempo (ms, escala log)")
    ax.set_title("GPU: cómputo vs. transferencias CPU↔GPU", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(DIR / "gpu.png")
    plt.close(fig)
    print("  gpu.png")


# ---------------------------------------------------------------------------
# Resumen en Markdown
# ---------------------------------------------------------------------------
def _texto(x):
    """iterrows convierte las filas a float: 128 se volvería "128.0"."""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def _tabla(df, columnas, formatos):
    enc = "| " + " | ".join(columnas) + " |\n|" + "---|" * len(columnas) + "\n"
    filas = ["| " + " | ".join(formatos[c](f[c]) if c in formatos else _texto(f[c]) for c in columnas) + " |"
             for _, f in df.iterrows()]
    return enc + "\n".join(filas) + "\n"


def resumen_md():
    partes = ["# Resumen de resultados\n", f"Equipo: {_titulo_equipo()}\n"]
    num = lambda x: "—" if pd.isna(x) else f"{x:.3f}"  # noqa: E731
    if (DIR / "hilos_resumen.csv").exists():
        r = pd.read_csv(DIR / "hilos_resumen.csv")
        r["hilos"] = r.hilos.apply(lambda h: "—" if pd.isna(h) else str(int(h)))
        for carga in r.carga.unique():
            d = r[r.carga == carga]
            partes.append(f"\n## {NOMBRE_CARGA.get(carga, carga)}\n")
            partes.append(_tabla(d, ["backend", "hilos", "media", "std", "speedup", "eficiencia",
                                     "f_puntual", "speedup_vs_ingenua"],
                                 {k: num for k in ["media", "std", "speedup", "eficiencia",
                                                   "f_puntual", "speedup_vs_ingenua"]}))
        a = pd.read_csv(DIR / "amdahl.csv")
        partes.append("\n## Ley de Amdahl (ajuste por mínimos cuadrados)\n")
        partes.append(_tabla(a, ["carga", "f_ajuste", "speedup_maximo"],
                             {"f_ajuste": num, "speedup_maximo": num}))
    if (DIR / "cache.csv").exists():
        c = pd.read_csv(DIR / "cache.csv")
        piv = c[c.variante != "bloques"].pivot(index="n", columns="variante", values="gflops")
        mejor = c[c.variante == "bloques"].sort_values("gflops").groupby("n").tail(1).set_index("n")
        piv["bloques (mejor)"] = mejor.gflops
        piv["mejor bs"] = mejor.bloque.astype(int)
        partes.append("\n## Reto extra: GFLOP/s con 1 hilo\n")
        piv = piv.reset_index()
        piv["n"] = piv.n.astype(int)
        piv = piv[["n", "ingenua", "ikj", "bloques (mejor)", "mejor bs"]]
        partes.append(_tabla(piv, list(piv.columns),
                             {k: (lambda x: f"{x:.2f}") for k in ("ingenua", "ikj", "bloques (mejor)")}))
    (DIR / "resumen.md").write_text("".join(partes), encoding="utf-8")
    print("  resumen.md")


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    print("Generando gráficas en results/:")
    graficas_hilos()
    graficas_cache()
    grafica_gpu()
    resumen_md()


if __name__ == "__main__":
    main()
