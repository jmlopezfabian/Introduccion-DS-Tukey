"""
Script maestro para generar todas las imágenes de los módulos del curso.
Ejecutar desde la raíz del proyecto: python gen_assets.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

# ─── Paleta de colores coherente con el tema oscuro del sitio ──────────────
DARK = "#0f0f11"
BG_ALT = "#17171b"
BORDER = "#2a2a30"
ACCENT = "#6c7fea"
RED = "#ea6c6c"
TEXT = "#e2e2e8"
TEXT2 = "#8a8a9a"
TEXT3 = "#525260"

plt.rcParams.update({
    "figure.facecolor": DARK,
    "axes.facecolor":   DARK,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT2,
    "xtick.color":      TEXT2,
    "ytick.color":      TEXT2,
    "text.color":       TEXT,
    "grid.color":       BORDER,
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "legend.facecolor": BG_ALT,
    "legend.edgecolor": BORDER,
})

SAVEFIG_KW = dict(dpi=150, bbox_inches="tight", facecolor=DARK)

def save(path):
    plt.savefig(path, **SAVEFIG_KW)
    print(f"  ✓ {path}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO: visualizacion
# ──────────────────────────────────────────────────────────────────────────────
def gen_visualizacion():
    D = Path("docs/assets/visualizacion")
    D.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-10, 10, 50)
    y1 = 3 * x**3 + x**2 + 10
    y2 = np.sin(x) * 1000

    # --- lineas.png ---
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, y1, color=ACCENT, label="y₁ = 3x³ + x² + 10", linestyle="--", marker="o", markersize=3)
    ax.plot(x, y2, color=RED,    label="y₂ = sin(x) × 1000",  linestyle="--", marker="^", markersize=3)
    ax.set_title("Dos funciones graficadas en el mismo eje", fontsize=12)
    ax.set_xlabel("Eje x")
    ax.set_ylabel("Eje y")
    ax.legend()
    ax.grid(True)
    save(D / "lineas.png")

    # --- subplots.png ---
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    fig.suptitle("Subplots: múltiples gráficas en una figura", fontsize=12)

    axes[0, 0].plot(x, y1, color=ACCENT)
    axes[0, 0].set_title("y₁ = 3x³ + x² + 10")
    axes[0, 0].set_xlabel("x")

    axes[0, 1].plot(x, y2, color=RED)
    axes[0, 1].set_title("y₂ = sin(x) × 1000")
    axes[0, 1].set_xlabel("x")

    axes[1, 0].scatter(x, y1, color=ACCENT, s=18)
    axes[1, 0].set_title("y₁ como scatter")

    axes[1, 1].bar(np.arange(10), np.abs(y2[:10]), color=ACCENT, edgecolor=BORDER)
    axes[1, 1].set_title("Primeros 10 valores de |y₂|")

    plt.tight_layout()
    save(D / "subplots.png")

    print("→ visualizacion: OK")


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO: gradiente
# ──────────────────────────────────────────────────────────────────────────────
def gen_gradiente():
    D = Path("docs/assets/gradiente")
    D.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("data/csvs/examenes.csv").sample(n=100, random_state=42)
    x = np.array(df["study_hours"])
    y = np.array(df["exam_score"])

    # --- scatter-datos.png ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color=ACCENT, alpha=0.7, s=30, edgecolors=BORDER, linewidths=0.5)
    ax.set_title("Horas de estudio vs calificación del examen", fontsize=12)
    ax.set_xlabel("Horas de estudio")
    ax.set_ylabel("Calificación")
    ax.grid(True)
    save(D / "scatter-datos.png")

    # --- ecm-pendiente.png ---
    def ecm(m, b, x, y):
        return ((y - (m * x + b)) ** 2).mean()

    b_fijo = 40.0
    pendientes = np.linspace(-5, 15, 200)
    errores = [ecm(m, b_fijo, x, y) for m in pendientes]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pendientes, errores, color=ACCENT, linewidth=2)
    ax.set_title("ECM según el valor de la pendiente (b = 40 fijo)", fontsize=12)
    ax.set_xlabel("Pendiente m")
    ax.set_ylabel("Error Cuadrático Medio")
    ax.grid(True)
    save(D / "ecm-pendiente.png")

    # --- convergencia.png ---
    def gradiente_paso_m(m, alpha, b, x, y):
        return m - alpha * (2 * (x * (m * x + b - y)).mean())

    def gradiente_paso_b(b, alpha, m, x, y):
        return b - alpha * (2 * (m * x + b - y).mean())

    alpha = 0.01
    m_act, b_act = 0.0, 0.0
    hist_ecm = []
    for _ in range(300):
        hist_ecm.append(ecm(m_act, b_act, x, y))
        m_nuevo = gradiente_paso_m(m_act, alpha, b_act, x, y)
        b_nuevo = gradiente_paso_b(b_act, alpha, m_act, x, y)
        m_act, b_act = m_nuevo, b_nuevo

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Gradiente descendente: convergencia del ECM", fontsize=12)

    axes[0].plot(hist_ecm, color=ACCENT, linewidth=1.5)
    axes[0].set_title("ECM por iteración")
    axes[0].set_xlabel("Iteración")
    axes[0].set_ylabel("ECM")
    axes[0].grid(True)

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = m_act * x_line + b_act
    axes[1].scatter(x, y, color=TEXT2, alpha=0.5, s=20)
    axes[1].plot(x_line, y_line, color=ACCENT, linewidth=2,
                 label=f"ŷ = {m_act:.2f}x + {b_act:.2f}")
    axes[1].set_title("Recta ajustada por gradiente descendente")
    axes[1].set_xlabel("Horas de estudio")
    axes[1].set_ylabel("Calificación")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    save(D / "convergencia.png")

    print("→ gradiente: OK")


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO: regresion-simple
# ──────────────────────────────────────────────────────────────────────────────
def gen_regresion_simple():
    D = Path("docs/assets/regresion-simple")
    D.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("data/csvs/examenes.csv").sample(n=100, random_state=42)
    x = np.array(df["study_hours"])
    y = np.array(df["exam_score"])

    n = len(x)
    m = (n * (x * y).sum() - x.sum() * y.sum()) / (n * (x**2).sum() - x.sum()**2)
    b = (y.sum() - m * x.sum()) / n

    # --- scatter-regresion.png ---
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = m * x_line + b

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color=ACCENT, alpha=0.6, s=28, edgecolors=BORDER, linewidths=0.5, label="Datos reales")
    ax.plot(x_line, y_line, color=RED, linewidth=2, label=f"ŷ = {m:.2f}x + {b:.2f}")
    ax.set_title("Regresión lineal simple: horas de estudio → calificación", fontsize=11)
    ax.set_xlabel("Horas de estudio")
    ax.set_ylabel("Calificación")
    ax.legend()
    ax.grid(True)
    save(D / "scatter-regresion.png")

    # --- ecm-pendiente.png ---
    pendientes = np.linspace(-5, 15, 200)
    errores = [((y - (mi * x + b)) ** 2).mean() for mi in pendientes]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pendientes, errores, color=ACCENT, linewidth=2)
    ax.axvline(m, color=RED, linestyle="--", linewidth=1.5, label=f"m óptima = {m:.2f}")
    ax.set_title("Función de costo ECM en función de la pendiente", fontsize=11)
    ax.set_xlabel("Pendiente m")
    ax.set_ylabel("ECM")
    ax.legend()
    ax.grid(True)
    save(D / "ecm-pendiente.png")

    print("→ regresion-simple: OK")


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO: regresion-statsmodels
# ──────────────────────────────────────────────────────────────────────────────
def gen_regresion_statsmodels():
    D = Path("docs/assets/regresion-statsmodels")
    D.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("data/csvs/examenes.csv").sample(n=100, random_state=42)
    X = np.array(df[["study_hours"]])
    y = np.array(df["exam_score"])

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    x_line = np.linspace(X.min(), X.max(), 100)
    y_pred = model.predict(sm.add_constant(x_line))

    # --- scatter-ols.png ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X, y, color=ACCENT, alpha=0.6, s=28, edgecolors=BORDER, linewidths=0.5, label="Datos reales")
    ax.plot(x_line, y_pred, color=RED, linewidth=2, label="Regresión OLS")
    b0, b1 = model.params
    ax.set_title(f"Regresión OLS — ŷ = {b0:.2f} + {b1:.2f}·x   R² = {model.rsquared:.3f}", fontsize=11)
    ax.set_xlabel("Horas de estudio")
    ax.set_ylabel("Calificación")
    ax.legend()
    ax.grid(True)
    save(D / "scatter-ols.png")

    # --- residuos.png ---
    residuos = y - model.predict(X_const)
    y_fitted = model.predict(X_const)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Análisis de residuos del modelo OLS", fontsize=12)

    axes[0].scatter(y_fitted, residuos, color=ACCENT, alpha=0.6, s=24)
    axes[0].axhline(0, color=RED, linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Valores ajustados ŷ")
    axes[0].set_ylabel("Residuos")
    axes[0].set_title("Residuos vs Valores ajustados")
    axes[0].grid(True)

    axes[1].hist(residuos, bins=20, color=ACCENT, edgecolor=BORDER, linewidth=0.4)
    axes[1].set_xlabel("Residuo")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribución de residuos")
    axes[1].grid(True)

    plt.tight_layout()
    save(D / "residuos.png")

    print("→ regresion-statsmodels: OK")


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO: regresion-avanzada
# ──────────────────────────────────────────────────────────────────────────────
def gen_regresion_avanzada():
    D = Path("docs/assets/regresion-avanzada")
    D.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("data/csvs/examenes.csv")
    df_enc = pd.get_dummies(df, columns=["study_method"], dtype=int, drop_first=True)

    var_obj = "exam_score"
    vars_pred = ["study_hours",
                 "study_method_group study",
                 "study_method_mixed",
                 "study_method_online videos",
                 "study_method_self-study"]
    X = np.array(df_enc[vars_pred])
    y = np.array(df_enc[var_obj])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    model = sm.OLS(y_train, X_train).fit()

    # --- predicciones-vs-reales.png ---
    y_pred_test = model.predict(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Regresión avanzada: predicciones vs valores reales", fontsize=12)

    sample_idx = np.random.RandomState(0).choice(len(y_test), 80, replace=False)
    axes[0].scatter(y_test[sample_idx], y_pred_test[sample_idx],
                    color=ACCENT, alpha=0.5, s=22, edgecolors=BORDER, linewidths=0.3)
    lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
    axes[0].plot(lims, lims, color=RED, linestyle="--", linewidth=1.5, label="Predicción perfecta")
    axes[0].set_xlabel("Valores reales")
    axes[0].set_ylabel("Predicciones")
    axes[0].set_title("Predicciones vs reales (conjunto test)")
    axes[0].legend()
    axes[0].grid(True)

    residuos = y_test - y_pred_test
    axes[1].hist(residuos, bins=30, color=ACCENT, edgecolor=BORDER, linewidth=0.3)
    axes[1].axvline(0, color=RED, linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Residuo")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribución de residuos en test")
    axes[1].grid(True)

    plt.tight_layout()
    save(D / "predicciones-vs-reales.png")

    # --- rmse-folds.png ---
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    rmse_train_list, rmse_test_list = [], []

    for train_idx, test_idx in kf.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        res = sm.OLS(ytr, Xtr).fit()
        rmse_train_list.append(mean_squared_error(ytr, res.predict(Xtr)) ** 0.5)
        rmse_test_list.append(mean_squared_error(yte, res.predict(Xte)) ** 0.5)

    folds = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(folds, rmse_train_list, color=ACCENT, marker="o", markersize=5, label="RMSE Train")
    ax.plot(folds, rmse_test_list,  color=RED,   marker="s", markersize=5, label="RMSE Test")
    ax.set_title("RMSE por fold — K-Fold Cross Validation (k=10)", fontsize=11)
    ax.set_xlabel("Fold")
    ax.set_ylabel("RMSE")
    ax.set_xticks(folds)
    ax.legend()
    ax.grid(True)
    save(D / "rmse-folds.png")

    print("→ regresion-avanzada: OK")


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO: eda
# ──────────────────────────────────────────────────────────────────────────────
def gen_eda():
    D = Path("docs/assets/eda")
    D.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("data/csvs/examenes.csv")

    # --- scatter-horas-calificacion.png ---
    muestra = df.sample(500, random_state=42)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(muestra["study_hours"], muestra["exam_score"],
               color=ACCENT, alpha=0.45, s=20, edgecolors="none")
    ax.set_title("Horas de estudio vs calificación del examen", fontsize=12)
    ax.set_xlabel("Horas de estudio")
    ax.set_ylabel("Calificación")
    ax.grid(True)
    save(D / "scatter-horas-calificacion.png")

    # --- boxplot-genero.png ---
    generos = ["male", "female", "other"]
    datos_por_genero = [df[df["gender"] == g]["exam_score"].values for g in generos]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(datos_por_genero, patch_artist=True,
                    medianprops=dict(color=RED, linewidth=2),
                    whiskerprops=dict(color=TEXT2),
                    capprops=dict(color=TEXT2),
                    flierprops=dict(marker="o", markersize=3, color=TEXT3, alpha=0.5))
    for patch, color in zip(bp["boxes"], [ACCENT, "#5ecf8a", "#e0a84c"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticklabels(["Masculino", "Femenino", "Otro"])
    ax.set_title("Distribución de calificaciones por género", fontsize=12)
    ax.set_ylabel("Calificación del examen")
    ax.grid(True, axis="y")
    save(D / "boxplot-genero.png")

    # --- histograma-calificaciones.png ---
    fig, ax = plt.subplots(figsize=(8, 4))
    for g, color, label in [("male", ACCENT, "Masculino"),
                              ("female", RED, "Femenino"),
                              ("other", "#5ecf8a", "Otro")]:
        ax.hist(df[df["gender"] == g]["exam_score"], bins=35,
                alpha=0.5, color=color, label=label, edgecolor="none")
    ax.set_title("Distribución de calificaciones por género", fontsize=12)
    ax.set_xlabel("Calificación")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    ax.grid(True, axis="y")
    save(D / "histograma-calificaciones.png")

    # --- barras-genero.png ---
    conteo = df["gender"].value_counts().reindex(["male", "female", "other"])
    colores = [ACCENT, RED, "#5ecf8a"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["Masculino", "Femenino", "Otro"], conteo.values,
           color=colores, edgecolor=BORDER, linewidth=0.5)
    ax.set_title("Cantidad de estudiantes por género", fontsize=12)
    ax.set_ylabel("Cantidad")
    ax.grid(True, axis="y")
    save(D / "barras-genero.png")

    # --- heatmap-correlacion.png ---
    categoricas = ["gender", "course", "internet_access", "sleep_quality",
                   "study_method", "facility_rating", "exam_difficulty"]
    df_num = df.drop(columns=categoricas + ["student_id"])
    matriz = df_num.corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(matriz, annot=True, fmt=".2f", cmap="Blues",
                ax=ax, linewidths=0.4, linecolor=BORDER,
                annot_kws={"size": 9},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Matriz de correlación de variables numéricas", fontsize=11)
    plt.tight_layout()
    save(D / "heatmap-correlacion.png")

    print("→ eda: OK")


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO: pruebas-hipotesis
# ──────────────────────────────────────────────────────────────────────────────
def gen_pruebas_hipotesis():
    D = Path("docs/assets/pruebas-hipotesis")
    D.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("data/csvs/examenes.csv")

    # --- boxplot-metodos.png ---
    metodos = df["study_method"].unique()
    datos_metodos = [df[df["study_method"] == m]["exam_score"].values for m in metodos]
    labels_metodos = [m.replace(" ", "\n") for m in metodos]

    colores_metodos = [ACCENT, RED, "#5ecf8a", "#e0a84c", "#c07de0"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(datos_metodos, patch_artist=True,
                    medianprops=dict(color=TEXT, linewidth=2),
                    whiskerprops=dict(color=TEXT2),
                    capprops=dict(color=TEXT2),
                    flierprops=dict(marker="o", markersize=2, color=TEXT3, alpha=0.3))
    for patch, color in zip(bp["boxes"], colores_metodos):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticklabels(labels_metodos, fontsize=9)
    ax.set_title("Distribución de calificaciones por método de estudio\n(Prueba ANOVA)", fontsize=11)
    ax.set_ylabel("Calificación del examen")
    ax.grid(True, axis="y")
    save(D / "boxplot-metodos.png")

    # --- tabla-contingencias.png ---
    tabla = pd.crosstab(df["gender"], df["study_method"])
    tabla.index = ["Femenino", "Masculino", "Otro"]

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(tabla, annot=True, fmt="d", cmap="Blues",
                ax=ax, linewidths=0.4, linecolor=BORDER,
                annot_kws={"size": 10},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Tabla de contingencias: género × método de estudio\n(Prueba Chi cuadrada)", fontsize=11)
    ax.set_xlabel("Método de estudio")
    ax.set_ylabel("Género")
    plt.tight_layout()
    save(D / "tabla-contingencias.png")

    print("→ pruebas-hipotesis: OK")


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO: problemas-regresion
# ──────────────────────────────────────────────────────────────────────────────
def gen_problemas_regresion():
    D = Path("docs/assets/problemas-regresion")
    D.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    n = 200

    # --- residuos-nolinealidad.png ---
    # Datos sintéticos con relación cuadrática
    x_nl = np.linspace(0, 10, n)
    y_nl = 2 * x_nl + 0.5 * x_nl**2 + rng.normal(0, 4, n)

    X_lin = sm.add_constant(x_nl)
    m_lin = sm.OLS(y_nl, X_lin).fit()

    X_poly = sm.add_constant(np.column_stack([x_nl, x_nl**2]))
    m_poly = sm.OLS(y_nl, X_poly).fit()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("No-linealidad: residuos vs valores ajustados", fontsize=12)

    axes[0].scatter(m_lin.fittedvalues, m_lin.resid,
                    color=ACCENT, alpha=0.5, s=18, edgecolors="none")
    axes[0].axhline(0, color=RED, linestyle="--", linewidth=1.5)
    axes[0].set_title("Modelo lineal — patrón en arco")
    axes[0].set_xlabel("Valores ajustados ŷ")
    axes[0].set_ylabel("Residuos")
    axes[0].grid(True)

    axes[1].scatter(m_poly.fittedvalues, m_poly.resid,
                    color=ACCENT, alpha=0.5, s=18, edgecolors="none")
    axes[1].axhline(0, color=RED, linestyle="--", linewidth=1.5)
    axes[1].set_title("Modelo cuadrático — scatter sin patrón")
    axes[1].set_xlabel("Valores ajustados ŷ")
    axes[1].set_ylabel("Residuos")
    axes[1].grid(True)

    plt.tight_layout()
    save(D / "residuos-nolinealidad.png")

    # --- correlacion-errores.png ---
    # Panel izquierdo: residuos autocorrelacionados (caminata aleatoria)
    # Panel derecho: residuos independientes (ruido blanco)
    resid_corr = np.cumsum(rng.normal(0, 1, n)) * 0.3
    resid_indep = rng.normal(0, 1, n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Correlación de errores: residuos en orden de observación", fontsize=12)

    dw_corr = durbin_watson(resid_corr)
    axes[0].plot(np.arange(n), resid_corr, color=ACCENT, linewidth=0.8, alpha=0.8)
    axes[0].axhline(0, color=RED, linestyle="--", linewidth=1.5)
    axes[0].set_title(f"Residuos autocorrelacionados  (DW = {dw_corr:.2f})")
    axes[0].set_xlabel("Índice de observación")
    axes[0].set_ylabel("Residuo")
    axes[0].grid(True)

    dw_indep = durbin_watson(resid_indep)
    axes[1].plot(np.arange(n), resid_indep, color=ACCENT, linewidth=0.8, alpha=0.8)
    axes[1].axhline(0, color=RED, linestyle="--", linewidth=1.5)
    axes[1].set_title(f"Residuos independientes  (DW = {dw_indep:.2f})")
    axes[1].set_xlabel("Índice de observación")
    axes[1].set_ylabel("Residuo")
    axes[1].grid(True)

    plt.tight_layout()
    save(D / "correlacion-errores.png")

    # --- heterocedasticidad.png ---
    # Datos sintéticos: varianza crece con x
    x_h = np.linspace(1, 10, n)
    y_h = 3 * x_h + rng.normal(0, 1, n) * x_h   # heterocedasticidad: σ ∝ x

    X_h = sm.add_constant(x_h)
    m_h = sm.OLS(y_h, X_h).fit()

    y_log = np.log(y_h - y_h.min() + 1)
    m_hlog = sm.OLS(y_log, X_h).fit()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Heterocedasticidad: varianza no constante de residuos", fontsize=12)

    axes[0].scatter(m_h.fittedvalues, m_h.resid,
                    color=ACCENT, alpha=0.5, s=18, edgecolors="none")
    axes[0].axhline(0, color=RED, linestyle="--", linewidth=1.5)
    axes[0].set_title("Escala original — patrón de embudo")
    axes[0].set_xlabel("Valores ajustados ŷ")
    axes[0].set_ylabel("Residuos")
    axes[0].grid(True)

    axes[1].scatter(m_hlog.fittedvalues, m_hlog.resid,
                    color=ACCENT, alpha=0.5, s=18, edgecolors="none")
    axes[1].axhline(0, color=RED, linestyle="--", linewidth=1.5)
    axes[1].set_title("Tras transformación log — varianza estable")
    axes[1].set_xlabel("Valores ajustados ŷ")
    axes[1].set_ylabel("Residuos")
    axes[1].grid(True)

    plt.tight_layout()
    save(D / "heterocedasticidad.png")

    # --- outliers.png ---
    df = pd.read_csv("data/csvs/examenes.csv").sample(n=100, random_state=42)
    X_o = sm.add_constant(np.array(df["study_hours"]))
    y_o = np.array(df["exam_score"])

    # Modelo sin outliers
    m_clean = sm.OLS(y_o, X_o).fit()

    # Inyectar dos outliers
    y_out = y_o.copy()
    y_out[0]  = y_o.mean() + 4 * y_o.std()
    y_out[10] = y_o.mean() - 4 * y_o.std()
    m_out = sm.OLS(y_out, X_o).fit()

    # Residuos estudentizados del modelo con outliers
    infl = OLSInfluence(m_out)
    r_stud = infl.resid_studentized_external

    x_vals = df["study_hours"].values
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Outliers: impacto en la recta de regresión", fontsize=12)

    axes[0].scatter(x_vals, y_o, color=TEXT2, alpha=0.5, s=20, edgecolors="none", label="Datos normales")
    axes[0].scatter(x_vals[[0, 10]], y_out[[0, 10]],
                    color=RED, s=80, marker="*", zorder=5, label="Outliers")
    axes[0].plot(x_line, m_clean.params[0] + m_clean.params[1] * x_line,
                 color=ACCENT, linewidth=2, label="Sin outliers")
    axes[0].plot(x_line, m_out.params[0] + m_out.params[1] * x_line,
                 color=RED, linewidth=2, linestyle="--", label="Con outliers")
    axes[0].set_title("Efecto de dos outliers en la recta")
    axes[0].set_xlabel("Horas de estudio")
    axes[0].set_ylabel("Calificación")
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    axes[1].scatter(np.arange(len(r_stud)), r_stud,
                    color=ACCENT, alpha=0.6, s=20, edgecolors="none")
    axes[1].axhline( 3, color=RED, linestyle="--", linewidth=1.5, label="|r| = 3")
    axes[1].axhline(-3, color=RED, linestyle="--", linewidth=1.5)
    outlier_idx = np.where(np.abs(r_stud) > 3)[0]
    axes[1].scatter(outlier_idx, r_stud[outlier_idx],
                    color=RED, s=60, zorder=5)
    axes[1].set_title("Residuos estudentizados")
    axes[1].set_xlabel("Índice de observación")
    axes[1].set_ylabel("Residuo estudentizado")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    plt.tight_layout()
    save(D / "outliers.png")

    # --- leverage.png ---
    X_lev = sm.add_constant(np.array(df["study_hours"]))
    y_lev = np.array(df["exam_score"])
    m_lev = sm.OLS(y_lev, X_lev).fit()
    infl_lev = OLSInfluence(m_lev)
    h_vals = infl_lev.hat_matrix_diag

    p = 1   # un predictor
    n_lev = len(y_lev)
    umbral = 2 * (p + 1) / n_lev
    alto = h_vals > umbral

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Alto leverage: puntos con posición inusual en X", fontsize=12)

    axes[0].scatter(x_vals[~alto], y_lev[~alto],
                    color=ACCENT, alpha=0.6, s=22, edgecolors="none", label="Normal")
    axes[0].scatter(x_vals[alto], y_lev[alto],
                    color=RED, s=50, zorder=5, label=f"Alto leverage (h > {umbral:.2f})")
    axes[0].set_title("Puntos de alto leverage resaltados")
    axes[0].set_xlabel("Horas de estudio")
    axes[0].set_ylabel("Calificación")
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    axes[1].scatter(np.arange(n_lev), h_vals,
                    color=ACCENT, alpha=0.6, s=18, edgecolors="none")
    axes[1].axhline(umbral, color=RED, linestyle="--", linewidth=1.5,
                    label=f"Umbral 2(p+1)/n = {umbral:.2f}")
    axes[1].scatter(np.where(alto)[0], h_vals[alto],
                    color=RED, s=50, zorder=5)
    axes[1].set_title("Estadístico h_ii por observación")
    axes[1].set_xlabel("Índice de observación")
    axes[1].set_ylabel("h_ii (leverage)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    plt.tight_layout()
    save(D / "leverage.png")

    # --- colinealidad.png ---
    df_full = pd.read_csv("data/csvs/examenes.csv").sample(n=500, random_state=7)
    x_sh = df_full["study_hours"].values
    x_ca = df_full["class_attendance"].values
    x_sl = df_full["sleep_hours"].values
    y_c  = df_full["exam_score"].values

    r = np.corrcoef(x_sh, x_ca)[0, 1]

    vars_multi = ["study_hours", "class_attendance", "sleep_hours"]
    X_vif = sm.add_constant(np.column_stack([x_sh, x_ca, x_sl]))
    vif_vals = [
        variance_inflation_factor(X_vif, i + 1)
        for i in range(len(vars_multi))
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Colinealidad: predictores correlacionados", fontsize=12)

    axes[0].scatter(x_sh, x_ca, color=ACCENT, alpha=0.3, s=15, edgecolors="none")
    axes[0].set_title(f"Horas de estudio vs Asistencia  (r = {r:.2f})")
    axes[0].set_xlabel("Horas de estudio")
    axes[0].set_ylabel("Asistencia a clase (%)")
    axes[0].grid(True)
    axes[0].text(0.05, 0.92, f"r = {r:.2f}", transform=axes[0].transAxes,
                 fontsize=12, color=RED, fontweight="bold")

    labels_short = ["study_hours", "class_attend.", "sleep_hours"]
    colors_vif = [RED if v > 10 else ACCENT for v in vif_vals]
    bars = axes[1].barh(labels_short, vif_vals, color=colors_vif,
                        edgecolor=BORDER, linewidth=0.5)
    axes[1].axvline(10, color=RED, linestyle="--", linewidth=1.5, label="VIF = 10")
    axes[1].axvline(5,  color=TEXT2, linestyle=":", linewidth=1.2, label="VIF = 5")
    for bar, v in zip(bars, vif_vals):
        axes[1].text(v + 0.1, bar.get_y() + bar.get_height() / 2,
                     f"{v:.1f}", va="center", fontsize=10)
    axes[1].set_title("Factor de Inflación de Varianza (VIF)")
    axes[1].set_xlabel("VIF")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="x")

    plt.tight_layout()
    save(D / "colinealidad.png")

    print("→ problemas-regresion: OK")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generando imágenes para todos los módulos...\n")
    gen_visualizacion()
    gen_gradiente()
    gen_regresion_simple()
    gen_regresion_statsmodels()
    gen_regresion_avanzada()
    gen_eda()
    gen_pruebas_hipotesis()
    gen_problemas_regresion()
    print("\nListo. Todas las imágenes generadas.")
