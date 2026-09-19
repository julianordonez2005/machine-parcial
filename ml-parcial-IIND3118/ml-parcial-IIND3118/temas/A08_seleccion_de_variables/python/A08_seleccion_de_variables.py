"""
A8. Selección de variables
Tema A08_seleccion_de_variables — Introducción al Machine Learning (IIND3118)

CÓMO USARLO
  1. Cambia el bloque marcado con ✏️ abajo por tu dataset.
  2. Corre el archivo:  python A08_seleccion_de_variables.py
  3. Las gráficas se guardan en la carpeta figuras/ de este tema.

La explicación del método está en el README.md de esta misma carpeta.
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no abre ventanas: funciona en cualquier entorno
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import scale
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
                                  RidgeCV, LassoCV, ElasticNetCV, LogisticRegression)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score)
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn import manifold

# ── Guardado automático de figuras ──────────────────────────────────────
AQUI = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(AQUI, "..", "figuras")
os.makedirs(FIGS, exist_ok=True)
_n = [0]
def _save():
    _n[0] += 1
    ruta = os.path.join(FIGS, f"{_n[0]:02d}_figura.png")
    plt.savefig(ruta, dpi=110, bbox_inches="tight"); plt.close()
    print(f"   [figura guardada: figuras/{os.path.basename(ruta)}]")
plt.show = _save                          # cada plt.show() guarda en vez de abrir

RNG = np.random.default_rng(0)

# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  ✏️  CAMBIA SOLO ESTE BLOQUE POR TU DATASET                             ║
# ╚═════════════════════════════════════════════════════════════════════════╝
DATOS  = os.path.join(AQUI, "..", "..", "..", "data", "ejemplo_regresion.csv")
TARGET = "target"          # nombre EXACTO de la columna respuesta. None si no hay.
# ╚═════════════════════════════════════════════════════════════════════════╝

df = pd.read_csv(DATOS).dropna()

if TARGET is not None:
    y_serie = df[TARGET]; X_df = df.drop(columns=[TARGET])
else:
    y_serie = None; X_df = df.copy()

X_df = pd.get_dummies(X_df, drop_first=True).astype(float)
nombres_X = list(X_df.columns)
X  = X_df.values
Xs = scale(X)

if y_serie is not None:
    y = pd.factorize(y_serie)[0] if y_serie.dtype == object else y_serie.values
    ES_CLASIFICACION = (len(np.unique(y)) <= 10) and np.all(y == np.round(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=0)
    Xs_train, Xs_test, _, _ = train_test_split(Xs, y, test_size=.30, random_state=0)
else:
    y = None; ES_CLASIFICACION = False

def ModeloLineal(**kw):
    return (LogisticRegression(max_iter=5000, **kw) if ES_CLASIFICACION
            else LinearRegression(**kw))

col = y if y is not None else "steelblue"
Z   = PCA().fit_transform(Xs)        # proyección PCA, se usa para graficar
Z2  = Z[:, :2]

print("="*74)
print("A8. Selección de variables")
print("="*74)
print(f"datos: {X.shape[0]} obs. x {X.shape[1]} predictores")
if y is not None:
    print(f"y ('{TARGET}'): " + ("CLASIFICACIÓN" if ES_CLASIFICACION else "REGRESIÓN"))
print()

# ════════════════════════════════════════════════════════════════════════
#  CÓDIGO DEL MÉTODO
# ════════════════════════════════════════════════════════════════════════

# ═══ A8. Selección de variables ═══
from sklearn.feature_selection import SequentialFeatureSelector

p = X_train.shape[1]

# ── MÉTODO FORWARD: probar cada número de variables y elegir por R² ajustado ──
r2adj = []
n_tr = len(y_train)
for k in range(1, p + 1):
    if k < p:
        sfs = SequentialFeatureSelector(
            LinearRegression(),
            n_features_to_select=k,     # ⬅ cuántas variables en esta iteración
            direction="forward",        # ⬅ 'forward' o 'backward'
            cv=5,
        ).fit(X_train, y_train)
        Xk = X_train[:, sfs.get_support()]
    else:
        Xk = X_train
    r2 = LinearRegression().fit(Xk, y_train).score(Xk, y_train)
    r2adj.append(1 - (1 - r2) * (n_tr - 1) / (n_tr - k - 1))   # R² ajustado

k_opt = int(np.argmax(r2adj)) + 1

plt.plot(range(1, p + 1), r2adj, "o-")
plt.axvline(k_opt, color="red", ls="--", label=f"óptimo = {k_opt} variables")
plt.xlabel("número de variables"); plt.ylabel("R² ajustado")
plt.title("Forward: R² ajustado vs. nº de variables"); plt.legend(); plt.show()

# Modelo final con las variables seleccionadas
if k_opt < p:
    sfs_final = SequentialFeatureSelector(LinearRegression(),
                    n_features_to_select=k_opt, direction="forward", cv=5).fit(X_train, y_train)
    mask = sfs_final.get_support()
else:
    mask = np.ones(p, dtype=bool)

vars_fwd = [nombres_X[i] for i in range(p) if mask[i]]
modelo_fwd = LinearRegression().fit(X_train[:, mask], y_train)
mse_fwd = mean_squared_error(y_test, modelo_fwd.predict(X_test[:, mask]))

print(f"Variables seleccionadas ({k_opt}): {vars_fwd}")
print(f"MSE test (forward): {mse_fwd:.3f}")

# ── MÉTODO EXHAUSTIVO ───────────────────────────────────────────────────
# ⚠️ Solo si p es pequeño (<= 15). Con más, tarda demasiado.
if p <= 15:
    from mlxtend.feature_selection import ExhaustiveFeatureSelector
    efs = ExhaustiveFeatureSelector(
        LinearRegression(),
        min_features=1, max_features=p,        # ⬅ rango de tamaños a explorar
        scoring="neg_mean_squared_error",
        cv=5, print_progress=False,
    ).fit(X_train, y_train)

    idx_exh = list(efs.best_idx_)
    vars_exh = [nombres_X[i] for i in idx_exh]
    modelo_exh = LinearRegression().fit(X_train[:, idx_exh], y_train)
    mse_exh = mean_squared_error(y_test, modelo_exh.predict(X_test[:, idx_exh]))
    print(f"\nVariables seleccionadas ({len(idx_exh)}): {vars_exh}")
    print(f"MSE test (exhaustivo): {mse_exh:.3f}")

    plt.bar(["Forward", "Exhaustivo"], [mse_fwd, mse_exh], color=["steelblue","indianred"])
    plt.ylabel("MSE en test"); plt.title("Comparación de métodos de selección")
    for i, v in enumerate([mse_fwd, mse_exh]):
        plt.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    plt.show()
else:
    print(f"\np = {p} es demasiado grande para el método exhaustivo (2^{p}-1 modelos).")

print("\nListo. Revisa la carpeta figuras/")
