"""
A3. KNN y kernel regression
Tema A03_knn_y_kernel_regression — Introducción al Machine Learning (IIND3118)

CÓMO USARLO
  1. Cambia el bloque marcado con ✏️ abajo por tu dataset.
  2. Corre el archivo:  python A03_knn_y_kernel_regression.py
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
print("A3. KNN y kernel regression")
print("="*74)
print(f"datos: {X.shape[0]} obs. x {X.shape[1]} predictores")
if y is not None:
    print(f"y ('{TARGET}'): " + ("CLASIFICACIÓN" if ES_CLASIFICACION else "REGRESIÓN"))
print()

# ════════════════════════════════════════════════════════════════════════
#  CÓDIGO DEL MÉTODO
# ════════════════════════════════════════════════════════════════════════

# ═══ A3. KNN — calibrar k y evaluar ═══
# ⚠️ SIEMPRE con Xs (estandarizado), porque usa distancias.

Modelo = KNeighborsClassifier if ES_CLASIFICACION else KNeighborsRegressor

# --- 1. Calibrar k con la curva de test ---------------------------------
ks = range(1, 41)
errores = []
for k in ks:
    m = Modelo(n_neighbors=k).fit(Xs_train, y_train)
    p = m.predict(Xs_test)
    errores.append(1 - accuracy_score(y_test, p) if ES_CLASIFICACION
                   else mean_squared_error(y_test, p))

k_opt = list(ks)[int(np.argmin(errores))]

plt.plot(ks, errores, "o-", ms=3)
plt.axvline(k_opt, color="red", ls="--", label=f"k óptimo = {k_opt}")
plt.xlabel("k (número de vecinos)")
plt.ylabel("error de clasificación" if ES_CLASIFICACION else "MSE test")
plt.title("Calibración de k"); plt.legend(); plt.show()

# --- 2. Modelo final con el k calibrado ---------------------------------
knn = Modelo(
    n_neighbors=k_opt,      # ⬅ el k que calibraste
    weights="uniform",      # 'uniform' o 'distance'
).fit(Xs_train, y_train)

pred_knn = knn.predict(Xs_test)
mse_knn = mean_squared_error(y_test, pred_knn)
print(f"k óptimo = {k_opt}   |   MSE test = {mse_knn:.3f}")


# ═══ Kernel regression (ventana h) — implementación de clase ═══
def kernelreg(xstar, X_, Y_, h):
    """Promedio de los Y cuyos X caen dentro de [xstar-h, xstar+h].
       h GRANDE = modelo rígido (mucho sesgo). h PEQUEÑO = flexible."""
    pred = np.zeros(len(xstar))
    for i in range(len(xstar)):
        dentro = np.abs(xstar[i] - X_) <= h
        pred[i] = np.sum(np.where(dentro, Y_, 0)) / max(dentro.sum(), 1)
    return pred

# Demostración en 1 variable (la de mayor correlación con y)
j = int(np.argmax([abs(np.corrcoef(X[:, c], y)[0, 1]) for c in range(X.shape[1])]))
x1 = Xs[:, j]
orden = np.argsort(x1)
malla = np.linspace(x1.min(), x1.max(), 200)

plt.plot(x1, y, "o", ms=3, color="gray", alpha=.5, label="datos")
for h, col in [(0.05, "green"), (0.3, "blue"), (1.5, "red")]:
    plt.plot(malla, kernelreg(malla, x1, y, h), col, lw=2, label=f"h = {h}")
plt.xlabel(f"{nombres_X[j]} (estandarizada)"); plt.ylabel(TARGET)
plt.title("Kernel regression: efecto de h"); plt.legend(); plt.show()

print("h chico (verde) = persigue el ruido -> VARIANZA")
print("h grande (rojo) = casi una recta plana -> SESGO")

print("\nListo. Revisa la carpeta figuras/")
