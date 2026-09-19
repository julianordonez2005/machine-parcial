"""
A6. Validación cruzada (k-fold)
Tema A06_validacion_cruzada — Introducción al Machine Learning (IIND3118)

CÓMO USARLO
  1. Cambia el bloque marcado con ✏️ abajo por tu dataset.
  2. Corre el archivo:  python A06_validacion_cruzada.py
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
print("A6. Validación cruzada (k-fold)")
print("="*74)
print(f"datos: {X.shape[0]} obs. x {X.shape[1]} predictores")
if y is not None:
    print(f"y ('{TARGET}'): " + ("CLASIFICACIÓN" if ES_CLASIFICACION else "REGRESIÓN"))
print()

# ════════════════════════════════════════════════════════════════════════
#  CÓDIGO DEL MÉTODO
# ════════════════════════════════════════════════════════════════════════

# ═══ A6. Validación cruzada ═══

# ── Forma 1: la más rápida, cross_val_score ──────────────────────────────
modelo = KNeighborsRegressor(n_neighbors=5)

scores = cross_val_score(
    modelo, Xs, y,
    cv=10,                                  # ⬅ número de folds (5 o 10)
    scoring="neg_mean_squared_error",       # ⬅ OJO: negativo, hay que voltearlo
)
mse_cv = -scores                            # volteamos el signo

print(f"MSE de cada uno de los 10 folds:\n{np.round(mse_cv, 2)}")
print(f"\nMSE promedio (CV): {mse_cv.mean():.3f}  ±  {mse_cv.std():.3f}")
print("La desviación indica qué tan estable es el resultado entre folds.")

# ── Forma 2: el bucle manual, por si piden mostrar el procedimiento ─────
kf = KFold(n_splits=10, shuffle=True, random_state=0)
errores = []
for idx_train, idx_test in kf.split(Xs):
    m = KNeighborsRegressor(n_neighbors=5).fit(Xs[idx_train], y[idx_train])
    errores.append(mean_squared_error(y[idx_test], m.predict(Xs[idx_test])))
print(f"\nMSE promedio (bucle manual): {np.mean(errores):.3f}")

# ── Forma 3: calibrar un parámetro CON validación cruzada ───────────────
ks = range(1, 41)
mse_por_k = [-cross_val_score(KNeighborsRegressor(n_neighbors=k), Xs, y,
                              cv=10, scoring="neg_mean_squared_error").mean()
             for k in ks]
k_cv = list(ks)[int(np.argmin(mse_por_k))]

plt.plot(ks, mse_por_k, "o-", ms=3)
plt.axvline(k_cv, color="red", ls="--", label=f"k óptimo (CV) = {k_cv}")
plt.xlabel("k"); plt.ylabel("MSE por validación cruzada")
plt.title("Calibración de k con 10-fold CV"); plt.legend(); plt.show()
print(f"k óptimo por CV = {k_cv}   (MSE = {min(mse_por_k):.3f})")

# ── Forma 4: GridSearchCV, lo más práctico ─────────────────────────────
grid = GridSearchCV(
    KNeighborsRegressor(),
    param_grid={"n_neighbors": range(1, 41),         # ⬅ valores a probar
                "weights": ["uniform", "distance"]},  # ⬅ puede probar varios a la vez
    cv=10,
    scoring="neg_mean_squared_error",
).fit(Xs, y)

print(f"\nGridSearchCV -> mejores parámetros: {grid.best_params_}")
print(f"MSE del mejor modelo: {-grid.best_score_:.3f}")

print("\nListo. Revisa la carpeta figuras/")
