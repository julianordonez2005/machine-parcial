"""
A2. Flexibilidad y el dilema sesgo–varianza
Tema A02_flexibilidad_sesgo_varianza — Introducción al Machine Learning (IIND3118)

CÓMO USARLO
  1. Cambia el bloque marcado con ✏️ abajo por tu dataset.
  2. Corre el archivo:  python A02_flexibilidad_sesgo_varianza.py
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
print("A2. Flexibilidad y el dilema sesgo–varianza")
print("="*74)
print(f"datos: {X.shape[0]} obs. x {X.shape[1]} predictores")
if y is not None:
    print(f"y ('{TARGET}'): " + ("CLASIFICACIÓN" if ES_CLASIFICACION else "REGRESIÓN"))
print()

# ════════════════════════════════════════════════════════════════════════
#  CÓDIGO DEL MÉTODO
# ════════════════════════════════════════════════════════════════════════

# ═══ A2. La curva en U del MSE de test ═══
# Se ilustra con KNN variando k. El mismo patrón aplica a cualquier modelo.

# KNN = el regresor o el clasificador, según el problema
KNN = KNeighborsClassifier if ES_CLASIFICACION else KNeighborsRegressor

ks = range(1, 41)
mse_train_list, mse_test_list = [], []

for k in ks:
    m = KNN(n_neighbors=k).fit(Xs_train, y_train)
    mse_train_list.append(mean_squared_error(y_train, m.predict(Xs_train)))
    mse_test_list.append(mean_squared_error(y_test,  m.predict(Xs_test)))

k_opt = list(ks)[int(np.argmin(mse_test_list))]

# ⚠️ El eje x va de k grande a k pequeño para que "flexibilidad" crezca hacia
#    la derecha, como en las diapositivas.
plt.plot(ks, mse_train_list, "r-", lw=2, label="MSE TRAIN (siempre baja)")
plt.plot(ks, mse_test_list,  "b-", lw=2, label="MSE TEST (forma de U)")
plt.axvline(k_opt, color="k", ls="--", label=f"k óptimo = {k_opt}")
plt.gca().invert_xaxis()                      # ⬅ izquierda = rígido, derecha = flexible
plt.xlabel("k  (⬅ menos flexible    |    más flexible ➡)")
plt.ylabel("MSE"); plt.legend(); plt.title("Sesgo–varianza: la curva en U")
plt.show()

print(f"k óptimo             : {k_opt}")
print(f"MSE test en el óptimo: {min(mse_test_list):.3f}")
print("\nIzquierda (k grande) = mucho SESGO, el modelo es muy rígido.")
print("Derecha  (k pequeño) = mucha VARIANZA, el modelo persigue el ruido.")

print("\nListo. Revisa la carpeta figuras/")
