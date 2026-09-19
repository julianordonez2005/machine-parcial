"""
A10. Comparación final de modelos
Tema A10_comparacion_de_modelos — Introducción al Machine Learning (IIND3118)

CÓMO USARLO
  1. Cambia el bloque marcado con ✏️ abajo por tu dataset.
  2. Corre el archivo:  python A10_comparacion_de_modelos.py
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
print("A10. Comparación final de modelos")
print("="*74)
print(f"datos: {X.shape[0]} obs. x {X.shape[1]} predictores")
if y is not None:
    print(f"y ('{TARGET}'): " + ("CLASIFICACIÓN" if ES_CLASIFICACION else "REGRESIÓN"))
print()

# ════════════════════════════════════════════════════════════════════════
#  CÓDIGO DEL MÉTODO
# ════════════════════════════════════════════════════════════════════════

# ═══ A10. Comparar todos los modelos de un golpe ═══

# Este script recalcula TODOS los modelos desde cero, así que se puede correr
# por sí solo sin haber ejecutado los temas anteriores.

alphas = np.logspace(-4, 2, 100)

# Ridge, Lasso y Elastic Net (⚠️ con Xs, estandarizado)
mse_ridge = mean_squared_error(y_test,
    RidgeCV(alphas=alphas, cv=10).fit(Xs_train, y_train).predict(Xs_test))
mse_lasso = mean_squared_error(y_test,
    LassoCV(alphas=alphas, cv=10, max_iter=10000, random_state=0)
        .fit(Xs_train, y_train).predict(Xs_test))
mse_elastic = mean_squared_error(y_test,
    ElasticNetCV(alphas=alphas, l1_ratio=[.1,.5,.9,1], cv=10,
                 max_iter=10000, random_state=0).fit(Xs_train, y_train).predict(Xs_test))

# Selección forward
from sklearn.feature_selection import SequentialFeatureSelector
p_var = X_train.shape[1]
n_tr  = len(y_train)
r2adj = []
for k in range(1, p_var + 1):
    if k < p_var:
        sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k,
                                        direction="forward", cv=5).fit(X_train, y_train)
        Xk = X_train[:, sfs.get_support()]
    else:
        Xk = X_train
    r2 = LinearRegression().fit(Xk, y_train).score(Xk, y_train)
    r2adj.append(1 - (1-r2)*(n_tr-1)/(n_tr-k-1))
k_opt = int(np.argmax(r2adj)) + 1
if k_opt < p_var:
    mask = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k_opt,
                                     direction="forward", cv=5).fit(X_train, y_train).get_support()
else:
    mask = np.ones(p_var, dtype=bool)
mse_fwd = mean_squared_error(y_test,
    LinearRegression().fit(X_train[:, mask], y_train).predict(X_test[:, mask]))

resultados = {}

# Lineal
resultados["Lineal (OLS)"] = mean_squared_error(y_test,
    LinearRegression().fit(X_train, y_train).predict(X_test))

# KNN calibrado por CV
k_best = GridSearchCV(KNeighborsRegressor(), {"n_neighbors": range(1, 41)},
                      cv=10, scoring="neg_mean_squared_error").fit(Xs_train, y_train)
resultados[f"KNN (k={k_best.best_params_['n_neighbors']})"] = mean_squared_error(
    y_test, k_best.predict(Xs_test))

# Penalizados
resultados["Ridge"]       = mse_ridge
resultados["Lasso"]       = mse_lasso
resultados["Elastic Net"] = mse_elastic

# Forward
resultados[f"Forward ({k_opt} vars)"] = mse_fwd

# ── Tabla ordenada ──────────────────────────────────────────────────────
tabla = pd.DataFrame({"modelo": list(resultados.keys()),
                      "MSE test": list(resultados.values())})
tabla["RMSE"] = np.sqrt(tabla["MSE test"])
tabla = tabla.sort_values("MSE test").reset_index(drop=True)
print(tabla.to_string(index=False))
print(f"\n🏆 Mejor modelo: {tabla.iloc[0]['modelo']}  (MSE = {tabla.iloc[0]['MSE test']:.3f})")

plt.barh(tabla["modelo"][::-1], tabla["MSE test"][::-1], color="steelblue")
plt.xlabel("MSE en test (menor = mejor)"); plt.title("Comparación de todos los modelos")
plt.tight_layout(); plt.show()

print("\nListo. Revisa la carpeta figuras/")
