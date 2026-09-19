"""
A5. AIC, BIC, Cp de Mallows y R² ajustado
Tema A05_aic_bic_cp_r2ajustado — Introducción al Machine Learning (IIND3118)

CÓMO USARLO
  1. Cambia el bloque marcado con ✏️ abajo por tu dataset.
  2. Corre el archivo:  python A05_aic_bic_cp_r2ajustado.py
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
print("A5. AIC, BIC, Cp de Mallows y R² ajustado")
print("="*74)
print(f"datos: {X.shape[0]} obs. x {X.shape[1]} predictores")
if y is not None:
    print(f"y ('{TARGET}'): " + ("CLASIFICACIÓN" if ES_CLASIFICACION else "REGRESIÓN"))
print()

# ════════════════════════════════════════════════════════════════════════
#  CÓDIGO DEL MÉTODO
# ════════════════════════════════════════════════════════════════════════

# ═══ A5. AIC, BIC, Cp y R² ajustado ═══
import statsmodels.api as sm

def criterios(X_, y_, nombre=""):
    """Calcula las 4 métricas de un modelo lineal. X_ sin columna de unos."""
    n, p = X_.shape
    Xc = sm.add_constant(X_)                   # statsmodels necesita el intercepto
    mod = sm.OLS(y_, Xc).fit()
    sse = np.sum(mod.resid ** 2)
    sigma2 = sse / (n - p - 1)
    cp = (sse + 2 * p * sigma2) / n
    print(f"{nombre:28s} p={p:2d}  R²adj={mod.rsquared_adj:7.4f}  "
          f"AIC={mod.aic:9.2f}  BIC={mod.bic:9.2f}  Cp={cp:9.2f}")
    return dict(p=p, r2adj=mod.rsquared_adj, aic=mod.aic, bic=mod.bic, cp=cp)

# Comparar modelos con 1, 2, ..., p variables (las primeras columnas)
res = [criterios(X_train[:, :k], y_train, f"primeras {k} variables")
       for k in range(1, X_train.shape[1] + 1)]

r2a = [r["r2adj"] for r in res]; aic = [r["aic"] for r in res]
bic = [r["bic"] for r in res];   cp  = [r["cp"] for r in res]
ks = range(1, len(res) + 1)

fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
for a, v, t, mejor in [(ax[0], r2a, "R² ajustado (MAX)", np.argmax(r2a)),
                       (ax[1], aic, "AIC (MIN)",         np.argmin(aic)),
                       (ax[2], bic, "BIC (MIN)",         np.argmin(bic)),
                       (ax[3], cp,  "Cp Mallows (MIN)",  np.argmin(cp))]:
    a.plot(ks, v, "o-"); a.axvline(list(ks)[mejor], color="red", ls="--")
    a.set_xlabel("nº de variables"); a.set_title(t)
plt.tight_layout(); plt.show()

print(f"\nMejor según R²adj: {np.argmax(r2a)+1} variables")
print(f"Mejor según AIC  : {np.argmin(aic)+1} variables")
print(f"Mejor según BIC  : {np.argmin(bic)+1} variables  <- suele elegir MENOS")
print(f"Mejor según Cp   : {np.argmin(cp)+1} variables")
print("\nBIC castiga más la complejidad (p·log n) que AIC (2p).")

print("\nListo. Revisa la carpeta figuras/")
