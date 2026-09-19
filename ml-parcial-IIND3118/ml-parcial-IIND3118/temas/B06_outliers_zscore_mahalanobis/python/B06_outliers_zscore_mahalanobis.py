"""
B6. Outliers: Z-Score y Mahalanobis
Tema B06_outliers_zscore_mahalanobis — Introducción al Machine Learning (IIND3118)

CÓMO USARLO
  1. Cambia el bloque marcado con ✏️ abajo por tu dataset.
  2. Corre el archivo:  python B06_outliers_zscore_mahalanobis.py
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
print("B6. Outliers: Z-Score y Mahalanobis")
print("="*74)
print(f"datos: {X.shape[0]} obs. x {X.shape[1]} predictores")
if y is not None:
    print(f"y ('{TARGET}'): " + ("CLASIFICACIÓN" if ES_CLASIFICACION else "REGRESIÓN"))
print()

# ════════════════════════════════════════════════════════════════════════
#  CÓDIGO DEL MÉTODO
# ════════════════════════════════════════════════════════════════════════

# ═══ B6. Outliers estadísticos ═══
from scipy.stats import chi2
from sklearn.covariance import MinCovDet

# ── Z-Score clásico y modificado, por variable ─────────────────────────
z = np.abs((X - X.mean(0)) / X.std(0))
mad = np.median(np.abs(X - np.median(X,0)), axis=0)
z_mod = np.abs(0.6745 * (X - np.median(X,0)) / np.where(mad==0, 1e-9, mad))

print("Outliers por variable (|Z| > 3):")
for j, nom in enumerate(nombres_X):
    print(f"  {nom:22s} Z clásico: {int((z[:,j]>3).sum()):3d}   Z modificado: {int((z_mod[:,j]>3).sum()):3d}")
print("El Z modificado detecta MÁS porque no se deja enmascarar.")

# ── Mahalanobis clásica y robusta ──────────────────────────────────────
p_var = X.shape[1]
umbral = chi2.ppf(0.975, df=p_var)          # ⬅ df = nº de VARIABLES

dif = X - X.mean(0)
d2 = np.einsum("ij,jk,ik->i", dif, np.linalg.inv(np.cov(X, rowvar=False)), dif)
d2_rob = MinCovDet(random_state=0).fit(X).mahalanobis(X)   # versión robusta (MCD)

print(f"\numbral chi2(0.975, df={p_var}) = {umbral:.2f}")
print(f"atípicos, Mahalanobis clásica : {int((d2>umbral).sum())}")
print(f"atípicos, Mahalanobis robusta : {int((d2_rob>umbral).sum())}  <- detecta más")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for a, d, t in [(ax[0], d2, "Mahalanobis clásica"), (ax[1], d2_rob, "Mahalanobis robusta (MCD)")]:
    m = d > umbral
    a.scatter(Z2[~m,0], Z2[~m,1], alpha=.6, label="normales")
    a.scatter(Z2[m,0], Z2[m,1], c="red", s=55, label=f"atípicos ({int(m.sum())})")
    a.set_xlabel("PC1"); a.set_ylabel("PC2"); a.set_title(t); a.legend()
plt.tight_layout(); plt.show()

print(f"\nFilas más atípicas (índices): {np.argsort(d2_rob)[::-1][:8]}")

print("\nListo. Revisa la carpeta figuras/")
