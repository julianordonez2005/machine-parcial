"""
B4. DBSCAN
Tema B04_dbscan — Introducción al Machine Learning (IIND3118)

CÓMO USARLO
  1. Cambia el bloque marcado con ✏️ abajo por tu dataset.
  2. Corre el archivo:  python B04_dbscan.py
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
print("B4. DBSCAN")
print("="*74)
print(f"datos: {X.shape[0]} obs. x {X.shape[1]} predictores")
if y is not None:
    print(f"y ('{TARGET}'): " + ("CLASIFICACIÓN" if ES_CLASIFICACION else "REGRESIÓN"))
print()

# ════════════════════════════════════════════════════════════════════════
#  CÓDIGO DEL MÉTODO
# ════════════════════════════════════════════════════════════════════════

# ═══ B4. DBSCAN ═══
from sklearn.neighbors import NearestNeighbors

# ── Paso 1: curva k-dist para elegir eps ───────────────────────────────
k = min(5, len(Xs)-1)
d_k = np.sort(NearestNeighbors(n_neighbors=k).fit(Xs).kneighbors(Xs)[0][:,1:].reshape(-1))

plt.plot(d_k); plt.xlabel("puntos ordenados"); plt.ylabel(f"distancia al vecino k={k}")
plt.title("Curva k-dist: busca el CODO para elegir eps"); plt.show()
print(f"Sugerencia: eps entre {np.percentile(d_k,80):.2f} (percentil 80) y "
      f"{np.percentile(d_k,95):.2f} (percentil 95)")

# ── Paso 2: probar varios eps ──────────────────────────────────────────
print("\nSensibilidad a eps:")
for eps in np.round(np.percentile(d_k, [50,70,80,90,95]), 2):
    lab = DBSCAN(eps=eps, min_samples=k+1).fit_predict(Xs)
    n_cl = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"  eps={eps:5.2f} -> {n_cl} clusters, {int((lab==-1).sum())} puntos de ruido")

# ── Paso 3: modelo final ───────────────────────────────────────────────
EPS = float(np.percentile(d_k, 90))     # ⬅ ajusta según la curva y la tabla de arriba
db = DBSCAN(eps=EPS, min_samples=k+1).fit(Xs)

n_cl = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
print(f"\nFINAL: eps={EPS:.2f} -> {n_cl} clusters, {int((db.labels_==-1).sum())} ruido")

mask = db.labels_ == -1
plt.scatter(Z2[~mask,0], Z2[~mask,1], c=db.labels_[~mask], cmap="viridis", alpha=.7)
plt.scatter(Z2[mask,0], Z2[mask,1], c="red", marker="x", s=60, label="ruido (-1)")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(f"DBSCAN (eps={EPS:.2f})")
plt.legend(); plt.show()

print("\nListo. Revisa la carpeta figuras/")
