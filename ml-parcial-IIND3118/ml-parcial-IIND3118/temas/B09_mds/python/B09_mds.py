"""
B9. MDS (Multidimensional Scaling)
Tema B09_mds — Introducción al Machine Learning (IIND3118)

CÓMO USARLO
  1. Cambia el bloque marcado con ✏️ abajo por tu dataset.
  2. Corre el archivo:  python B09_mds.py
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
print("B9. MDS (Multidimensional Scaling)")
print("="*74)
print(f"datos: {X.shape[0]} obs. x {X.shape[1]} predictores")
if y is not None:
    print(f"y ('{TARGET}'): " + ("CLASIFICACIÓN" if ES_CLASIFICACION else "REGRESIÓN"))
print()

# ════════════════════════════════════════════════════════════════════════
#  CÓDIGO DEL MÉTODO
# ════════════════════════════════════════════════════════════════════════

# ═══ B9. MDS ═══
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform

def cmdscale(D, k=2):
    """MDS clásico (= cmdscale de R). Devuelve (coordenadas, autovalores)."""
    n = D.shape[0]
    C = np.eye(n) - np.ones((n,n))/n          # matriz de centrado
    B = -0.5 * C @ (D**2) @ C                 # doble centrado
    val, vec = np.linalg.eigh(B)
    o = np.argsort(val)[::-1]; val, vec = val[o], vec[:,o]
    pos = val[:k] > 0
    coords = np.zeros((n,k)); coords[:,pos] = vec[:,:k][:,pos]*np.sqrt(val[:k][pos])
    return coords, val

D = squareform(pdist(Xs, metric="euclidean"))   # ⬅ cambia la métrica si quieres
pts, val = cmdscale(D, k=2)

print(f"autovalores (primeros 5): {np.round(val[:5],3)}")
print(f"negativos relevantes    : {int((val<-1e-8).sum())}  (0 => D SÍ es euclidiana)")
print(f"varianza explicada 2D   : {val[:2].sum()/val[val>0].sum():.2%}")

# Verificación: MDS clásico == PCA
print(f"\nMDS clásico vs PCA, diferencia máxima: "
      f"{np.abs(np.abs(pts)-np.abs(Z[:,:2])).max():.2e}  <- deben ser iguales")

# MDS no métrico (solo usa el ORDEN de las disimilitudes)
mds_nm = MDS(n_components=2, metric=False, dissimilarity="precomputed",
             random_state=0, n_init=4).fit_transform(D)

def stress(D_, p_):
    o, e = squareform(D_, checks=False), pdist(p_)
    s = np.sum(o*e)/np.sum(e**2)
    return np.sqrt(np.sum((o-s*e)**2)/np.sum(o**2))
print(f"STRESS-1 (no métrico): {stress(D, mds_nm):.4f}  (<0.10 es bueno)")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].scatter(pts[:,0], pts[:,1], c=col, cmap="viridis", alpha=.7); ax[0].set_title("MDS clásico (= PCA)")
ax[1].scatter(mds_nm[:,0], mds_nm[:,1], c=col, cmap="viridis", alpha=.7); ax[1].set_title("MDS no métrico")
o, e = squareform(D, checks=False), pdist(pts)
ax[2].scatter(o, e, alpha=.15, s=6); lim=[0,max(o.max(),e.max())]
ax[2].plot(lim, lim, "r--"); ax[2].set_xlabel("disimilitud original")
ax[2].set_ylabel("distancia en el mapa")
ax[2].set_title(f"Diagrama de Shepard (r={np.corrcoef(o,e)[0,1]:.3f})")
plt.tight_layout(); plt.show()
print("Shepard: entre más pegados a la diagonal, mejor preserva las distancias.")

print("\nListo. Revisa la carpeta figuras/")
