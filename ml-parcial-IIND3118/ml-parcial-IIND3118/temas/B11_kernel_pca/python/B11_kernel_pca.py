"""
B11. Kernel PCA
Tema B11_kernel_pca — Introducción al Machine Learning (IIND3118)

CÓMO USARLO
  1. Cambia el bloque marcado con ✏️ abajo por tu dataset.
  2. Corre el archivo:  python B11_kernel_pca.py
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
print("B11. Kernel PCA")
print("="*74)
print(f"datos: {X.shape[0]} obs. x {X.shape[1]} predictores")
if y is not None:
    print(f"y ('{TARGET}'): " + ("CLASIFICACIÓN" if ES_CLASIFICACION else "REGRESIÓN"))
print()

# ════════════════════════════════════════════════════════════════════════
#  CÓDIGO DEL MÉTODO
# ════════════════════════════════════════════════════════════════════════

# ═══ B11. Kernel PCA ═══
from scipy.spatial.distance import pdist as pd_

gamma_sug = 1/np.median(pd_(Xs)**2)       # heurística de la mediana
print(f"gamma sugerido por la heurística de la mediana: {gamma_sug:.4f}")

# ── Barrido de gamma: el parámetro que lo decide todo ──────────────────
gammas = [gamma_sug/100, gamma_sug/10, gamma_sug, gamma_sug*10, gamma_sug*100]
fig, axes = plt.subplots(1, len(gammas)+1, figsize=(3.2*(len(gammas)+1), 3.4))

axes[0].scatter(Z[:,0], Z[:,1], c=col, cmap="viridis", s=18)
axes[0].set_title("PCA lineal\n(compara contra esto)"); axes[0].set_xticks([]); axes[0].set_yticks([])

for a, g in zip(axes[1:], gammas):
    sc = KernelPCA(n_components=2, kernel="rbf", gamma=g).fit_transform(Xs)
    a.scatter(sc[:,0], sc[:,1], c=col, cmap="viridis", s=18)
    a.set_title(f"gamma={g:.4f}"); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()
print("gamma chico -> se parece a PCA lineal | gamma grande -> se desarma")

# ── Modelo final ───────────────────────────────────────────────────────
kpca = KernelPCA(
    n_components=2,
    kernel="rbf",          # ⬅ 'rbf' | 'poly' | 'linear' | 'sigmoid'
    gamma=gamma_sug,       # ⬅ EL parámetro crítico
).fit(Xs)
sc_kpca = kpca.transform(Xs)

plt.scatter(sc_kpca[:,0], sc_kpca[:,1], c=col, cmap="viridis", alpha=.7)
plt.xlabel("KPC1"); plt.ylabel("KPC2"); plt.title(f"Kernel PCA (rbf, gamma={gamma_sug:.4f})")
plt.show()

# KPCA SÍ puede proyectar datos nuevos (a diferencia de MDS/t-SNE)
print(f"Proyección de datos nuevos: kpca.transform(X_nuevo) -> {kpca.transform(Xs[:5]).shape}")

print("\nListo. Revisa la carpeta figuras/")
