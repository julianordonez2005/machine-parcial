"""
B7. Outliers: LOF e Isolation Forest
Tema B07_outliers_lof_isolationforest — Introducción al Machine Learning (IIND3118)

CÓMO USARLO
  1. Cambia el bloque marcado con ✏️ abajo por tu dataset.
  2. Corre el archivo:  python B07_outliers_lof_isolationforest.py
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
print("B7. Outliers: LOF e Isolation Forest")
print("="*74)
print(f"datos: {X.shape[0]} obs. x {X.shape[1]} predictores")
if y is not None:
    print(f"y ('{TARGET}'): " + ("CLASIFICACIÓN" if ES_CLASIFICACION else "REGRESIÓN"))
print()

# ════════════════════════════════════════════════════════════════════════
#  CÓDIGO DEL MÉTODO
# ════════════════════════════════════════════════════════════════════════

# ═══ B7. LOF e Isolation Forest ═══

# ── LOF ────────────────────────────────────────────────────────────────
lof = LocalOutlierFactor(n_neighbors=20, contamination="auto")  # ⬅ n_neighbors: el crítico
y_lof = lof.fit_predict(Xs)                      # -1 = outlier, 1 = normal
score_lof = lof.negative_outlier_factor_         # ⬅ es -LOF: más negativo = más atípico

# ── Isolation Forest ───────────────────────────────────────────────────
iso = IsolationForest(n_estimators=100, max_samples=256,
                      contamination="auto", random_state=0).fit(Xs)
y_if = iso.predict(Xs)                           # -1 = outlier
score_if = iso.decision_function(Xs)             # ⬅ más BAJO = más anómalo

es_lof, es_if = y_lof==-1, y_if==-1
print(f"outliers según LOF            : {int(es_lof.sum())}")
print(f"outliers según Isolation Forest: {int(es_if.sum())}")
print(f"\ncoinciden en                  : {int((es_lof & es_if).sum())}")
print(f"solo LOF (LOCALES)            : {int((es_lof & ~es_if).sum())}")
print(f"solo iForest (GLOBALES)       : {int((~es_lof & es_if).sum())}")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].scatter(Z2[:,0], Z2[:,1], c=es_lof, cmap="coolwarm", alpha=.7); ax[0].set_title("LOF")
ax[1].scatter(Z2[:,0], Z2[:,1], c=es_if, cmap="coolwarm", alpha=.7); ax[1].set_title("Isolation Forest")
ax[2].scatter(Z2[~es_lof & ~es_if,0], Z2[~es_lof & ~es_if,1], c="lightgray", s=25, label="normales")
ax[2].scatter(Z2[es_lof & es_if,0],   Z2[es_lof & es_if,1],   c="red", s=65, label="ambos")
ax[2].scatter(Z2[es_lof & ~es_if,0],  Z2[es_lof & ~es_if,1],  c="blue", marker="^", s=65, label="solo LOF")
ax[2].scatter(Z2[~es_lof & es_if,0],  Z2[~es_lof & es_if,1],  c="green", marker="s", s=65, label="solo iForest")
ax[2].legend(fontsize=8); ax[2].set_title("Dónde coinciden y dónde no")
for a in ax: a.set_xlabel("PC1"); a.set_ylabel("PC2")
plt.tight_layout(); plt.show()

print(f"\nFilas más anómalas según iForest: {np.argsort(score_if)[:8]}")

print("\nListo. Revisa la carpeta figuras/")
