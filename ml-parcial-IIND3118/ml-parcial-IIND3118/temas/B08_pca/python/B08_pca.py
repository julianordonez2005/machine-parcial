"""
B8. PCA (Análisis de Componentes Principales)
Tema B08_pca — Introducción al Machine Learning (IIND3118)

CÓMO USARLO
  1. Cambia el bloque marcado con ✏️ abajo por tu dataset.
  2. Corre el archivo:  python B08_pca.py
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
print("B8. PCA (Análisis de Componentes Principales)")
print("="*74)
print(f"datos: {X.shape[0]} obs. x {X.shape[1]} predictores")
if y is not None:
    print(f"y ('{TARGET}'): " + ("CLASIFICACIÓN" if ES_CLASIFICACION else "REGRESIÓN"))
print()

# ════════════════════════════════════════════════════════════════════════
#  CÓDIGO DEL MÉTODO
# ════════════════════════════════════════════════════════════════════════

# ═══ B8. PCA ═══  (usa Xs)
pca = PCA(n_components=None).fit(Xs)     # ⬅ None = todos; 0.95 = los que expliquen 95%
Z = pca.transform(Xs)                    # Z = los SCORES (coordenadas nuevas)

print("Varianza explicada por componente:")
for i,(lam,r) in enumerate(zip(pca.explained_variance_, pca.explained_variance_ratio_),1):
    print(f"  PC{i}: lambda={lam:7.4f}  ({r:6.2%})  acumulada={pca.explained_variance_ratio_[:i].sum():6.2%}")

n_90 = int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.90)) + 1
print(f"\nSe necesitan {n_90} componentes para explicar el 90% de la varianza.")
print(f"Criterio de Kaiser (lambda>1): {int((pca.explained_variance_>1).sum())} componentes")

# ── LOS COEFICIENTES (loadings) ────────────────────────────────────────
loadings = pd.DataFrame(pca.components_.T,
                        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
                        index=nombres_X)
print("\nLoadings (coeficientes de cada componente):")
print(loadings.iloc[:, :min(4, loadings.shape[1])].round(3).to_string())
print("\nLee cada COLUMNA: qué variables pesan más en ese componente.")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].plot(range(1,len(pca.explained_variance_)+1), pca.explained_variance_, "o-")
ax[0].axhline(1, color="red", ls="--", label="Kaiser (λ=1)")
ax[0].set_xlabel("componente"); ax[0].set_ylabel("varianza (λ)")
ax[0].set_title("Scree plot (codo)"); ax[0].legend()

ax[1].plot(range(1,len(pca.explained_variance_)+1),
           np.cumsum(pca.explained_variance_ratio_), "o-")
ax[1].axhline(.90, color="red", ls="--", label="90%")
ax[1].set_xlabel("nº de componentes"); ax[1].set_ylabel("varianza acumulada")
ax[1].set_title("Varianza acumulada"); ax[1].legend()

col = y if y is not None else "steelblue"
sc = ax[2].scatter(Z[:,0], Z[:,1], c=col, cmap="viridis", alpha=.7)
ax[2].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax[2].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax[2].set_title("Datos proyectados en 2D")
plt.tight_layout(); plt.show()

# ── Biplot: observaciones + flechas de variables ───────────────────────
esc = np.abs(Z[:,:2]).max() / np.abs(pca.components_[:2].T).max() * 0.6
plt.scatter(Z[:,0], Z[:,1], alpha=.25, s=25)
for i, var in enumerate(nombres_X):
    plt.arrow(0,0, pca.components_[0,i]*esc, pca.components_[1,i]*esc,
              color="red", head_width=esc*.03, lw=1.5)
    plt.text(pca.components_[0,i]*esc*1.15, pca.components_[1,i]*esc*1.15,
             var, color="red", fontsize=8)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Biplot: observaciones + variables")
plt.show()

print("\nListo. Revisa la carpeta figuras/")
