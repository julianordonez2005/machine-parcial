"""
B12. SVD (Singular Value Decomposition)
Tema B12_svd — Introducción al Machine Learning (IIND3118)

CÓMO USARLO
  1. Cambia el bloque marcado con ✏️ abajo por tu dataset.
  2. Corre el archivo:  python B12_svd.py
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
print("B12. SVD (Singular Value Decomposition)")
print("="*74)
print(f"datos: {X.shape[0]} obs. x {X.shape[1]} predictores")
if y is not None:
    print(f"y ('{TARGET}'): " + ("CLASIFICACIÓN" if ES_CLASIFICACION else "REGRESIÓN"))
print()

# ════════════════════════════════════════════════════════════════════════
#  CÓDIGO DEL MÉTODO
# ════════════════════════════════════════════════════════════════════════

# ═══ B12. SVD ═══
# PCA se calcula aquí para poder verificar que SVD == PCA
pca = PCA().fit(Xs)

Xc = Xs - Xs.mean(0)                              # datos centrados
U, S, Vt = np.linalg.svd(Xc, full_matrices=False) # ⬅ False = versión económica

print(f"U: {U.shape}   S: {S.shape}   Vt: {Vt.shape}  (Vt ya está TRANSPUESTA)")
print(f"valores singulares: {np.round(S,3)}")
print(f"\nError de la descomposición: {np.abs(Xc - U@np.diag(S)@Vt).max():.2e}")

# ── SVD == PCA ─────────────────────────────────────────────────────────
print(f"\nvarianzas PCA      : {np.round(pca.explained_variance_,4)}")
print(f"sigma²/(n-1) de SVD: {np.round(S**2/(len(Xs)-1),4)}   <- iguales")

# ── Eckart-Young: el error se sabe en fórmula cerrada ─────────────────
energia = np.cumsum(S**2)/np.sum(S**2)
print(f"\n{'k':>3} | {'energía':>9} | {'error real':>11} | {'error teórico':>13}")
print("-"*45)
for k in range(1, min(6, len(S))+1):
    Ak = U[:,:k]@np.diag(S[:k])@Vt[:k,:]
    print(f"{k:3d} | {energia[k-1]:9.4f} | {np.linalg.norm(Xc-Ak,'fro'):11.4f} | "
          f"{np.sqrt(np.sum(S[k:]**2)):13.4f}")
print("Los valores singulares descartados dicen EXACTAMENTE cuánto se pierde.")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(range(1,len(S)+1), S, "o-"); ax[0].set_yscale("log")
ax[0].set_xlabel("i"); ax[0].set_ylabel("valor singular σᵢ")
ax[0].set_title("Valores singulares (escala log)")
ax[1].plot(range(1,len(S)+1), energia, "o-")
ax[1].axhline(.95, color="red", ls="--", label="95%")
ax[1].set_xlabel("k"); ax[1].set_ylabel("energía acumulada")
ax[1].set_title("Fracción de información retenida"); ax[1].legend()
plt.tight_layout(); plt.show()

print("\nListo. Revisa la carpeta figuras/")
