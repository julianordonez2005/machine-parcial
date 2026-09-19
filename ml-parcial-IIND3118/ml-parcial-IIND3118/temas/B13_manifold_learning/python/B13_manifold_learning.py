"""
B13. Manifold Learning: Isomap, LLE y t-SNE
Tema B13_manifold_learning — Introducción al Machine Learning (IIND3118)

CÓMO USARLO
  1. Cambia el bloque marcado con ✏️ abajo por tu dataset.
  2. Corre el archivo:  python B13_manifold_learning.py
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
print("B13. Manifold Learning: Isomap, LLE y t-SNE")
print("="*74)
print(f"datos: {X.shape[0]} obs. x {X.shape[1]} predictores")
if y is not None:
    print(f"y ('{TARGET}'): " + ("CLASIFICACIÓN" if ES_CLASIFICACION else "REGRESIÓN"))
print()

# ════════════════════════════════════════════════════════════════════════
#  CÓDIGO DEL MÉTODO
# ════════════════════════════════════════════════════════════════════════

# ═══ B13. Manifold Learning ═══

K_VEC = min(10, len(Xs)-1)       # ⬅ n_neighbors: el parámetro crítico
PERP  = min(30, (len(Xs)-1)//3)  # ⬅ perplexity de t-SNE (5-50)

emb_pca  = Z[:, :2]
emb_iso  = manifold.Isomap(n_neighbors=K_VEC, n_components=2).fit_transform(Xs)
# LLE es numéricamente delicado: si 'ltsa' falla, se prueban las otras
# variantes. 'hessian' es la más frágil de todas.
emb_lle = None
for metodo in ["ltsa", "modified", "standard"]:          # ⬅ variantes de LLE
    try:
        emb_lle = manifold.LocallyLinearEmbedding(
            n_neighbors=K_VEC, n_components=2,
            method=metodo, eigen_solver="dense",         # 'dense' es más estable
            random_state=0).fit_transform(Xs)
        print(f"LLE: funcionó con method='{metodo}'")
        break
    except Exception as e:
        print(f"LLE: method='{metodo}' falló ({type(e).__name__})")
if emb_lle is None:
    emb_lle = Z[:, :2]
    print("LLE falló con todas las variantes; se muestra PCA en su lugar.")
emb_tsne = manifold.TSNE(n_components=2, perplexity=PERP, init="pca",
                         random_state=0).fit_transform(Xs)

fig, ax = plt.subplots(1, 4, figsize=(17, 4))
for a,(e,t) in zip(ax, [(emb_pca,"PCA (lineal)"), (emb_iso,f"Isomap (k={K_VEC})"),
                        (emb_lle,"LLE"), (emb_tsne,f"t-SNE (perp={PERP})")]):
    a.scatter(e[:,0], e[:,1], c=col, cmap="viridis", s=20, alpha=.8)
    a.set_title(t); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()

# ── Sensibilidad a k en Isomap ─────────────────────────────────────────
ks_try = [3, 5, 10, 25, min(60, len(Xs)-1)]
fig, ax = plt.subplots(1, len(ks_try), figsize=(3.2*len(ks_try), 3.4))
for a, kk in zip(ax, ks_try):
    try:
        e = manifold.Isomap(n_neighbors=kk, n_components=2).fit_transform(Xs)
        a.scatter(e[:,0], e[:,1], c=col, cmap="viridis", s=15)
        a.set_title(f"k={kk}")
    except Exception:
        a.set_title(f"k={kk}: falla\n(grafo desconectado)")
    a.set_xticks([]); a.set_yticks([])
fig.suptitle("Isomap: con k grande aparecen ATAJOS que rompen el manifold")
plt.tight_layout(); plt.show()

# ── ⚠️ EL CONTROL: ¿qué sale con datos SIN estructura? ────────────────
ruido = np.random.default_rng(0).standard_normal(Xs.shape)   # ruido puro
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].scatter(*PCA(n_components=2).fit_transform(ruido).T, s=15, alpha=.7)
ax[0].set_title("PCA sobre RUIDO")
for i, pp in enumerate([5, PERP]):
    e = manifold.TSNE(n_components=2, perplexity=pp, init="random",
                      random_state=0).fit_transform(ruido)
    ax[i+1].scatter(*e.T, s=15, alpha=.7)
    ax[i+1].set_title(f"t-SNE sobre RUIDO (perp={pp})")
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.suptitle("Todos dibujan algo, aunque NO haya nada que encontrar", fontsize=13)
plt.tight_layout(); plt.show()
print("Con perplexity baja, t-SNE fabrica grumos convincentes sobre ruido puro.")
print("MORALEJA: un gráfico bonito NO es evidencia de estructura.")

print("\nListo. Revisa la carpeta figuras/")
