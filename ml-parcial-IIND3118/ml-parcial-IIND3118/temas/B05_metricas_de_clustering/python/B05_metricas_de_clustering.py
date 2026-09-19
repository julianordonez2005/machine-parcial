"""
B5. Métricas de clustering — cómo elegir K
Tema B05_metricas_de_clustering — Introducción al Machine Learning (IIND3118)

CÓMO USARLO
  1. Cambia el bloque marcado con ✏️ abajo por tu dataset.
  2. Corre el archivo:  python B05_metricas_de_clustering.py
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
print("B5. Métricas de clustering — cómo elegir K")
print("="*74)
print(f"datos: {X.shape[0]} obs. x {X.shape[1]} predictores")
if y is not None:
    print(f"y ('{TARGET}'): " + ("CLASIFICACIÓN" if ES_CLASIFICACION else "REGRESIÓN"))
print()

# ════════════════════════════════════════════════════════════════════════
#  CÓDIGO DEL MÉTODO
# ════════════════════════════════════════════════════════════════════════

# ═══ B5. Elegir K ═══
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             calinski_harabasz_score, davies_bouldin_score)

Ks = range(2, 11)
wss, sil, ch, db_i = [], [], [], []

for k in Ks:
    lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xs)
    wss.append(KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs).inertia_)
    sil.append(silhouette_score(Xs, lab))            # MAXIMIZAR
    ch.append(calinski_harabasz_score(Xs, lab))      # MAXIMIZAR
    db_i.append(davies_bouldin_score(Xs, lab))       # MINIMIZAR

print(f"{'K':>3} | {'WSS':>9} | {'Silhouette':>10} | {'Calinski-H':>10} | {'Davies-B':>9}")
print("-"*52)
for i,k in enumerate(Ks):
    print(f"{k:3d} | {wss[i]:9.1f} | {sil[i]:10.3f} | {ch[i]:10.1f} | {db_i[i]:9.3f}")

k_sil = list(Ks)[int(np.argmax(sil))]
print(f"\nMejor K según Silhouette       : {k_sil}")
print(f"Mejor K según Calinski-Harabasz: {list(Ks)[int(np.argmax(ch))]}")
print(f"Mejor K según Davies-Bouldin   : {list(Ks)[int(np.argmin(db_i))]}")

fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
for a, v, t, best in [(ax[0], wss, "Codo: WSS (busca el quiebre)", None),
                      (ax[1], sil, "Silhouette (MAX)", np.argmax(sil)),
                      (ax[2], ch,  "Calinski-Harabasz (MAX)", np.argmax(ch)),
                      (ax[3], db_i,"Davies-Bouldin (MIN)", np.argmin(db_i))]:
    a.plot(list(Ks), v, "o-"); a.set_xlabel("K")
    if best is not None: a.axvline(list(Ks)[best], color="red", ls="--")
    a.set_title(t)
plt.tight_layout(); plt.show()

# ── Gráfico de silueta por cluster (más informativo que el promedio) ───
import matplotlib.cm as cm
lab = KMeans(n_clusters=k_sil, n_init=10, random_state=0).fit_predict(Xs)
si = silhouette_samples(Xs, lab)
y_low = 10
fig, a = plt.subplots(figsize=(7,4.5))
for kk in range(k_sil):
    v = np.sort(si[lab==kk]); y_up = y_low+len(v)
    a.fill_betweenx(np.arange(y_low,y_up), 0, v, facecolor=cm.viridis(kk/k_sil), alpha=.8)
    a.text(-0.05, y_low+.5*len(v), str(kk)); y_low = y_up+10
a.axvline(si.mean(), color="red", ls="--", label=f"promedio = {si.mean():.3f}")
a.set_xlabel("s(i)"); a.set_title(f"Silueta por cluster (K={k_sil})"); a.legend()
plt.show()
print(f"Observaciones con s(i)<0 (mal asignadas): {int((si<0).sum())}")

print("\nListo. Revisa la carpeta figuras/")
