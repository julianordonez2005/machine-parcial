"""
A9. Regularización: Ridge, Lasso y Elastic Net
Tema A09_ridge_lasso_elasticnet — Introducción al Machine Learning (IIND3118)

CÓMO USARLO
  1. Cambia el bloque marcado con ✏️ abajo por tu dataset.
  2. Corre el archivo:  python A09_ridge_lasso_elasticnet.py
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
print("A9. Regularización: Ridge, Lasso y Elastic Net")
print("="*74)
print(f"datos: {X.shape[0]} obs. x {X.shape[1]} predictores")
if y is not None:
    print(f"y ('{TARGET}'): " + ("CLASIFICACIÓN" if ES_CLASIFICACION else "REGRESIÓN"))
print()

# ════════════════════════════════════════════════════════════════════════
#  CÓDIGO DEL MÉTODO
# ════════════════════════════════════════════════════════════════════════

# ═══ A9. Ridge, Lasso y Elastic Net ═══
# ⚠️ SIEMPRE con Xs (estandarizado)

alphas = np.logspace(-4, 2, 100)     # ⬅ rejilla de lambdas a probar

# ── 1. RIDGE ────────────────────────────────────────────────────────────
ridge = RidgeCV(alphas=alphas, cv=10).fit(Xs_train, y_train)
mse_ridge = mean_squared_error(y_test, ridge.predict(Xs_test))
print(f"RIDGE   lambda óptimo = {ridge.alpha_:.5f}   MSE test = {mse_ridge:.3f}")
print(f"        coeficientes en cero: {np.sum(ridge.coef_ == 0)} de {len(ridge.coef_)}  <- Ridge NUNCA da ceros")

# ── 2. LASSO ────────────────────────────────────────────────────────────
lasso = LassoCV(alphas=alphas, cv=10, max_iter=10000, random_state=0).fit(Xs_train, y_train)
mse_lasso = mean_squared_error(y_test, lasso.predict(Xs_test))
n_cero = np.sum(lasso.coef_ == 0)
print(f"\nLASSO   lambda óptimo = {lasso.alpha_:.5f}   MSE test = {mse_lasso:.3f}")
print(f"        coeficientes en cero: {n_cero} de {len(lasso.coef_)}  <- SELECCIONA variables")
print(f"        variables que sobreviven: {[nombres_X[i] for i in np.where(lasso.coef_ != 0)[0]]}")

# ── 3. ELASTIC NET ──────────────────────────────────────────────────────
elastic = ElasticNetCV(
    alphas=alphas,
    l1_ratio=[.1, .3, .5, .7, .9, .95, 1],   # ⬅ calibra también la mezcla
    cv=10, max_iter=10000, random_state=0,
).fit(Xs_train, y_train)
mse_elastic = mean_squared_error(y_test, elastic.predict(Xs_test))
print(f"\nELASTIC lambda = {elastic.alpha_:.5f}  l1_ratio = {elastic.l1_ratio_}  MSE test = {mse_elastic:.3f}")

# ── GRÁFICA CLAVE: caminos de los coeficientes ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
for ax, Modelo, nombre, kw in [
        (axes[0], Ridge,      "Ridge (norma 2)",     {}),
        (axes[1], Lasso,      "Lasso (norma 1)",     {"max_iter": 10000}),
        (axes[2], ElasticNet, "Elastic Net",         {"max_iter": 10000, "l1_ratio": 0.5})]:
    coefs = [Modelo(alpha=a, **kw).fit(Xs_train, y_train).coef_ for a in alphas]
    ax.plot(alphas, coefs)
    ax.set_xscale("log"); ax.invert_xaxis()     # lambda grande a la izquierda
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("lambda (⬅ mucho castigo | poco castigo ➡)")
    ax.set_ylabel("coeficientes"); ax.set_title(nombre)
plt.tight_layout(); plt.show()
print("En Lasso las líneas TOCAN el cero y se quedan ahí: eso es selección de")
print("variables. En Ridge se acercan a cero pero nunca lo tocan.")

# ── Comparación final ───────────────────────────────────────────────────
mse_ols = mean_squared_error(y_test, LinearRegression().fit(Xs_train, y_train).predict(Xs_test))
nombres_m = ["Lineal (OLS)", "Ridge", "Lasso", "Elastic Net"]
valores_m = [mse_ols, mse_ridge, mse_lasso, mse_elastic]

plt.bar(nombres_m, valores_m, color=["gray","steelblue","indianred","seagreen"])
plt.ylabel("MSE en test"); plt.title("Comparación de modelos con penalización")
for i, v in enumerate(valores_m): plt.text(i, v, f"{v:.1f}", ha="center", va="bottom")
plt.show()
print(f"Mejor modelo: {nombres_m[int(np.argmin(valores_m))]}")

print("\nListo. Revisa la carpeta figuras/")
