"""
A7. Regresión lineal
Tema A07_regresion_lineal — Introducción al Machine Learning (IIND3118)

CÓMO USARLO
  1. Cambia el bloque marcado con ✏️ abajo por tu dataset.
  2. Corre el archivo:  python A07_regresion_lineal.py
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
print("A7. Regresión lineal")
print("="*74)
print(f"datos: {X.shape[0]} obs. x {X.shape[1]} predictores")
if y is not None:
    print(f"y ('{TARGET}'): " + ("CLASIFICACIÓN" if ES_CLASIFICACION else "REGRESIÓN"))
print()

# ════════════════════════════════════════════════════════════════════════
#  CÓDIGO DEL MÉTODO
# ════════════════════════════════════════════════════════════════════════

# ═══ A7. Regresión lineal ═══

reg = LinearRegression(   # ⬅ para clasificación usa LogisticRegression
    
    fit_intercept=True,     # ⬅ True casi siempre (incluye el β₀)
).fit(X_train, y_train)

# ── Coeficientes ordenados por magnitud ─────────────────────────────────
coefs = pd.DataFrame({"variable": nombres_X, "coeficiente": reg.coef_})
coefs["|coef|"] = coefs["coeficiente"].abs()
print(f"Intercepto (β₀): {reg.intercept_:.4f}\n")
print(coefs.sort_values("|coef|", ascending=False).to_string(index=False))
print("\nLectura: si la variable sube 1 unidad, y cambia en 'coeficiente'")
print("unidades, MANTENIENDO LAS DEMÁS CONSTANTES.")

pred_lm = reg.predict(X_test)
mse_lm = mean_squared_error(y_test, pred_lm)
print(f"\nMSE test: {mse_lm:.3f}   |   R² test: {r2_score(y_test, pred_lm):.4f}")

# ── Gráfica de coeficientes ─────────────────────────────────────────────
c = coefs.sort_values("coeficiente")
plt.barh(c["variable"], c["coeficiente"], color=np.where(c["coeficiente"]>0, "steelblue", "indianred"))
plt.axvline(0, color="k", lw=.8)
plt.xlabel("coeficiente"); plt.title("Coeficientes de la regresión lineal")
plt.tight_layout(); plt.show()

# ── Tabla completa con valores p (statsmodels) ──────────────────────────
import statsmodels.api as sm
ols = sm.OLS(y_train, sm.add_constant(X_train)).fit()
print(ols.summary().tables[1])
print("\nP>|t| < 0.05  ->  la variable es estadísticamente significativa.")

# ── Diagnóstico de multicolinealidad (VIF) ─────────────────────────────
from statsmodels.stats.outliers_influence import variance_inflation_factor
Xc = sm.add_constant(X_train)
vif = pd.DataFrame({
    "variable": ["const"] + nombres_X,
    "VIF": [variance_inflation_factor(Xc, i) for i in range(Xc.shape[1])]
})
print("\n", vif[vif.variable != "const"].sort_values("VIF", ascending=False).to_string(index=False))
print("VIF > 10  ->  multicolinealidad seria. Usa Ridge/Lasso (tema A9).")

print("\nListo. Revisa la carpeta figuras/")
