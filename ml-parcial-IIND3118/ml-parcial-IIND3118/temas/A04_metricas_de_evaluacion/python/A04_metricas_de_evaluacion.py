"""
A4. Métricas de evaluación
Tema A04_metricas_de_evaluacion — Introducción al Machine Learning (IIND3118)

CÓMO USARLO
  1. Cambia el bloque marcado con ✏️ abajo por tu dataset.
  2. Corre el archivo:  python A04_metricas_de_evaluacion.py
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
print("A4. Métricas de evaluación")
print("="*74)
print(f"datos: {X.shape[0]} obs. x {X.shape[1]} predictores")
if y is not None:
    print(f"y ('{TARGET}'): " + ("CLASIFICACIÓN" if ES_CLASIFICACION else "REGRESIÓN"))
print()

# ════════════════════════════════════════════════════════════════════════
#  CÓDIGO DEL MÉTODO
# ════════════════════════════════════════════════════════════════════════

# ═══ A4. Todas las métricas de un modelo ═══

def evaluar(y_real, y_pred, nombre="modelo"):
    """Imprime las métricas correctas según el tipo de problema."""
    print(f"── {nombre} ──")
    if ES_CLASIFICACION:
        cm = confusion_matrix(y_real, y_pred)
        print(f"Accuracy : {accuracy_score(y_real, y_pred):.4f}")
        print("\nMatriz de confusión (filas = real, columnas = predicho):")
        print(cm)
        if cm.shape == (2, 2):                      # métricas binarias
            tn, fp, fn, tp = cm.ravel()
            print(f"\nSensibilidad (recall): {tp/(tp+fn):.4f}  <- de los SÍ, cuántos detecté")
            print(f"Especificidad       : {tn/(tn+fp):.4f}  <- de los NO, cuántos descarté")
            print(f"Precisión           : {tp/(tp+fp):.4f}  <- de mis SÍ, cuántos acerté")
        print("\n", classification_report(y_real, y_pred, zero_division=0))
    else:
        mse = mean_squared_error(y_real, y_pred)
        print(f"MSE  : {mse:.4f}")
        print(f"RMSE : {np.sqrt(mse):.4f}   (unidades de y)")
        print(f"MAE  : {mean_absolute_error(y_real, y_pred):.4f}   (robusto a outliers)")
        print(f"R²   : {r2_score(y_real, y_pred):.4f}   (% de varianza explicada)")
        if np.all(y_real != 0):
            print(f"MAPE : {np.mean(np.abs((y_pred-y_real)/y_real))*100:.2f}%")
    return None

modelo = ModeloLineal().fit(Xs_train, y_train)
nombre_mod = "Regresión logística" if ES_CLASIFICACION else "Regresión lineal"
evaluar(y_test, modelo.predict(Xs_test), nombre_mod)

# ── Gráfica de residuales (solo regresión) ──────────────────────────────
if not ES_CLASIFICACION:
    pred = modelo.predict(Xs_test)
    resid = y_test - pred
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(pred, resid, alpha=.6); ax[0].axhline(0, color="red", ls="--")
    ax[0].set_xlabel("predicho"); ax[0].set_ylabel("residual")
    ax[0].set_title("Residuales: deben verse SIN patrón")
    ax[1].hist(resid, bins=25, color="steelblue")
    ax[1].set_title("Histograma de residuales (ideal: centrado en 0)")
    plt.tight_layout(); plt.show()

# ── Curva ROC (solo clasificación binaria) ──────────────────────────────
if ES_CLASIFICACION and len(np.unique(y)) == 2:
    clf = LogisticRegression(max_iter=2000).fit(Xs_train, y_train)
    proba = clf.predict_proba(Xs_test)[:, 1]        # probabilidad de la clase 1
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0,1], [0,1], "r--", label="azar (AUC = 0.5)")
    plt.xlabel("1 - Especificidad (FPR)"); plt.ylabel("Sensibilidad (TPR)")
    plt.title("Curva ROC"); plt.legend(); plt.show()

print("\nListo. Revisa la carpeta figuras/")
