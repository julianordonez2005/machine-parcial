"""
B10. Análisis Factorial
Tema B10_analisis_factorial — Introducción al Machine Learning (IIND3118)

CÓMO USARLO
  1. Cambia el bloque marcado con ✏️ abajo por tu dataset.
  2. Corre el archivo:  python B10_analisis_factorial.py
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
print("B10. Análisis Factorial")
print("="*74)
print(f"datos: {X.shape[0]} obs. x {X.shape[1]} predictores")
if y is not None:
    print(f"y ('{TARGET}'): " + ("CLASIFICACIÓN" if ES_CLASIFICACION else "REGRESIÓN"))
print()

# ════════════════════════════════════════════════════════════════════════
#  CÓDIGO DEL MÉTODO
# ════════════════════════════════════════════════════════════════════════

# ═══ B10. Análisis Factorial ═══
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_d

def factanal(R, m):
    """Análisis factorial por máxima verosimilitud (= factanal de R)."""
    p = R.shape[0]
    def crit(psi):
        psi = np.clip(psi, 1e-6, 1.0); sc = 1/np.sqrt(psi)
        ev = np.linalg.eigvalsh(sc[:,None]*R*sc[None,:])[::-1]
        e = ev[m:]; return -np.sum(np.log(e)-e) - (p-m)
    r = minimize(crit, np.diag(np.linalg.inv(R))**-1*(1-m/(2*p)),
                 method="L-BFGS-B", bounds=[(.005,1)]*p)
    psi = np.clip(r.x,.005,1); sc = 1/np.sqrt(psi)
    val, vec = np.linalg.eigh(sc[:,None]*R*sc[None,:])
    o = np.argsort(val)[::-1]; val, vec = val[o], vec[:,o]
    L = np.sqrt(psi)[:,None]*vec[:,:m]*np.sqrt(np.maximum(val[:m]-1,0))
    for j in range(m):                       # signo arbitrario: se fija por legibilidad
        if L[np.argmax(np.abs(L[:,j])),j] < 0: L[:,j] *= -1
    return L, psi, r.fun

def varimax(L, tol=1e-6, itmax=500):
    """Rotación varimax: empuja cada carga hacia 0 o ±1 para poder interpretar."""
    p_, k = L.shape; Rot = np.eye(k); d = 0.
    for _ in range(itmax):
        Lr = L@Rot
        u,s,vt = np.linalg.svd(L.T@(Lr**3 - Lr@np.diag(np.sum(Lr**2,0))/p_))
        Rot = u@vt; dn = np.sum(s)
        if d != 0 and dn/d < 1+tol: break
        d = dn
    return L@Rot

R_mat = np.corrcoef(X, rowvar=False)
n_obs, p_var = X.shape

# ── Verificación previa ────────────────────────────────────────────────
stat_b = -(n_obs-1-(2*p_var+5)/6)*np.linalg.slogdet(R_mat)[1]
p_b = 1-chi2_d.cdf(stat_b, p_var*(p_var-1)/2)
Ri = np.linalg.inv(R_mat); dd = np.sqrt(np.diag(Ri))
par = -Ri/np.outer(dd,dd); np.fill_diagonal(par,0)
Rc = R_mat.copy(); np.fill_diagonal(Rc,0)
kmo = np.sum(Rc**2)/(np.sum(Rc**2)+np.sum(par**2))

print(f"Bartlett: p = {p_b:.2e}  -> {'hay correlaciones que factorizar' if p_b<.05 else 'NADA que factorizar'}")
print(f"KMO     : {kmo:.3f}  (>0.6 aceptable)")

m_max = int(np.floor((2*p_var+1-np.sqrt(8*p_var+1))/2 - 1e-9))
print(f"Máximo de factores identificables: {m_max}  (con p={p_var} variables)")

# ── Prueba chi2 para elegir m ──────────────────────────────────────────
print(f"\n{'m':>2} | {'chi2':>9} | {'gl':>3} | {'p-valor':>9} | decisión")
print("-"*50)
for m in range(1, min(m_max, 5)+1):
    L,psi,f = factanal(R_mat, m)
    stat = (n_obs-1-(2*p_var+5)/6-2*m/3)*f
    gl = int(((p_var-m)**2-p_var-m)/2)
    pv = 1-chi2_d.cdf(stat, gl) if gl>0 else np.nan
    print(f"{m:2d} | {stat:9.2f} | {gl:3d} | {pv:9.4f} | {'ADECUADO' if pv>=.05 else 'insuficiente'}")
print("⚠️  p-valor ALTO = el modelo SÍ es adecuado (al revés de lo habitual)")

# ── Ajuste final con rotación varimax ──────────────────────────────────
M = min(2, m_max)                      # ⬅ elige m según la tabla de arriba
L, psi, _ = factanal(R_mat, M)
Lv = varimax(L) if M > 1 else L

tab = pd.DataFrame(Lv, index=nombres_X, columns=[f"Factor{i+1}" for i in range(M)])
tab["comunalidad"] = (Lv**2).sum(1)    # varianza COMPARTIDA
tab["unicidad"] = psi                  # varianza PROPIA (PCA no separa esto)
print(f"\nCargas con rotación varimax (m={M}):")
print(tab.round(3).to_string())
print("\ncomunalidad + unicidad = 1. Cargas |·|>0.3 son las que 'definen' el factor.")

fig, ax = plt.subplots(figsize=(6, max(3, .35*len(nombres_X))))
im = ax.imshow(Lv, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(M)); ax.set_xticklabels([f"F{i+1}" for i in range(M)])
ax.set_yticks(range(len(nombres_X))); ax.set_yticklabels(nombres_X, fontsize=8)
for i in range(len(nombres_X)):
    for j in range(M):
        ax.text(j,i,f"{Lv[i,j]:.2f}",ha="center",va="center",fontsize=8,
                color="white" if abs(Lv[i,j])>.5 else "black")
plt.colorbar(im, label="carga"); ax.set_title("Cargas factoriales (varimax)")
plt.tight_layout(); plt.show()

print("\nListo. Revisa la carpeta figuras/")
