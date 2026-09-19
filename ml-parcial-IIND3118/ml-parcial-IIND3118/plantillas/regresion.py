# ============================================================================
# PLANTILLA: PROBLEMA DE REGRESIÓN — copiar y pegar
# Cambia solo la primera línea y el nombre de la columna respuesta.
# ============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ── ✏️ CAMBIA SOLO ESTO ─────────────────────────────────────────────────
df     = pd.read_csv("data/ejemplo_regresion.csv")
TARGET = "target"
# ────────────────────────────────────────────────────────────────────────

df = df.dropna()
X_df = pd.get_dummies(df.drop(columns=[TARGET]), drop_first=True).astype(float)
nombres_X = list(X_df.columns)
X, y = X_df.values, df[TARGET].values
Xs = scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=0)
Xs_train, Xs_test, _, _          = train_test_split(Xs, y, test_size=.30, random_state=0)

res = {}

# 1. Regresión lineal
res["Lineal"] = mean_squared_error(y_test, LinearRegression().fit(X_train,y_train).predict(X_test))

# 2. KNN (calibrar k con CV)  ⚠️ con Xs
g = GridSearchCV(KNeighborsRegressor(), {"n_neighbors": range(1,41)},
                 cv=10, scoring="neg_mean_squared_error").fit(Xs_train, y_train)
res[f"KNN (k={g.best_params_['n_neighbors']})"] = mean_squared_error(y_test, g.predict(Xs_test))

# 3. Ridge / Lasso / Elastic Net  ⚠️ con Xs
alphas = np.logspace(-4, 2, 100)
for nom, M in [("Ridge", RidgeCV(alphas=alphas, cv=10)),
               ("Lasso", LassoCV(alphas=alphas, cv=10, max_iter=10000, random_state=0)),
               ("ElasticNet", ElasticNetCV(alphas=alphas, l1_ratio=[.1,.5,.9,1],
                                           cv=10, max_iter=10000, random_state=0))]:
    m = M.fit(Xs_train, y_train)
    res[nom] = mean_squared_error(y_test, m.predict(Xs_test))
    print(f"{nom:12s} lambda={m.alpha_:.5f}  coefs en cero: {int(np.sum(m.coef_==0))}")

# ── Resultados ──────────────────────────────────────────────────────────
t = pd.DataFrame({"modelo": res.keys(), "MSE": res.values()}).sort_values("MSE")
t["RMSE"] = np.sqrt(t["MSE"])
print("\n", t.to_string(index=False))
print(f"\nMejor: {t.iloc[0]['modelo']}")

plt.barh(t["modelo"][::-1], t["MSE"][::-1], color="steelblue")
plt.xlabel("MSE en test"); plt.title("Comparación de modelos"); plt.tight_layout(); plt.show()
