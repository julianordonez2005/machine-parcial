# ============================================================================
# PLANTILLA: PROBLEMA DE CLASIFICACIÓN — copiar y pegar
# ============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, roc_auc_score)

# ── ✏️ CAMBIA SOLO ESTO ─────────────────────────────────────────────────
df     = pd.read_csv("data/ejemplo_clasificacion.csv")
TARGET = "target"
# ────────────────────────────────────────────────────────────────────────

df = df.dropna()
X_df = pd.get_dummies(df.drop(columns=[TARGET]), drop_first=True).astype(float)
X  = X_df.values
y  = pd.factorize(df[TARGET])[0] if df[TARGET].dtype == object else df[TARGET].values
Xs = scale(X)

# stratify=y mantiene la proporción de clases en train y test
Xs_train, Xs_test, y_train, y_test = train_test_split(
    Xs, y, test_size=.30, random_state=0, stratify=y)

for nom, mod in [("Logística", LogisticRegression(max_iter=5000)),
                 ("KNN", GridSearchCV(KNeighborsClassifier(),
                                      {"n_neighbors": range(1,41)}, cv=10))]:
    m = mod.fit(Xs_train, y_train)
    pred = m.predict(Xs_test)
    print(f"\n── {nom} ──")
    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    cm = confusion_matrix(y_test, pred)
    print("Matriz de confusión (filas=real, columnas=predicho):\n", cm)
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        print(f"Sensibilidad : {tp/(tp+fn):.4f}")
        print(f"Especificidad: {tn/(tn+fp):.4f}")
    print(classification_report(y_test, pred, zero_division=0))

# Curva ROC (solo binaria)
if len(np.unique(y)) == 2:
    proba = LogisticRegression(max_iter=5000).fit(Xs_train,y_train).predict_proba(Xs_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y_test, proba):.3f}")
    plt.plot([0,1],[0,1],"r--", label="azar")
    plt.xlabel("1 - Especificidad"); plt.ylabel("Sensibilidad")
    plt.title("Curva ROC"); plt.legend(); plt.show()
