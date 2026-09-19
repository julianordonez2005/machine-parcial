# ============================================================================
# PLANTILLA: PROBLEMA NO SUPERVISADO — copiar y pegar
# ============================================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# ── ✏️ CAMBIA SOLO ESTO ─────────────────────────────────────────────────
df = pd.read_csv("data/ejemplo_regresion.csv")
# ────────────────────────────────────────────────────────────────────────

df = df.dropna()
X_df = pd.get_dummies(df, drop_first=True).astype(float)
nombres_X = list(X_df.columns)
Xs = scale(X_df.values)                    # ⚠️ SIEMPRE estandarizar

# ── 1. PCA: reducir y visualizar ────────────────────────────────────────
pca = PCA().fit(Xs)
Z = pca.transform(Xs)
print("varianza explicada:", np.round(pca.explained_variance_ratio_[:5], 3))
print(f"componentes para 90%: {int(np.argmax(np.cumsum(pca.explained_variance_ratio_)>=.90))+1}")

loadings = pd.DataFrame(pca.components_.T, index=nombres_X,
                        columns=[f"PC{i+1}" for i in range(pca.n_components_)])
print("\nLoadings:\n", loadings.iloc[:, :3].round(3).to_string())

# ── 2. Elegir K con Silhouette ──────────────────────────────────────────
sil = [silhouette_score(Xs, KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xs))
       for k in range(2, 11)]
K = list(range(2,11))[int(np.argmax(sil))]
print(f"\nMejor K según Silhouette: {K}  (score = {max(sil):.3f})")

# ── 3. K-Means con el K elegido ─────────────────────────────────────────
km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(Xs)
print("tamaño de grupos:", np.bincount(km.labels_))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(range(2,11), sil, "o-"); ax[0].axvline(K, color="red", ls="--")
ax[0].set_xlabel("K"); ax[0].set_title("Silhouette (MAX)")
ax[1].plot(np.cumsum(pca.explained_variance_ratio_), "o-")
ax[1].axhline(.90, color="red", ls="--"); ax[1].set_title("Varianza acumulada (PCA)")
ax[2].scatter(Z[:,0], Z[:,1], c=km.labels_, cmap="viridis")
ax[2].set_xlabel("PC1"); ax[2].set_ylabel("PC2"); ax[2].set_title(f"K-Means (K={K})")
plt.tight_layout(); plt.show()
