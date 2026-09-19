# Introducción al Machine Learning — IIND3118
### Repositorio de estudio y consulta para el parcial

Cubre **todo el curso**: aprendizaje supervisado (clases 10–14) y no supervisado (clases 1–9).
23 temas, cada uno con explicación, código en Python, código en R y gráficas.

---

## 🚀 Lo único que necesitas saber para usarlo

Todo el código está escrito para que **solo cambies el nombre de tu dataset**. En cada archivo busca este bloque:

```
╔═════════════════════════════════════════════════════════╗
║  ✏️  CAMBIA SOLO ESTE BLOQUE POR TU DATASET             ║
╚═════════════════════════════════════════════════════════╝
DATOS  = "data/mis_datos.csv"
TARGET = "nombre_de_la_columna_y"      # None si no hay respuesta
```

Y ya. El código automáticamente:

- convierte las columnas de texto a variables dummy
- estandariza y hace la partición train/test 70–30
- **detecta solo si es regresión o clasificación** y ajusta modelos y métricas
- guarda las gráficas en `figuras/`

Después de esa línea quedan listas estas variables, **iguales en todos los temas**:

| Variable | Qué es |
|---|---|
| `X` / `Xs` | Predictores, crudos / estandarizados |
| `y` | Variable respuesta |
| `nombres_X` | Nombres de las columnas |
| `X_train, X_test, y_train, y_test` | Partición 70/30 |
| `Xs_train, Xs_test` | La misma partición, estandarizada |
| `Z2` | Proyección PCA en 2D (para graficar) |

> **La única regla que hay que recordar:** si el método usa **distancias** (KNN, K-Means, DBSCAN, PCA, LOF) o es **lineal con penalización** (Ridge/Lasso), usa **`Xs`**. La regresión lineal simple funciona con `X`.

---

## 📂 Tres formas de usar el repositorio

| Quiero… | Abre |
|---|---|
| **Consultar rápido durante el parcial** | [`cheatsheet_python.ipynb`](cheatsheet_python.ipynb) — todo en un notebook, ya ejecutado con 32 gráficas |
| **Estudiar un tema a fondo** | La carpeta de ese tema en [`temas/`](temas/) — explicación + código que corre solo |
| **Resolver un punto ya, sin leer** | [`plantillas/`](plantillas/) — código mínimo copiar-y-pegar |

---

## 📑 Índice de temas

### Parte A — Aprendizaje SUPERVISADO (clases 10–14)

| | Tema | Qué resuelve |
|---|---|---|
| [A01](temas/A01_train_test_y_mse/) | Partición train/test y MSE | Cómo evaluar bien un modelo |
| [A02](temas/A02_flexibilidad_sesgo_varianza/) | Flexibilidad y sesgo–varianza | El concepto base: la curva en U |
| [A03](temas/A03_knn_y_kernel_regression/) | KNN y kernel regression | Modelo flexible sin supuestos |
| [A04](temas/A04_metricas_de_evaluacion/) | Métricas | MSE, RMSE, MAE, R², confusión, ROC |
| [A05](temas/A05_aic_bic_cp_r2ajustado/) | AIC, BIC, Cp, R² ajustado | Comparar modelos castigando complejidad |
| [A06](temas/A06_validacion_cruzada/) | Validación cruzada (k-fold) | Calibrar sin depender del azar |
| [A07](temas/A07_regresion_lineal/) | Regresión lineal | El modelo base interpretable |
| [A08](temas/A08_seleccion_de_variables/) | Selección de variables | Forward, backward, exhaustiva |
| [A09](temas/A09_ridge_lasso_elasticnet/) | Ridge, Lasso, Elastic Net | Multicolinealidad y $p$ grande |
| [A10](temas/A10_comparacion_de_modelos/) | Comparación de modelos | Cerrar el análisis |

### Parte B — Aprendizaje NO SUPERVISADO (clases 1–9)

| | Tema | Qué resuelve |
|---|---|---|
| [B01](temas/B01_kmeans/) | K-Means | Agrupar, clusters redondos |
| [B02](temas/B02_kmedoides_gower/) | K-Medoides + Gower | Agrupar con outliers o categóricas |
| [B03](temas/B03_clustering_jerarquico/) | Clustering jerárquico | Ver la jerarquía, no fijar K |
| [B04](temas/B04_dbscan/) | DBSCAN | Formas raras + detectar ruido |
| [B05](temas/B05_metricas_de_clustering/) | Métricas de clustering | Elegir K |
| [B06](temas/B06_outliers_zscore_mahalanobis/) | Z-Score y Mahalanobis | Outliers, pocas variables |
| [B07](temas/B07_outliers_lof_isolationforest/) | LOF e Isolation Forest | Outliers locales / muchas variables |
| [B08](temas/B08_pca/) | PCA | Reducir dimensión, visualizar |
| [B09](temas/B09_mds/) | MDS | Solo tengo distancias |
| [B10](temas/B10_analisis_factorial/) | Análisis Factorial | Factores latentes interpretables |
| [B11](temas/B11_kernel_pca/) | Kernel PCA | Estructura no lineal |
| [B12](temas/B12_svd/) | SVD | Aproximación de rango bajo, LSI |
| [B13](temas/B13_manifold_learning/) | Isomap, LLE, t-SNE | Visualizar estructura curva |

Cada carpeta tiene: `README.md` (explicación) · `python/` · `R/` · `figuras/` (se llena al correr).

---

## 🧭 Qué método usar

**¿Tienes una variable respuesta `y`?**

**SÍ → supervisado**

| Situación | Método |
|---|---|
| Punto de partida siempre | Regresión lineal ([A07](temas/A07_regresion_lineal/)) |
| Relación no lineal | KNN ([A03](temas/A03_knn_y_kernel_regression/)) |
| Muchas variables correlacionadas | **Ridge** ([A09](temas/A09_ridge_lasso_elasticnet/)) |
| Muchas variables, pocas importan | **Lasso** (da ceros) ([A09](temas/A09_ridge_lasso_elasticnet/)) |
| Correlación **y** quieres selección | **Elastic Net** ([A09](temas/A09_ridge_lasso_elasticnet/)) |
| $p$ pequeño, quieres el mejor subconjunto | Exhaustivo ([A08](temas/A08_seleccion_de_variables/)) |
| $p$ grande, seleccionar rápido | Forward ([A08](temas/A08_seleccion_de_variables/)) |
| Calibrar cualquier parámetro | Validación cruzada ([A06](temas/A06_validacion_cruzada/)) |

**NO → no supervisado**

| Situación | Método |
|---|---|
| Agrupar, clusters redondos | K-Means ([B01](temas/B01_kmeans/)) |
| Agrupar con outliers o categóricas | K-Medoides + Gower ([B02](temas/B02_kmedoides_gower/)) |
| Clusters de forma rara + ruido | DBSCAN ([B04](temas/B04_dbscan/)) |
| ¿Cuántos grupos? | Codo / Silhouette ([B05](temas/B05_metricas_de_clustering/)) |
| Reducir dimensión / visualizar | PCA ([B08](temas/B08_pca/)) |
| Outliers, pocas variables | Mahalanobis ([B06](temas/B06_outliers_zscore_mahalanobis/)) |
| Outliers, muchas variables | Isolation Forest ([B07](temas/B07_outliers_lof_isolationforest/)) |
| Factores ocultos interpretables | Análisis Factorial ([B10](temas/B10_analisis_factorial/)) |

---

## ⚙️ Instalación

**Python**
```bash
pip install -r requirements.txt
```

**R**
```r
source("install.R")
```

---

## ▶️ Cómo correr un tema

```bash
cd temas/B08_pca/python
python B08_pca.py
```

```bash
cd temas/B08_pca/R
Rscript B08_pca.R
```

Las gráficas quedan en `temas/B08_pca/figuras/`.

---

## ⚠️ Estado de verificación

| Qué | Estado |
|---|---|
| `cheatsheet_python.ipynb` | ✅ Ejecutado completo: 52 celdas, 32 gráficas, **0 errores** |
| Flujo "cambiar una línea" | ✅ Probado con **dos datasets distintos** (regresión y clasificación) |
| Los 23 scripts de Python de `temas/` | ✅ Corridos uno por uno, **todos pasan de forma autónoma** |
| Las 3 plantillas de Python | ✅ Corren sin errores |
| `cheatsheet_R.R` y los 23 scripts de R | ⚠️ **No se pudieron ejecutar**: el entorno donde se generó este repositorio no tenía R instalado. Solo se verificó que la sintaxis esté balanceada. **Córrelos una vez antes del parcial.** |

---

## 📝 Las trampas del curso

Cada cheatsheet cierra con **las 15 trampas más preguntadas** y las **equivalencias entre métodos**. Vale la pena leerlas antes de entrar. Un adelanto:

- El MSE de **train** siempre baja con más flexibilidad — calibra con test o CV
- En KNN, Ridge y kernel, **el parámetro grande = modelo rígido** (al revés de lo intuitivo)
- **BIC castiga más que AIC** ⇒ elige modelos más pequeños
- **Lasso da ceros exactos** (selecciona variables); **Ridge nunca**
- **p-valor ALTO** en análisis factorial = **modelo adecuado** (al revés de lo habitual)
- En **t-SNE**, los tamaños y las distancias entre clusters **no significan nada**
- **MDS clásico = PCA** · **PCA = SVD centrada** · **Isomap = MDS geodésico**
