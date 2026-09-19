---
# B7. Outliers: LOF e Isolation Forest

Estos dos atacan el problema desde ángulos **opuestos**.

## LOF (Local Outlier Factor)
Compara la **densidad de un punto con la de sus vecinos**. Si vive en una zona mucho más vacía que sus vecinos, es outlier.

| Valor | Significa |
|---|---|
| $LOF\approx 1$ | densidad normal |
| $LOF\gg 1$ | **outlier** |
| $LOF<1$ | dentro de un cluster denso |

Detecta **outliers LOCALES**: puntos normales globalmente pero raros en su vecindario.

> ⚠️ En sklearn, `negative_outlier_factor_` es **−LOF**: más negativo = más atípico. En R (`dbscan::lof`) es al revés.

## Isolation Forest
Idea distinta: **los puntos raros son fáciles de aislar**. Hace cortes aleatorios; un outlier queda solo tras pocos cortes, un punto en el montón necesita muchos. El puntaje es la longitud del camino.

Detecta **outliers GLOBALES**. Es $O(n\log n)$ → escala muy bien.

## Cuál usar
| Situación | Método |
|---|---|
| n grande, muchas variables | **Isolation Forest** |
| Clusters de densidades muy distintas | **LOF** |
| Pocas variables, datos ~normales | **Mahalanobis** (B6) |

**En la práctica: corre varios y mira dónde coinciden.**

## Parámetros
| Parámetro | Qué poner |
|---|---|
| `n_neighbors` (LOF) | **El crítico.** 10–35. Pequeño = ruidoso; grande = se vuelve global |
| `n_estimators` (iForest) | 100 basta |
| `max_samples` | 256 funciona tan bien como usar todo |
| `contamination` | Proporción esperada de outliers. `'auto'`, o fija un valor si lo sabes |

---

## Archivos de este tema

- **`python/B07_outliers_lof_isolationforest.py`** — código listo para correr. Cambia el bloque ✏️ por tu dataset.
- **`R/B07_outliers_lof_isolationforest.R`** — el equivalente en R.
- Las gráficas se guardan en `figuras/` al ejecutar.

## Cómo correrlo

```bash
cd temas/B07_outliers_lof_isolationforest/python
python B07_outliers_lof_isolationforest.py
```

Para volver al índice general: [`../../README.md`](../../README.md) ·
Para el cheatsheet completo: [`../../cheatsheet_python.ipynb`](../../cheatsheet_python.ipynb)
