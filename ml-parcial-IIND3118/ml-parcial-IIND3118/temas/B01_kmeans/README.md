---
# B1. K-Means

## Qué hace, en simple
Parte los datos en **K grupos**. Cada grupo tiene un **centro** (el promedio de sus puntos), y cada punto se asigna **al centro más cercano**. Repite: recalcula centros → reasigna puntos, hasta que nada cambia.

## Fórmula
$$\min \sum_{k}\sum_{x\in C_k}\|x-\bar x_k\|^2 \quad (\text{WSS})$$

**Identidad ANOVA:** $SST = BSS + WSS$, con $SST$ fijo ⇒ **minimizar WSS = maximizar BSS**. Coherencia interna y separación son el mismo objetivo.

## Parámetros
| Parámetro | Qué poner |
|---|---|
| `n_clusters` (K) | El número de grupos. **Calibrar con el codo o Silhouette (B5)** |
| `n_init` | **10 o más.** Repite con distintos arranques y se queda con el mejor. Defensa contra óptimos locales |
| `init` | `'k-means++'` (por defecto, mejor que aleatorio) |
| `random_state` | Fijo, para reproducir |

## Ventajas / desventajas
✅ Rapidísimo, escala a millones · centroides interpretables · asigna puntos nuevos sin reentrenar
❌ **Hay que fijar K** · sensible a inicialización, escala y outliers · **solo clusters redondos y de tamaño parecido** · **siempre devuelve K grupos aunque no haya estructura**

---

## Archivos de este tema

- **`python/B01_kmeans.py`** — código listo para correr. Cambia el bloque ✏️ por tu dataset.
- **`R/B01_kmeans.R`** — el equivalente en R.
- Las gráficas se guardan en `figuras/` al ejecutar.

## Cómo correrlo

```bash
cd temas/B01_kmeans/python
python B01_kmeans.py
```

Para volver al índice general: [`../../README.md`](../../README.md) ·
Para el cheatsheet completo: [`../../cheatsheet_python.ipynb`](../../cheatsheet_python.ipynb)
