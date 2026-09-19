---
# B5. Métricas de clustering — cómo elegir K

## El problema
En clustering no hay respuesta correcta conocida. Estas métricas ayudan a decidir K.

## Método del codo
Grafica WSS vs. K. El WSS **siempre baja** (con K=n es 0), así que no se minimiza: se busca **dónde deja de bajar rápido**.

## Silhouette — la más usada
Por cada observación:
$$s(i)=\frac{b(i)-a(i)}{\max\{a(i),b(i)\}}\in[-1,1]$$
- $a(i)$ = distancia promedio a los de **su** grupo (cohesión)
- $b(i)$ = distancia promedio a los del grupo **vecino más cercano** (separación)

| $s(i)$ | Significa |
|---|---|
| ≈ 1 | bien asignada |
| ≈ 0 | en la frontera |
| < 0 | **probablemente mal asignada** |

Lectura del promedio: >0.7 fuerte · 0.5–0.7 razonable · 0.25–0.5 débil · <0.25 sin estructura

## Las otras dos
- **Calinski-Harabasz** = $\frac{BSS/(K-1)}{WSS/(n-K)}$ → **maximizar**
- **Davies-Bouldin** → **minimizar**

## ⚠️ La trampa (esto lo preguntan)
**Las tres asumen clusters redondos y separados por sus centros.** Con anillos concéntricos premian a K-Means aunque esté visiblemente mal, porque los dos anillos comparten centroide. **No las uses para comparar K-Means contra DBSCAN en formas irregulares.**

Y si las tres discrepan: no las promedies. Significa que **no hay estructura clara** y K se decide por criterio del problema.

---

## Archivos de este tema

- **`python/B05_metricas_de_clustering.py`** — código listo para correr. Cambia el bloque ✏️ por tu dataset.
- **`R/B05_metricas_de_clustering.R`** — el equivalente en R.
- Las gráficas se guardan en `figuras/` al ejecutar.

## Cómo correrlo

```bash
cd temas/B05_metricas_de_clustering/python
python B05_metricas_de_clustering.py
```

Para volver al índice general: [`../../README.md`](../../README.md) ·
Para el cheatsheet completo: [`../../cheatsheet_python.ipynb`](../../cheatsheet_python.ipynb)
