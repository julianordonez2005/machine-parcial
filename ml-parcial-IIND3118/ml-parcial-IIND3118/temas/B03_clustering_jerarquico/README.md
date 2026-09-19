---
# B3. Clustering jerárquico

## Qué hace, en simple
Empieza con cada punto solo. En cada paso **une los dos grupos más cercanos**. Sigue hasta que queda uno. El resultado es un **árbol (dendrograma)** que muestra todo el proceso.

Para obtener grupos concretos, **cortas el árbol** a una altura. Cortar más arriba = menos grupos.

## El linkage: lo que de verdad decide el resultado
| Linkage | Distancia entre grupos | Comportamiento |
|---|---|---|
| `single` | la **mínima** entre puntos | formas alargadas; sufre *chaining* (une grupos por una cadena de puntos) |
| `complete` | la **máxima** | grupos compactos; sensible a outliers |
| `average` | el **promedio** | término medio, buen default |
| `ward` | menor aumento de WSS | grupos balanceados, lo más parecido a K-Means |

> **El resultado depende más del linkage que del algoritmo.** En R usa `ward.D2`, no `ward.D`.

## Parámetros
`method` (el linkage) · `metric` (`euclidean` obligatoria con ward) · `t` + `criterion='maxclust'` para pedir K grupos

✅ No hay que fijar K de antemano · determinista · el dendrograma da información de parentesco
❌ **Fusiones irreversibles** (greedy) · $O(n^2)$–$O(n^3)$ · muy sensible al linkage · siempre dibuja un árbol, haya estructura o no

---

## Archivos de este tema

- **`python/B03_clustering_jerarquico.py`** — código listo para correr. Cambia el bloque ✏️ por tu dataset.
- **`R/B03_clustering_jerarquico.R`** — el equivalente en R.
- Las gráficas se guardan en `figuras/` al ejecutar.

## Cómo correrlo

```bash
cd temas/B03_clustering_jerarquico/python
python B03_clustering_jerarquico.py
```

Para volver al índice general: [`../../README.md`](../../README.md) ·
Para el cheatsheet completo: [`../../cheatsheet_python.ipynb`](../../cheatsheet_python.ipynb)
