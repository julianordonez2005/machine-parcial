---
# B4. DBSCAN

## Qué hace, en simple
Busca **zonas densas**: un cluster es un montón de puntos pegados entre sí. No exige que todos los puntos del grupo estén cerca del centro, solo que haya una **cadena de vecinos densos** que los conecte. Por eso encuentra **formas raras** (espirales, anillos, medialunas).

Además, **los puntos que no caen en ninguna zona densa los marca como RUIDO** (`-1`). Es el único método de clustering del curso que dice "este punto no va en ningún lado".

## Los 3 tipos de punto
- **core**: tiene al menos `min_samples` vecinos dentro del radio `eps`
- **border**: no es core, pero está dentro del radio de un core
- **noise**: ninguna de las dos → etiqueta **`-1`**

## Cómo elegir `eps` — la curva k-dist
1. Calcula la distancia de cada punto a su k-ésimo vecino
2. Ordénalas y grafícalas
3. El **"codo"** de la curva es un buen `eps`

## Parámetros
| Parámetro | Qué poner |
|---|---|
| `eps` | **El crítico.** Radio de la vecindad. Se lee del codo de la curva k-dist |
| `min_samples` | Regla práctica: **≥ p+1**, o `2p` si hay ruido |

**Diagnóstico rápido:** casi todo es ruido → `eps` muy chico. Un solo cluster gigante → `eps` muy grande.

✅ Formas arbitrarias · no fija K · detecta outliers
❌ Calibración difícil y muy sensible · **falla si los clusters tienen densidades distintas** (usa HDBSCAN) · se degrada en dimensión alta

---

## Archivos de este tema

- **`python/B04_dbscan.py`** — código listo para correr. Cambia el bloque ✏️ por tu dataset.
- **`R/B04_dbscan.R`** — el equivalente en R.
- Las gráficas se guardan en `figuras/` al ejecutar.

## Cómo correrlo

```bash
cd temas/B04_dbscan/python
python B04_dbscan.py
```

Para volver al índice general: [`../../README.md`](../../README.md) ·
Para el cheatsheet completo: [`../../cheatsheet_python.ipynb`](../../cheatsheet_python.ipynb)
