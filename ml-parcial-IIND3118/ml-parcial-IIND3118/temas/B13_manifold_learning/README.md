---
# B13. Manifold Learning: Isomap, LLE y t-SNE

## La idea, en simple
A veces los datos viven sobre una **superficie doblada** dentro del espacio. Imagina una hoja de papel enrollada en 3D: en realidad es 2D, solo está torcida.

PCA la **aplasta** y pone juntos puntos que sobre la hoja están lejísimos. Estos métodos la **desenrollan**.

La clave: distinguir la distancia **en línea recta** (a través del aire) de la distancia **caminando sobre la superficie** (geodésica).

## Isomap
= **MDS con distancias geodésicas**. Construye un grafo de $k$ vecinos, calcula caminos más cortos sobre el grafo (eso aproxima la geodésica) y aplica MDS.

✅ Preserva estructura **global** · solución cerrada
❌ **Muy sensible a k**: k grande → crea **"atajos"** que saltan de un pliegue a otro y destruyen todo; k chico → el grafo se desconecta · $O(n^3)$

## LLE
Preserva **relaciones lineales locales**: los pesos con que cada punto se reconstruye desde sus vecinos deben seguir valiendo en el mapa nuevo.
Variantes: `standard` · `modified` · `hessian` · **`ltsa`** (suele dar lo mejor)

❌ No preserva distancias globales · tiende a **colapsar regiones**

## t-SNE
Convierte distancias en **probabilidades** y minimiza la divergencia KL. Usa una **t de Student** en baja dimensión (colas pesadas) para dar espacio entre clusters.

### ⚠️ Lo que NO se puede leer de un t-SNE (esto lo preguntan)
- **Los tamaños de los clusters no significan nada**
- **Las distancias ENTRE clusters no significan nada**
- Puede **inventar clusters** en datos sin estructura
- **No es determinista** · **no proyecta datos nuevos**

## Parámetros
| Método | Parámetro | Qué poner |
|---|---|---|
| Isomap | `n_neighbors` | **El crítico.** 5–15 típico |
| LLE | `n_neighbors`, `method` | `'ltsa'` suele ser la mejor variante |
| t-SNE | **`perplexity`** | **5–50.** Baja → grumos falsos; alta → se aplana |
| t-SNE | `init` | `'pca'` es más estable que `'random'` |

## ⚠️ EL CONTROL OBLIGATORIO
**Estos métodos SIEMPRE dibujan algo bonito, incluso con ruido puro.** Antes de interpretar, corre el mismo método sobre datos permutados y compara. El código lo hace.

---

## Archivos de este tema

- **`python/B13_manifold_learning.py`** — código listo para correr. Cambia el bloque ✏️ por tu dataset.
- **`R/B13_manifold_learning.R`** — el equivalente en R.
- Las gráficas se guardan en `figuras/` al ejecutar.

## Cómo correrlo

```bash
cd temas/B13_manifold_learning/python
python B13_manifold_learning.py
```

Para volver al índice general: [`../../README.md`](../../README.md) ·
Para el cheatsheet completo: [`../../cheatsheet_python.ipynb`](../../cheatsheet_python.ipynb)
