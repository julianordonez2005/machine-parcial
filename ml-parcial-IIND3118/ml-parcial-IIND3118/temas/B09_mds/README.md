---
# B9. MDS (Multidimensional Scaling)

## Qué hace, en simple
Tienes **solo las distancias** entre observaciones, no sus coordenadas. MDS **reconstruye un mapa** de puntos que respete esas distancias.

El ejemplo clásico: con las distancias por carretera entre ciudades europeas, MDS **dibuja el mapa de Europa** sin haber visto nunca una latitud.

## Cómo funciona
Doble centrado para pasar de distancias a productos punto, y luego descomposición espectral:
$$B=-\tfrac12 C D^{(2)} C, \qquad C=I-\tfrac1n\mathbf{1}\mathbf{1}^\top$$

## ⚠️ Equivalencia clave (la preguntan)
> **MDS clásico con distancia euclidiana = PCA.** Es el mismo procedimiento visto desde dos lados: PCA descompone la covarianza ($p\times p$), MDS la de productos punto ($n\times n$).

## Autovalores negativos = diagnóstico
Si aparecen y son grandes, **la matriz D no es euclidiana**: ninguna configuración de puntos puede reproducir esas distancias exactamente. Pasa con Gower y con distancias por carretera.

## MDS no métrico (Kruskal)
Solo respeta el **orden** de las disimilitudes, no sus valores. Para datos ordinales (Likert, rankings). Minimiza el **STRESS**:
> <0.05 excelente · <0.10 bueno · <0.20 aceptable

## Parámetros
`n_components` (2 para visualizar) · `metric` (True=clásico, False=Kruskal) · `dissimilarity='precomputed'` si le pasas una matriz D

❌ $O(n^3)$ · **no proyecta datos nuevos** (PCA sí) · los ejes **no tienen interpretación** (se puede rotar/reflejar libremente)

---

## Archivos de este tema

- **`python/B09_mds.py`** — código listo para correr. Cambia el bloque ✏️ por tu dataset.
- **`R/B09_mds.R`** — el equivalente en R.
- Las gráficas se guardan en `figuras/` al ejecutar.

## Cómo correrlo

```bash
cd temas/B09_mds/python
python B09_mds.py
```

Para volver al índice general: [`../../README.md`](../../README.md) ·
Para el cheatsheet completo: [`../../cheatsheet_python.ipynb`](../../cheatsheet_python.ipynb)
