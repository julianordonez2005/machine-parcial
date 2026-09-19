---
# A3. KNN y kernel regression

## KNN (k vecinos más cercanos), en simple

Para predecir un punto nuevo: **busca los k puntos más parecidos** en los datos de entrenamiento y devuelve el **promedio** de sus $y$ (regresión) o la **clase más votada** (clasificación).

No aprende ninguna fórmula: se guarda todos los datos y compara. Por eso se llama método *perezoso*.

## Kernel regression, en simple

Casi igual, pero en vez de "los k más cercanos" usa **todos los que caigan dentro de una ventana de ancho h**:

$$\hat f(X^*) = \frac{\sum_i \mathbb{I}(|X_i - X^*| \leq h)\, Y_i}{\sum_i \mathbb{I}(|X_i - X^*| \leq h)}$$

Diferencia: en KNN el número de vecinos es fijo y la distancia varía; en kernel la distancia es fija y el número de vecinos varía.

## ⚠️ Lo más importante: ESTANDARIZAR

Ambos usan **distancias**. Si una variable está en pesos (millones) y otra en años (decenas), la de pesos domina todo. **Usa siempre `Xs`, no `X`.**

## Parámetros

| Parámetro | Qué poner |
|---|---|
| `n_neighbors` (k) | El parámetro a calibrar. Empieza probando 1 a 40. **k pequeño = flexible** |
| `weights` | `'uniform'` (todos pesan igual) o `'distance'` (los más cercanos pesan más) |
| `metric` | `'minkowski'` con `p=2` es euclidiana (por defecto); `p=1` es Manhattan |

## Ventajas
- Muy simple, sin supuestos sobre la forma de la relación
- Captura relaciones no lineales automáticamente

## Desventajas
- **Hay que estandarizar sí o sí**
- Lento al predecir (compara contra todos los datos)
- **Falla en dimensión alta** (maldición de la dimensionalidad: en muchas dimensiones "cerca" deja de significar algo)
- No da un modelo interpretable: no hay coeficientes

---

## Archivos de este tema

- **`python/A03_knn_y_kernel_regression.py`** — código listo para correr. Cambia el bloque ✏️ por tu dataset.
- **`R/A03_knn_y_kernel_regression.R`** — el equivalente en R.
- Las gráficas se guardan en `figuras/` al ejecutar.

## Cómo correrlo

```bash
cd temas/A03_knn_y_kernel_regression/python
python A03_knn_y_kernel_regression.py
```

Para volver al índice general: [`../../README.md`](../../README.md) ·
Para el cheatsheet completo: [`../../cheatsheet_python.ipynb`](../../cheatsheet_python.ipynb)
