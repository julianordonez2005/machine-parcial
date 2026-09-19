---
# A6. Validación cruzada (k-fold)

## Qué hace, en simple

Partir en train/test una sola vez tiene un problema: **el resultado depende de qué datos cayeron en test por azar**. Con mala suerte te toca una partición rara y evalúas mal el modelo.

La validación cruzada arregla eso: parte los datos en **k pedazos (folds)**. Entrena k veces; en cada una deja un pedazo distinto fuera para evaluar. Al final **promedia los k errores**.

Así **todos los datos sirven para entrenar y todos sirven para evaluar**, solo que en rondas distintas.

## El procedimiento con k=5

```
Ronda 1:  [TEST] [train][train][train][train]
Ronda 2:  [train][TEST] [train][train][train]
Ronda 3:  [train][train][TEST] [train][train]
Ronda 4:  [train][train][train][TEST] [train]
Ronda 5:  [train][train][train][train][TEST]
                  ↓
        MSE_CV = promedio de los 5 MSE
```

## Parámetros

| Parámetro | Qué poner |
|---|---|
| `cv` / `n_splits` (k) | **5 o 10** son los estándar. k grande = menos sesgo pero más lento |
| `scoring` | `'neg_mean_squared_error'` (regresión) · `'accuracy'` o `'roc_auc'` (clasificación). **Ojo: sklearn devuelve el MSE en negativo** porque siempre maximiza |
| `shuffle=True` | Mezcla antes de partir. **Ponlo**, salvo en series de tiempo |
| `StratifiedKFold` | Úsalo en clasificación: mantiene la proporción de clases en cada fold |

## Casos especiales
- **LOOCV** (*leave-one-out*): k = n. Cada fold es una sola observación. Sin sesgo pero muy lento y con alta varianza
- **`GridSearchCV`**: hace CV automáticamente probando todas las combinaciones de parámetros. Es lo que usarás en la práctica

## Ventajas / desventajas
- ✅ Usa todos los datos, resultado más estable, no depende de una partición con suerte
- ❌ **k veces más lento**; con series de tiempo no se puede usar así (hay que usar `TimeSeriesSplit`)

---

## Archivos de este tema

- **`python/A06_validacion_cruzada.py`** — código listo para correr. Cambia el bloque ✏️ por tu dataset.
- **`R/A06_validacion_cruzada.R`** — el equivalente en R.
- Las gráficas se guardan en `figuras/` al ejecutar.

## Cómo correrlo

```bash
cd temas/A06_validacion_cruzada/python
python A06_validacion_cruzada.py
```

Para volver al índice general: [`../../README.md`](../../README.md) ·
Para el cheatsheet completo: [`../../cheatsheet_python.ipynb`](../../cheatsheet_python.ipynb)
