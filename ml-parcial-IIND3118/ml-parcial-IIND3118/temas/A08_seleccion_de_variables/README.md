---
# A8. Selección de variables

## El problema

Con $p$ variables hay $2^p - 1$ modelos posibles. Con 20 variables son más de un millón. ¿Cuál conjunto usar?

Meter todas no es la respuesta: variables irrelevantes **agregan varianza sin agregar señal**, y hacen el modelo menos interpretable.

## Método exhaustivo

Prueba **todos** los $2^p - 1$ modelos y se queda con el mejor.

- ✅ **Garantiza encontrar el mejor** conjunto bajo la métrica elegida
- ❌ **Crece exponencialmente**: imposible más allá de ~20 variables

## Método forward (hacia adelante)

1. Empieza sin ninguna variable
2. Agrega la que **más mejore** el modelo
3. Repite, dejando fijas las ya escogidas
4. Para cuando ninguna mejora

Compara solo $\frac{p(p+1)}{2}$ modelos: con 20 variables son 210, no un millón.

- ✅ Mucho más rápido, sirve con $p$ grande
- ❌ **Es una heurística: NO garantiza el mejor modelo.** Una variable escogida al principio nunca se vuelve a quitar, aunque deje de ser útil

## Método backward (hacia atrás)

Al revés: empieza con todas y va **quitando** la menos útil. Requiere $n > p$.

## Comparación

| | Exhaustivo | Forward | Backward |
|---|---|---|---|
| Modelos evaluados | $2^p - 1$ | $p(p+1)/2$ | $p(p+1)/2$ |
| Garantiza el óptimo | **Sí** | No | No |
| Sirve con $p$ grande | No | **Sí** | Necesita $n>p$ |

## Parámetros

| Parámetro | Qué poner |
|---|---|
| `n_features_to_select` | Cuántas variables quedarse. `'auto'` deja que decida |
| `direction` | `'forward'` o `'backward'` |
| `scoring` | `'neg_mean_squared_error'`, `'r2'`, `'accuracy'`… |
| `cv` | Folds para evaluar cada candidato. 5 está bien |

---

## Archivos de este tema

- **`python/A08_seleccion_de_variables.py`** — código listo para correr. Cambia el bloque ✏️ por tu dataset.
- **`R/A08_seleccion_de_variables.R`** — el equivalente en R.
- Las gráficas se guardan en `figuras/` al ejecutar.

## Cómo correrlo

```bash
cd temas/A08_seleccion_de_variables/python
python A08_seleccion_de_variables.py
```

Para volver al índice general: [`../../README.md`](../../README.md) ·
Para el cheatsheet completo: [`../../cheatsheet_python.ipynb`](../../cheatsheet_python.ipynb)
