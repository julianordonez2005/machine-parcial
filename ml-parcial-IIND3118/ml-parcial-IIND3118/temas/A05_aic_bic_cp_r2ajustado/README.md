---
# A5. AIC, BIC, Cp de Mallows y R² ajustado

## El problema que resuelven

Si agregas variables a una regresión, el **R² siempre sube** y el **MSE de train siempre baja**, aunque las variables sean basura. Necesitas una métrica que **castigue la complejidad**.

Estas cuatro métricas hacen exactamente eso: miden el ajuste **y le restan un castigo** por cada parámetro extra. Son una alternativa al MSE de test cuando no quieres (o no puedes) partir la muestra.

## Las fórmulas

| Métrica | Fórmula | Dirección |
|---|---|---|
| **R² ajustado** | $1 - \frac{(1-R^2)(n-1)}{n-p-1}$ | **Maximizar** |
| **AIC** | $-2\log L + 2p$ | **Minimizar** |
| **BIC** | $-2\log L + p\log n$ | **Minimizar** |
| **Cp de Mallows** | $\frac{1}{n}(SSE + 2p\hat\sigma^2)$ | **Minimizar** |

donde $p$ = número de parámetros y $n$ = número de observaciones.

## AIC vs. BIC — la diferencia que preguntan

El castigo del BIC es $p\log n$; el del AIC es $2p$. Como $\log n > 2$ siempre que $n > 7$:

> **El BIC castiga más fuerte la complejidad que el AIC.** Por eso el BIC tiende a elegir **modelos más pequeños** (menos variables) que el AIC.

- **AIC**: busca el modelo que mejor predice
- **BIC**: busca el modelo "verdadero", es más conservador

---

## Archivos de este tema

- **`python/A05_aic_bic_cp_r2ajustado.py`** — código listo para correr. Cambia el bloque ✏️ por tu dataset.
- **`R/A05_aic_bic_cp_r2ajustado.R`** — el equivalente en R.
- Las gráficas se guardan en `figuras/` al ejecutar.

## Cómo correrlo

```bash
cd temas/A05_aic_bic_cp_r2ajustado/python
python A05_aic_bic_cp_r2ajustado.py
```

Para volver al índice general: [`../../README.md`](../../README.md) ·
Para el cheatsheet completo: [`../../cheatsheet_python.ipynb`](../../cheatsheet_python.ipynb)
