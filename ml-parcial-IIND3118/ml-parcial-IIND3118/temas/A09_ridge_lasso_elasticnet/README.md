---
# A9. Regularización: Ridge, Lasso y Elastic Net

## Qué hacen, en simple

La regresión normal solo minimiza el error. Estas versiones minimizan **el error MÁS un castigo por tener coeficientes grandes**:

$$\hat\beta = \arg\min \underbrace{\sum_i (Y_i - \beta^tX_i)^2}_{\text{error}} + \underbrace{\lambda \cdot \text{castigo}(\beta)}_{\text{penalización}}$$

La idea: si un coeficiente es enorme, el modelo probablemente está sobreajustando. **Encogiendo los coeficientes hacia cero se gana estabilidad**, a cambio de un poquito de sesgo.

## Las tres variantes

| Modelo | Castigo | Efecto sobre los coeficientes |
|---|---|---|
| **Ridge** (norma 2) | $\lambda\sum\beta_j^2$ | Los **encoge** hacia 0, pero **nunca los deja en 0 exacto**. Se quedan todas las variables |
| **Lasso** (norma 1) | $\lambda\sum\lvert\beta_j\rvert$ | Puede dejar coeficientes **exactamente en 0** → **selecciona variables automáticamente** |
| **Elastic Net** | $\lambda[(1-\alpha)\sum\beta_j^2 + \alpha\sum\lvert\beta_j\rvert]$ | Mezcla de los dos |

> **La diferencia clave que preguntan:** Lasso hace **selección de variables** (pone ceros); Ridge no, solo encoge.

En Elastic Net: $\alpha = 0$ → es Ridge · $\alpha = 1$ → es Lasso · $0 < \alpha < 1$ → mezcla.

## El parámetro $\lambda$ (`alpha` en sklearn)

| Valor | Efecto |
|---|---|
| $\lambda = 0$ | No hay castigo → es la regresión lineal normal |
| $\lambda$ pequeño | Poco castigo → modelo flexible |
| $\lambda$ grande | Mucho castigo → todos los coeficientes tienden a 0 → modelo rígido |
| $\lambda \to \infty$ | Todos los coeficientes en 0, el modelo predice la media |

**Se calibra con validación cruzada**, usando `RidgeCV`, `LassoCV` o `ElasticNetCV`.

## ⚠️ ESTANDARIZAR ES OBLIGATORIO

El castigo suma los coeficientes. Si una variable está en pesos y otra en años, sus coeficientes están en escalas distintas y el castigo las trata injustamente. **Usa siempre `Xs`.**

## Cuándo usar cuál

| Situación | Modelo |
|---|---|
| Muchas variables correlacionadas entre sí, todas aportan algo | **Ridge** |
| Muchas variables pero sospechas que pocas importan | **Lasso** |
| $p > n$ | Lasso o Elastic Net |
| Variables correlacionadas **y** quieres selección | **Elastic Net** |

## Parámetros

| Parámetro | Qué poner |
|---|---|
| `alphas` | Rejilla de $\lambda$ a probar. `np.logspace(-4, 2, 100)` cubre un rango amplio |
| `cv` | Folds para calibrar. 5 o 10 |
| `l1_ratio` | Solo Elastic Net: es el $\alpha$ de la fórmula (0=Ridge, 1=Lasso) |
| `max_iter` | Súbelo a 10000 si sale warning de convergencia |

---

## Archivos de este tema

- **`python/A09_ridge_lasso_elasticnet.py`** — código listo para correr. Cambia el bloque ✏️ por tu dataset.
- **`R/A09_ridge_lasso_elasticnet.R`** — el equivalente en R.
- Las gráficas se guardan en `figuras/` al ejecutar.

## Cómo correrlo

```bash
cd temas/A09_ridge_lasso_elasticnet/python
python A09_ridge_lasso_elasticnet.py
```

Para volver al índice general: [`../../README.md`](../../README.md) ·
Para el cheatsheet completo: [`../../cheatsheet_python.ipynb`](../../cheatsheet_python.ipynb)
