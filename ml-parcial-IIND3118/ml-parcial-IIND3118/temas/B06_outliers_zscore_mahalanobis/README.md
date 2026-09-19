---
# B6. Outliers: Z-Score y Mahalanobis

## Z-Score (una variable a la vez)
$$Z(x_i)=\frac{x_i-\bar x}{sd}$$
Regla: $|Z|>3$ es atípico (≈0.27% bajo normalidad).

### ⚠️ El enmascaramiento
Los outliers **inflan la media y la sd**, y así **se esconden a sí mismos**. Con varios outliers juntos, sus propios Z caen bajo 3 y no se detectan. Además el Z tiene un **techo**: $|Z|\le\frac{n-1}{\sqrt n}$.

**Solución: Z modificado**, con mediana y MAD (aguanta hasta 50% de contaminación):
$$Z_{mod}=\frac{0.6745(x_i-\text{mediana})}{MAD}$$

## Mahalanobis (todas las variables juntas)
$$d_M(x)^2=(x-\bar x)^\top S^{-1}(x-\bar x)\sim\chi^2_p$$

Corrige por **escala y correlación**. Sus fronteras son **elipses**, no círculos.
**Umbral formal:** percentil 97.5 de $\chi^2_p$ (con $p$ = número de **variables**).

> **Clave:** mirar variable por variable **NO** es lo mismo que Mahalanobis. Un punto puede ser normal en cada variable y atípico en conjunto por **romper la correlación**.

## Caso Hadlum (lo preguntan)
Un embarazo de 349 días es improbable pero posible. **Un outlier estadístico no es automáticamente un error o un fraude**: es una observación improbable *bajo el modelo asumido*. Con n grande, siempre habrá puntos en las colas.

## Versión robusta
`MinCovDet` (MCD) estima la covarianza con la mayoría "limpia" de los datos → no se deja engañar por el grupo contaminante.

---

## Archivos de este tema

- **`python/B06_outliers_zscore_mahalanobis.py`** — código listo para correr. Cambia el bloque ✏️ por tu dataset.
- **`R/B06_outliers_zscore_mahalanobis.R`** — el equivalente en R.
- Las gráficas se guardan en `figuras/` al ejecutar.

## Cómo correrlo

```bash
cd temas/B06_outliers_zscore_mahalanobis/python
python B06_outliers_zscore_mahalanobis.py
```

Para volver al índice general: [`../../README.md`](../../README.md) ·
Para el cheatsheet completo: [`../../cheatsheet_python.ipynb`](../../cheatsheet_python.ipynb)
