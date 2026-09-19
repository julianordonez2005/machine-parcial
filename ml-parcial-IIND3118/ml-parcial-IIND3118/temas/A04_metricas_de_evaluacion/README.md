---
# A4. Métricas de evaluación

## REGRESIÓN

| Métrica | Fórmula | Cómo se lee |
|---|---|---|
| **MSE** | $\frac{1}{n}\sum(\hat y_i - y_i)^2$ | El estándar. **Minimizar.** Castiga mucho los errores grandes |
| **RMSE** | $\sqrt{MSE}$ | Igual que MSE pero **en las unidades de $y$**. Más fácil de interpretar |
| **MAE** | $\frac{1}{n}\sum\lvert\hat y_i - y_i\rvert$ | Más **robusto a outliers** que el MSE |
| **MAPE** | $\frac{100}{n}\sum\lvert\frac{\hat y_i - y_i}{y_i}\rvert$ | Error en **%**. No sirve si $y$ puede ser 0 |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | % de varianza explicada. **Maximizar.** Entre 0 y 1 |

## CLASIFICACIÓN

La **matriz de confusión** es la base de todo:

| | Predicho 0 | Predicho 1 |
|---|---|---|
| **Real 0** | TN (verdadero negativo) | FP (falso positivo) |
| **Real 1** | FN (falso negativo) | TP (verdadero positivo) |

| Métrica | Fórmula | Cuándo importa |
|---|---|---|
| **Accuracy** | $\frac{TP+TN}{\text{total}}$ | Métrica general. **Engaña si las clases están desbalanceadas** |
| **Sensibilidad / Recall** | $\frac{TP}{TP+FN}$ | De los que **sí** eran, ¿cuántos detecté? Clave en diagnóstico médico y fraude |
| **Especificidad** | $\frac{TN}{TN+FP}$ | De los que **no** eran, ¿cuántos descarté bien? |
| **Precisión** | $\frac{TP}{TP+FP}$ | De los que **predije** positivos, ¿cuántos acerté? |
| **F1** | media armónica de precisión y recall | Balance entre las dos |
| **AUC-ROC** | área bajo la curva ROC | Calidad global sin fijar umbral. 0.5 = azar, 1 = perfecto |

> ⚠️ **La trampa del accuracy:** si el 99% de los datos son clase 0, un modelo que siempre dice "0" tiene 99% de accuracy y es inútil. Mira siempre la matriz de confusión.

---

## Archivos de este tema

- **`python/A04_metricas_de_evaluacion.py`** — código listo para correr. Cambia el bloque ✏️ por tu dataset.
- **`R/A04_metricas_de_evaluacion.R`** — el equivalente en R.
- Las gráficas se guardan en `figuras/` al ejecutar.

## Cómo correrlo

```bash
cd temas/A04_metricas_de_evaluacion/python
python A04_metricas_de_evaluacion.py
```

Para volver al índice general: [`../../README.md`](../../README.md) ·
Para el cheatsheet completo: [`../../cheatsheet_python.ipynb`](../../cheatsheet_python.ipynb)
