---
# A7. Regresión lineal

## Qué hace, en simple

Busca la **recta (o plano) que mejor pasa por los datos**, minimizando la suma de los errores al cuadrado:

$$\hat\beta = \arg\min \sum_{i=1}^n (Y_i - \beta^t X_i)^2$$

## Cómo se interpretan los coeficientes (esto siempre lo preguntan)

> **Si $X_j$ aumenta una unidad, el valor esperado de $Y$ cambia en $\beta_j$ unidades, manteniendo todas las demás variables constantes.**

Esa última parte ("manteniendo las demás constantes") es lo que hace la interpretación *ceteris paribus* y es la gran ventaja del modelo frente a KNN o un árbol.

## Ventajas
- **El más interpretable de todos**: cada coeficiente tiene significado directo
- Rápido, solución cerrada, sin parámetros que calibrar
- Da errores estándar, valores p e intervalos de confianza
- Es el punto de partida obligado: si un modelo complejo no le gana, no vale la pena

## Desventajas
- **Solo captura relaciones lineales**
- **Sensible a outliers** (los errores van al cuadrado)
- **Sufre con multicolinealidad**: si dos variables están muy correlacionadas, los coeficientes se vuelven inestables y enormes → esto es justo lo que arreglan Ridge/Lasso
- Si $p > n$ no tiene solución única
- Poca flexibilidad: sesgo alto si la relación real es curva

## Supuestos (por si los piden)
Linealidad · errores independientes · varianza constante (homocedasticidad) · errores normales (solo para los valores p)

---

## Archivos de este tema

- **`python/A07_regresion_lineal.py`** — código listo para correr. Cambia el bloque ✏️ por tu dataset.
- **`R/A07_regresion_lineal.R`** — el equivalente en R.
- Las gráficas se guardan en `figuras/` al ejecutar.

## Cómo correrlo

```bash
cd temas/A07_regresion_lineal/python
python A07_regresion_lineal.py
```

Para volver al índice general: [`../../README.md`](../../README.md) ·
Para el cheatsheet completo: [`../../cheatsheet_python.ipynb`](../../cheatsheet_python.ipynb)
