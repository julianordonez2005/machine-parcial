---
# A2. Flexibilidad y el dilema sesgo–varianza

## Qué es, en simple

La **flexibilidad** es cuánto se puede "doblar" el modelo para seguir a los datos.

- **Poca flexibilidad** → el modelo es muy rígido, no captura la forma real. Se equivoca siempre igual. Esto es **SESGO** (*bias*). El modelo *subajusta* (**underfitting**).
- **Mucha flexibilidad** → el modelo persigue hasta el ruido. Con otra muestra daría algo muy distinto. Esto es **VARIANZA**. El modelo *sobreajusta* (**overfitting**).

## La descomposición

$$EPE = \underbrace{\sigma^2}_{\text{error irreducible}} + \underbrace{\text{Sesgo}^2}_{\downarrow \text{ con flexibilidad}} + \underbrace{\text{Varianza}}_{\uparrow \text{ con flexibilidad}}$$

El **error irreducible** $\sigma^2$ es el ruido de los datos: **ningún modelo lo puede eliminar**. Es el piso.

## La forma de las curvas (esto sale en el parcial)

- **MSE de TRAIN**: baja siempre al aumentar flexibilidad. Tiende a 0. **No sirve para calibrar.**
- **MSE de TEST**: tiene forma de **U**. Baja, toca un mínimo, y vuelve a subir.

**El punto óptimo de flexibilidad es el mínimo de la U del MSE de test.**

## Cómo se controla la flexibilidad en cada modelo

| Modelo | Parámetro | Más flexible cuando… |
|---|---|---|
| KNN | `n_neighbors` (k) | **k pequeño** |
| Kernel regression | `h` (ancho de ventana) | **h pequeño** |
| Regresión lineal | número de variables | **más variables** |
| Ridge / Lasso | `alpha` ($\lambda$) | **alpha pequeño** |
| Árbol | profundidad | **más profundo** |

> Ojo con la dirección: en KNN, kernel y Ridge/Lasso, **el parámetro grande = modelo rígido**.

---

## Archivos de este tema

- **`python/A02_flexibilidad_sesgo_varianza.py`** — código listo para correr. Cambia el bloque ✏️ por tu dataset.
- **`R/A02_flexibilidad_sesgo_varianza.R`** — el equivalente en R.
- Las gráficas se guardan en `figuras/` al ejecutar.

## Cómo correrlo

```bash
cd temas/A02_flexibilidad_sesgo_varianza/python
python A02_flexibilidad_sesgo_varianza.py
```

Para volver al índice general: [`../../README.md`](../../README.md) ·
Para el cheatsheet completo: [`../../cheatsheet_python.ipynb`](../../cheatsheet_python.ipynb)
