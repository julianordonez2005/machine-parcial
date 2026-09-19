---
# A1. Partición train / test y el MSE

## Qué hace, en simple

Partes los datos en dos: con una parte **entrenas** el modelo y con la otra **lo calificas**. Es como estudiar con unos ejercicios y presentar el examen con otros distintos: si te calificas con los mismos que estudiaste, la nota no dice nada sobre si aprendiste.

## Por qué es necesario

El error medido en los datos de entrenamiento **siempre baja** si haces el modelo más flexible. Un modelo que memoriza cada punto tiene error de entrenamiento cero y no sirve para nada. El error que importa es el de **test**, sobre datos que el modelo nunca vio.

$$MSE_{test} = \frac{1}{n_{test}}\sum_{i=1}^{n_{test}}(\hat y_i - y_i)^2$$

## Parámetros de `train_test_split`

| Parámetro | Qué poner |
|---|---|
| `test_size` | Fracción para test. **Entre 0.10 y 0.50**; lo típico es `0.30` o `0.33` |
| `random_state` | Cualquier número fijo (ej. `0`). **Ponlo siempre** para que el resultado se repita |
| `stratify=y` | **Solo en clasificación.** Mantiene la proporción de clases en ambas partes |
| `shuffle` | `True` por defecto. Ponlo en `False` solo con series de tiempo |

---

## Archivos de este tema

- **`python/A01_train_test_y_mse.py`** — código listo para correr. Cambia el bloque ✏️ por tu dataset.
- **`R/A01_train_test_y_mse.R`** — el equivalente en R.
- Las gráficas se guardan en `figuras/` al ejecutar.

## Cómo correrlo

```bash
cd temas/A01_train_test_y_mse/python
python A01_train_test_y_mse.py
```

Para volver al índice general: [`../../README.md`](../../README.md) ·
Para el cheatsheet completo: [`../../cheatsheet_python.ipynb`](../../cheatsheet_python.ipynb)
