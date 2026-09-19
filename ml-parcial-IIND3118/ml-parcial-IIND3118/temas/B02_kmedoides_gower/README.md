---
# B2. K-Medoides (PAM) y distancia de Gower

## Qué hace, en simple
Igual que K-Means, pero el centro de cada grupo es **una observación real** (el *medoide*), no un promedio inventado.

$$\text{medoide}_k = \arg\min_{x_j\in C_k}\sum_{x_i\in C_k} d(x_i,x_j)$$

## Por qué importa el cambio
1. **Robusto a outliers**: un punto extremo puede arrastrar un promedio, pero no puede "sacar" al medoide de los datos.
2. **Solo necesita una matriz de distancias** → sirve con **variables categóricas o mixtas** usando **Gower**.

## Distancia de Gower, en simple
Compara dos observaciones variable por variable: si es numérica, la diferencia escalada por el rango; si es categórica, 0 si son iguales y 1 si son distintas. Luego **promedia**. Da un número entre 0 y 1.

## Parámetros
| Parámetro | Qué poner |
|---|---|
| `D` (matriz de distancias) | **La decisión más importante.** Euclidiana (numéricas) o Gower (mixtas) |
| `medoids` (K) | Número de grupos |

❌ **Costoso**: la matriz es $n\times n$. Con $n>10{,}000$ ya pesa. Alternativa: `CLARA`.

---

## Archivos de este tema

- **`python/B02_kmedoides_gower.py`** — código listo para correr. Cambia el bloque ✏️ por tu dataset.
- **`R/B02_kmedoides_gower.R`** — el equivalente en R.
- Las gráficas se guardan en `figuras/` al ejecutar.

## Cómo correrlo

```bash
cd temas/B02_kmedoides_gower/python
python B02_kmedoides_gower.py
```

Para volver al índice general: [`../../README.md`](../../README.md) ·
Para el cheatsheet completo: [`../../cheatsheet_python.ipynb`](../../cheatsheet_python.ipynb)
