---
# B12. SVD (Singular Value Decomposition)

## Qué hace, en simple
Descompone **cualquier** matriz (no tiene que ser cuadrada ni simétrica) en tres piezas:
$$A = U\Sigma V^\top$$

Los **valores singulares** $\sigma_1\ge\sigma_2\ge\dots$ dicen cuánta "información" hay en cada dirección. Si te quedas solo con los primeros $k$, obtienes una **versión comprimida** de la matriz.

## Eckart-Young (el teorema clave)
> Truncar a los primeros $k$ valores singulares da la **MEJOR aproximación posible de rango $k$**. Ninguna otra matriz de rango $k$ se acerca más.

Y el error se sabe **en fórmula cerrada**:
$$\|A-A_k\|_F=\sqrt{\sum_{i>k}\sigma_i^2}$$
→ los valores singulares descartados te dicen **exactamente** cuánto pierdes.

## Relación con PCA
> **PCA = SVD sobre los datos centrados.** $V$ son los componentes, $U\Sigma$ los scores, $\lambda_i=\sigma_i^2/(n-1)$.

Por eso `prcomp` y `sklearn.PCA` usan SVD: es **numéricamente más estable** que formar $X^\top X$.

## Aplicación: LSI (recuperación de información)
Matriz término-documento → SVD → espacio **semántico latente**. Resuelve la **sinonimia**: buscar "petroleum" recupera documentos que solo dicen "crude". Ranking por **similitud coseno** (no euclidiana, porque los documentos largos tendrían ventaja injusta).

## Parámetros
| Parámetro | Qué poner |
|---|---|
| `full_matrices` | **`False`** casi siempre (versión económica) |
| $k$ | Usa la **energía acumulada** $\sum_{i\le k}\sigma_i^2/\sum\sigma_i^2$ |

> ⚠️ `np.linalg.svd` devuelve **$V^\top$**; `svd()` de R devuelve $V$. Al traducir, transponer.

---

## Archivos de este tema

- **`python/B12_svd.py`** — código listo para correr. Cambia el bloque ✏️ por tu dataset.
- **`R/B12_svd.R`** — el equivalente en R.
- Las gráficas se guardan en `figuras/` al ejecutar.

## Cómo correrlo

```bash
cd temas/B12_svd/python
python B12_svd.py
```

Para volver al índice general: [`../../README.md`](../../README.md) ·
Para el cheatsheet completo: [`../../cheatsheet_python.ipynb`](../../cheatsheet_python.ipynb)
