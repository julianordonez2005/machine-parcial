---
# B11. Kernel PCA

## Qué hace, en simple
PCA solo encuentra estructura **recta**. Si los datos están sobre una curva o en círculos concéntricos, ninguna proyección lineal los separa.

KPCA los manda a un espacio de más dimensiones donde **sí** son separables linealmente, hace PCA ahí, y regresa. Lo notable: **nunca construye ese espacio**, usa el **truco del kernel** ($K(x,y)=\langle\phi(x),\phi(y)\rangle$).

## Kernels
| Kernel | Cuándo |
|---|---|
| `'linear'` | Reproduce PCA exactamente |
| `'poly'` | Interacciones entre variables |
| **`'rbf'`** | El más usado. $\exp(-\gamma\|x-y\|^2)$ |

## ⚠️ `gamma` lo decide TODO
- **muy grande** → cada punto solo se parece a sí mismo → **no hay estructura**
- **muy pequeño** → todos se parecen → **converge a PCA lineal**
- **Heurística de la mediana:** $\gamma\approx 1/\text{mediana}(\|x_i-x_j\|^2)$

> **Convención:** `kernlab` (R) usa `sigma`, sklearn usa `gamma` — **son lo mismo**. Pero otras librerías usan $\exp(-\|x-y\|^2/2\sigma^2)$, que NO lo es.

✅ Captura no linealidad · **sí proyecta datos nuevos** (MDS y t-SNE no)
❌ $O(n^3)$ · **componentes sin interpretación** · sin reconstrucción directa (*problema de la preimagen*) · **es fácil "encontrar" estructura ajustando gamma hasta que el gráfico se vea bien**

> **Regla:** verifica siempre contra PCA lineal primero. Si PCA ya resuelve, KPCA solo agrega opacidad.

---

## Archivos de este tema

- **`python/B11_kernel_pca.py`** — código listo para correr. Cambia el bloque ✏️ por tu dataset.
- **`R/B11_kernel_pca.R`** — el equivalente en R.
- Las gráficas se guardan en `figuras/` al ejecutar.

## Cómo correrlo

```bash
cd temas/B11_kernel_pca/python
python B11_kernel_pca.py
```

Para volver al índice general: [`../../README.md`](../../README.md) ·
Para el cheatsheet completo: [`../../cheatsheet_python.ipynb`](../../cheatsheet_python.ipynb)
