---
# B8. PCA (Análisis de Componentes Principales)

## Qué hace, en simple
Tienes muchas variables correlacionadas entre sí (redundantes). PCA las **resume en unas pocas variables nuevas** (los componentes), que son **combinaciones lineales** de las originales y no están correlacionadas entre sí. El primer componente es **la dirección donde los datos más se dispersan**.

## Fórmula
$$\max_{\|w\|=1}\operatorname{Var}(Xw)=w^\top S w \quad\Rightarrow\quad S=W\Lambda W^\top$$
- $w_j$ = autovector $j$ = componente $j$ (los **loadings**)
- $\lambda_j$ = autovalor = varianza en esa dirección
- **Scores** (coordenadas nuevas): $Z=(X-\bar X)W$
- Varianza explicada: $\lambda_j/\sum\lambda_i$

## Los coeficientes de cada componente
Están en **`pca.components_`**: una **fila por componente**, una **columna por variable**. Norma 1 cada fila.
> **El signo es arbitrario**: $-w$ también es autovector. R y Python pueden dar signos opuestos; no es un error.

## Cómo elegir cuántos componentes
- **Scree plot (codo)**: grafica $\lambda_j$ y busca dónde se aplana
- **Varianza acumulada**: quedarte con ~90%
- **Kaiser**: sobre datos estandarizados, los de $\lambda_j>1$

## Parámetros
| Parámetro | Qué poner |
|---|---|
| `n_components` | Un entero, o un **float en (0,1)** para pedir esa fracción de varianza (ej. `0.95`) |
| **Estandarizar antes** | **Crítico.** sklearn centra pero NO estandariza. Usar `Xs` = trabajar con la matriz de correlación |

✅ Óptimo entre proyecciones lineales · solución cerrada · elimina multicolinealidad · proyecta datos nuevos
❌ **Solo estructura lineal** · **componentes difíciles de interpretar** (mezclan todas las variables) · sensible a escala y outliers · **maximizar varianza ≠ maximizar información útil**

---

## Archivos de este tema

- **`python/B08_pca.py`** — código listo para correr. Cambia el bloque ✏️ por tu dataset.
- **`R/B08_pca.R`** — el equivalente en R.
- Las gráficas se guardan en `figuras/` al ejecutar.

## Cómo correrlo

```bash
cd temas/B08_pca/python
python B08_pca.py
```

Para volver al índice general: [`../../README.md`](../../README.md) ·
Para el cheatsheet completo: [`../../cheatsheet_python.ipynb`](../../cheatsheet_python.ipynb)
