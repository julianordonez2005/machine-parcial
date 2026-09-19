---
# B10. Análisis Factorial

## Qué hace, en simple
PCA resume variables. El análisis factorial va más allá: supone que hay **causas ocultas (factores latentes)** que **producen** lo que observas.

Ejemplo clásico: nueve pruebas de habilidad mental. El modelo dice que detrás hay tres factores no observables —habilidad verbal, espacial y numérica— que causan los puntajes.

$$X = \Lambda F + \varepsilon$$
- $\Lambda$ = **cargas**: cuánto pesa cada factor sobre cada variable
- $\Psi$ = **unicidades**: la parte de cada variable que no comparte con nadie

$$\Sigma = \Lambda\Lambda^\top + \Psi$$

## La diferencia con PCA (esto lo preguntan)
| | PCA | Análisis Factorial |
|---|---|---|
| Dirección | Los componentes son **consecuencia** de las variables | Los factores son **causa** |
| Varianza | Modela **toda** | Solo la **compartida**, separa el ruido |
| Solución | Única, cerrada | Iterativa, **no única** |
| Objetivo | Comprimir | **Interpretar** |

## La rotación
La solución no es única: si $\Lambda$ sirve, $\Lambda R$ también. **Se aprovecha para hacer los factores interpretables.**
- **Varimax** (ortogonal): factores NO correlacionados. El default
- **Promax** (oblicua): permite factores correlacionados

**La rotación no cambia el ajuste, solo el sistema de ejes.**

## ⚠️ La prueba χ²
$H_0$: $m$ factores bastan.
> **p-valor ALTO = el modelo con $m$ factores SÍ es adecuado.** Es al revés de lo habitual.

**Límite de identificabilidad:** $m<\frac{2p+1-\sqrt{8p+1}}{2}$. Con $p=4$ el máximo es **1 factor** (por eso `factanal(iris,2)` da error: no es un bug).

## Verificación previa
- **Bartlett**: $H_0$ = no hay nada que factorizar. Se quiere **rechazar**
- **KMO**: >0.8 excelente · >0.6 aceptable · <0.5 inadecuado

---

## Archivos de este tema

- **`python/B10_analisis_factorial.py`** — código listo para correr. Cambia el bloque ✏️ por tu dataset.
- **`R/B10_analisis_factorial.R`** — el equivalente en R.
- Las gráficas se guardan en `figuras/` al ejecutar.

## Cómo correrlo

```bash
cd temas/B10_analisis_factorial/python
python B10_analisis_factorial.py
```

Para volver al índice general: [`../../README.md`](../../README.md) ·
Para el cheatsheet completo: [`../../cheatsheet_python.ipynb`](../../cheatsheet_python.ipynb)
