# Plantillas — código mínimo para copiar y pegar

Versiones comprimidas del cheatsheet, para cuando ya sabes qué necesitas y
solo quieres el código corriendo rápido.

| Archivo | Para qué |
|---|---|
| `regresion.py` / `regresion.R` | Compara lineal, KNN, Ridge, Lasso, Elastic Net y forward. Devuelve la tabla ordenada por MSE |
| `clasificacion.py` | Logística y KNN, con matriz de confusión, sensibilidad, especificidad y curva ROC |
| `no_supervisado.py` | PCA + elección de K por Silhouette + K-Means, con las tres gráficas |

En los tres, cambia solo las dos líneas marcadas con ✏️:

```python
df     = pd.read_csv("data/mis_datos.csv")
TARGET = "nombre_de_la_columna_y"
```

Si necesitas el detalle de un método (parámetros, cuándo usarlo, ventajas y
desventajas), ve al cheatsheet completo: `../cheatsheet_python.ipynb`.

Los tres archivos de Python se ejecutaron y corren sin errores. El de R no se
pudo probar (no había R en el entorno de generación).
