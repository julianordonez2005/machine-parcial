# Temas

Cada carpeta es independiente: tiene la explicación del método, el código en
Python y en R, y guarda sus gráficas al correr.

## Estructura de cada tema

```
B08_pca/
├── README.md        ← qué hace el método, fórmulas, parámetros, ventajas/desventajas
├── python/B08_pca.py
├── R/B08_pca.R
└── figuras/         ← se crea al ejecutar
```

## Cómo correr uno

```bash
cd B08_pca/python && python B08_pca.py
```

Antes de correrlo, abre el archivo y cambia el bloque marcado con ✏️ por tu
dataset. Es lo único que hay que tocar.

## Orden sugerido para estudiar

**Supervisado:** A01 → A02 → A04 → A06 → A07 → A09
(los conceptos base primero: evaluación, sesgo-varianza, métricas, CV)

**No supervisado:** B08 (PCA) → B01 (K-Means) → B05 (elegir K) → el resto
(PCA primero porque se usa para graficar en todos los demás)
