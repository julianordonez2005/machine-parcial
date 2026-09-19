# ############################################################################
# B8. PCA
# Tema B08_pca — Introducción al Machine Learning (IIND3118)
#
# CÓMO USARLO
#   1. Cambia el bloque marcado con ✏️ por tu dataset.
#   2. Corre:  Rscript B08_pca.R     (o ábrelo en RStudio)
#   3. Las gráficas se guardan en la carpeta figuras/ de este tema.
#
# La explicación del método está en el README.md de esta misma carpeta.
# ############################################################################

for (p in c("MASS","cluster","dbscan","glmnet","leaps","caret","class","fpc",
            "clusterCrit","isotree","kernlab","Rtsne","psych","car","vegan")) {
  if (!requireNamespace(p, quietly=TRUE))
    try(install.packages(p, repos="https://cloud.r-project.org"), silent=TRUE)
}
suppressMessages({ library(MASS); library(cluster) })

set.seed(0)

# Las gráficas se guardan en figuras/ en vez de abrirse en pantalla
dir.create("../figuras", showWarnings=FALSE)
.n <- 0
guardar <- function() {
  .n <<- .n + 1
  dev.copy(png, sprintf("../figuras/%02d_figura.png", .n), width=900, height=650)
  invisible(dev.off())
}

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  ✏️  CAMBIA SOLO ESTE BLOQUE POR TU DATASET                           ║
# ╚═══════════════════════════════════════════════════════════════════════╝
DATOS  <- "../../../data/ejemplo_regresion.csv"
TARGET <- "target"        # nombre EXACTO de la columna respuesta. NULL si no hay.
# ╚═══════════════════════════════════════════════════════════════════════╝

df <- na.omit(read.csv(DATOS))

if (!is.null(TARGET)) {
  y_raw <- df[[TARGET]]
  X_df  <- df[, setdiff(names(df), TARGET), drop=FALSE]
} else { y_raw <- NULL; X_df <- df }

X_df <- as.data.frame(model.matrix(~ . - 1, data=X_df))
nombres_X <- names(X_df)
X  <- as.matrix(X_df)
Xs <- scale(X)

if (!is.null(y_raw)) {
  ES_CLASIFICACION <- is.factor(y_raw) || is.character(y_raw) || length(unique(y_raw)) <= 10
  y <- if (ES_CLASIFICACION) as.factor(y_raw) else as.numeric(y_raw)
  idx <- sample(seq_len(nrow(X)), floor(0.70*nrow(X)))
  Xtrain <- X[idx,]; Xtest <- X[-idx,]; ytrain <- y[idx]; ytest <- y[-idx]
  Xs_train <- Xs[idx,]; Xs_test <- Xs[-idx,]
} else { y <- NULL; ES_CLASIFICACION <- FALSE }

pca  <- prcomp(X, center=TRUE, scale.=TRUE)
Z2   <- pca$x[, 1:2]
col_pt <- if (!is.null(y)) as.integer(as.factor(y)) else "steelblue"

cat(strrep("=",74), "\n")
cat("B8. PCA\n")
cat(strrep("=",74), "\n")
cat(sprintf("datos: %d obs. x %d predictores\n\n", nrow(X), ncol(X)))

# ############################################################################
#  CÓDIGO DEL MÉTODO
# ############################################################################

# ============================================================================
# B8. PCA
# ----------------------------------------------------------------------------
# QUÉ HACE: tienes muchas variables correlacionadas (redundantes). PCA las
# RESUME en unas pocas variables nuevas (los componentes), que son
# COMBINACIONES LINEALES de las originales y NO están correlacionadas entre
# sí. El primer componente es LA DIRECCIÓN DONDE LOS DATOS MÁS SE DISPERSAN.
#
#   S = W L W'    (descomposición espectral de la covarianza)
#     w_j = autovector j = componente j (los LOADINGS)
#     l_j = autovalor    = varianza en esa dirección
#   Scores (coordenadas nuevas): Z = (X - media) W
#   Varianza explicada del componente j: l_j / suma(l)
#
# LOS COEFICIENTES DE CADA COMPONENTE están en pca$rotation:
#   una COLUMNA por componente, una FILA por variable. Norma 1 cada columna.
#   ⚠️ EL SIGNO ES ARBITRARIO (-w también es autovector). R y Python pueden
#      dar signos opuestos: NO es un error.
#
# CÓMO ELEGIR CUÁNTOS COMPONENTES
#   Scree plot (codo) | varianza acumulada (~90%) | Kaiser (l_j > 1)
#
# PARÁMETROS de prcomp()
#   center = TRUE   siempre
#   scale. = TRUE   usa la matriz de CORRELACIÓN (equivale a estandarizar)
#            FALSE  usa la de COVARIANZA
#            ⚠️ PON TRUE si las variables están en unidades distintas
#   prcomp (SVD, divisor n-1) es MÁS ESTABLE que princomp (espectral, div. n)
#
# ✅ Óptimo entre proyecciones lineales · solución cerrada ·
#    elimina multicolinealidad · proyecta datos nuevos
# ❌ SOLO ESTRUCTURA LINEAL · componentes difíciles de interpretar
#    (mezclan todas las variables) · sensible a escala y outliers ·
#    MAXIMIZAR VARIANZA NO ES LO MISMO QUE MAXIMIZAR INFORMACIÓN ÚTIL
# ============================================================================

pca <- prcomp(X, center=TRUE, scale.=TRUE)     # ⬅ scale.=TRUE casi siempre
print(summary(pca))

varexp <- pca$sdev^2 / sum(pca$sdev^2)
cat(sprintf("\nComponentes para el 90%% de varianza: %d\n",
            which(cumsum(varexp) >= .90)[1]))
cat(sprintf("Criterio de Kaiser (lambda>1): %d componentes\n", sum(pca$sdev^2 > 1)))

cat("\nLOADINGS (coeficientes de cada componente):\n")
print(round(pca$rotation[, 1:min(4, ncol(pca$rotation))], 3))
cat("Lee cada COLUMNA: qué variables pesan más en ese componente.\n")

par(mfrow=c(1,3))
plot(pca$sdev^2, type="b", pch=16, xlab="componente", ylab="varianza (lambda)",
     main="Scree plot (codo)")
abline(h=1, col="red", lty=2)
plot(cumsum(varexp), type="b", pch=16, xlab="nº componentes",
     ylab="varianza acumulada", main="Varianza acumulada")
abline(h=.90, col="red", lty=2)
plot(pca$x[,1], pca$x[,2], pch=16, col=if(!is.null(y)) as.integer(as.factor(y)) else "steelblue",
     xlab=sprintf("PC1 (%.1f%%)", 100*varexp[1]),
     ylab=sprintf("PC2 (%.1f%%)", 100*varexp[2]), main="Datos en 2D")
par(mfrow=c(1,1))

biplot(pca, main="Biplot: observaciones + variables")

cat("\nListo. Revisa la carpeta figuras/\n")
