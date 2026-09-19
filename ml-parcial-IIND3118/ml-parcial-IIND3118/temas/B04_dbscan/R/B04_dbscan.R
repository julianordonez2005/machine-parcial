# ############################################################################
# B4. DBSCAN
# Tema B04_dbscan — Introducción al Machine Learning (IIND3118)
#
# CÓMO USARLO
#   1. Cambia el bloque marcado con ✏️ por tu dataset.
#   2. Corre:  Rscript B04_dbscan.R     (o ábrelo en RStudio)
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
cat("B4. DBSCAN\n")
cat(strrep("=",74), "\n")
cat(sprintf("datos: %d obs. x %d predictores\n\n", nrow(X), ncol(X)))

# ############################################################################
#  CÓDIGO DEL MÉTODO
# ############################################################################

# ============================================================================
# B4. DBSCAN
# ----------------------------------------------------------------------------
# QUÉ HACE: busca ZONAS DENSAS. Un cluster es un montón de puntos pegados.
# No exige que todos estén cerca del centro, solo que haya una CADENA de
# vecinos densos. Por eso encuentra FORMAS RARAS (espirales, anillos).
# Además marca como RUIDO los puntos que no caen en ninguna zona densa.
# Es el ÚNICO método de clustering que dice "este punto no va en ningún lado".
#
# LOS 3 TIPOS DE PUNTO
#   core   : tiene al menos minPts vecinos dentro del radio eps
#   border : no es core, pero está dentro del radio de un core
#   noise  : ninguna de las dos -> etiqueta 0 EN R  (⚠️ en Python es -1)
#
# CÓMO ELEGIR eps: LA CURVA k-dist
#   1. distancia de cada punto a su k-ésimo vecino
#   2. ordenar y graficar  ->  kNNdistplot()
#   3. el CODO de la curva es un buen eps
#
# PARÁMETROS
#   eps    : ⚠️ EL CRÍTICO. Radio de la vecindad. Se lee del codo.
#   minPts : regla práctica >= p+1, o 2p si hay ruido
#
# DIAGNÓSTICO: casi todo es ruido -> eps muy chico.
#              un solo cluster gigante -> eps muy grande.
#
# ✅ Formas arbitrarias · no fija K · detecta outliers
# ❌ Calibración difícil y MUY sensible ·
#    FALLA si los clusters tienen densidades distintas (usa hdbscan) ·
#    se degrada en dimensión alta
# ============================================================================

library(dbscan)
k_vec <- min(5, nrow(Xs)-1)

# Paso 1: la curva k-dist
kNNdistplot(Xs, k=k_vec)
title("Curva k-dist: busca el CODO para elegir eps")
d_k <- sort(as.vector(kNNdist(Xs, k=k_vec)))
abline(h=quantile(d_k, .90), col="red", lty=2)
cat(sprintf("Sugerencia: eps entre %.2f (p80) y %.2f (p95)\n",
            quantile(d_k,.80), quantile(d_k,.95)))

# Paso 2: probar varios eps
cat("\nSensibilidad a eps:\n")
for (eps in round(quantile(d_k, c(.5,.7,.8,.9,.95)), 2)) {
  lab <- dbscan(Xs, eps=eps, minPts=k_vec+1)$cluster
  cat(sprintf("  eps=%5.2f -> %d clusters, %d puntos de ruido\n",
              eps, length(setdiff(unique(lab),0)), sum(lab==0)))
}

# Paso 3: modelo final
EPS <- as.numeric(quantile(d_k, .90))     # ⬅ ajusta según la curva
db  <- dbscan(Xs, eps=EPS, minPts=k_vec+1)
cat(sprintf("\nFINAL: eps=%.2f -> %d clusters, %d ruido\n",
            EPS, length(setdiff(unique(db$cluster),0)), sum(db$cluster==0)))

plot(Z2, col=ifelse(db$cluster==0, "red", db$cluster+1),
     pch=ifelse(db$cluster==0, 4, 16),
     xlab="PC1", ylab="PC2", main=sprintf("DBSCAN (eps=%.2f)", EPS))
legend("topright", "ruido (0)", col="red", pch=4)

# Si las densidades son muy distintas: hdbscan(Xs, minPts=5)

cat("\nListo. Revisa la carpeta figuras/\n")
