# ############################################################################
# B7. OUTLIERS: LOF
# Tema B07_outliers_lof_isolationforest — Introducción al Machine Learning (IIND3118)
#
# CÓMO USARLO
#   1. Cambia el bloque marcado con ✏️ por tu dataset.
#   2. Corre:  Rscript B07_outliers_lof_isolationforest.R     (o ábrelo en RStudio)
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
cat("B7. OUTLIERS: LOF\n")
cat(strrep("=",74), "\n")
cat(sprintf("datos: %d obs. x %d predictores\n\n", nrow(X), ncol(X)))

# ############################################################################
#  CÓDIGO DEL MÉTODO
# ############################################################################

# ============================================================================
# B7. OUTLIERS: LOF E ISOLATION FOREST
# ----------------------------------------------------------------------------
# Estos dos atacan el problema desde ángulos OPUESTOS.
#
# LOF (Local Outlier Factor)
#   Compara la DENSIDAD de un punto con la de sus vecinos. Si vive en una
#   zona mucho más vacía que sus vecinos, es outlier.
#     LOF ~ 1  densidad normal
#     LOF >> 1 OUTLIER
#     LOF < 1  dentro de un cluster denso
#   Detecta outliers LOCALES: normales globalmente, raros en su vecindario.
#   ⚠️ En R (dbscan::lof) MAYOR = más atípico.
#      En Python (negative_outlier_factor_) es al revés.
#
# ISOLATION FOREST
#   Idea distinta: LOS PUNTOS RAROS SON FÁCILES DE AISLAR. Hace cortes
#   aleatorios; un outlier queda solo tras pocos cortes. Detecta outliers
#   GLOBALES. Es O(n log n) -> escala muy bien.
#   ⚠️ En R (isotree) MAYOR = más anómalo. En Python decision_function al revés.
#
# CUÁL USAR
#   n grande, muchas variables        -> ISOLATION FOREST
#   densidades muy distintas          -> LOF
#   pocas variables, datos ~normales  -> MAHALANOBIS (B6)
#   EN LA PRÁCTICA: corre varios y mira dónde COINCIDEN.
#
# PARÁMETROS
#   minPts (LOF)    : ⚠️ EL CRÍTICO. 10-35. Pequeño=ruidoso, grande=global
#   ntrees          : 100 basta
#   sample_size     : 256 funciona tan bien como usar todo
# ============================================================================

library(dbscan); library(isotree)

score_lof <- lof(Xs, minPts=20)                    # ⬅ MAYOR = más atípico
es_lof    <- score_lof > 1.5                       # umbral habitual (arbitrario)

iso      <- isolation.forest(as.data.frame(Xs), ntrees=100, sample_size=256,
                             nthreads=1, seed=0)
score_if <- predict(iso, as.data.frame(Xs))        # ⬅ MAYOR = más anómalo
es_if    <- score_if > quantile(score_if, 0.90)

cat(sprintf("outliers según LOF            : %d\n", sum(es_lof)))
cat(sprintf("outliers según Isolation Forest: %d\n", sum(es_if)))
cat(sprintf("\ncoinciden en             : %d\n", sum(es_lof & es_if)))
cat(sprintf("solo LOF (LOCALES)       : %d\n", sum(es_lof & !es_if)))
cat(sprintf("solo iForest (GLOBALES)  : %d\n", sum(!es_lof & es_if)))

par(mfrow=c(1,3))
plot(Z2, col=ifelse(es_lof,"red","gray60"), pch=16, xlab="PC1", ylab="PC2", main="LOF")
plot(Z2, col=ifelse(es_if,"red","gray60"),  pch=16, xlab="PC1", ylab="PC2", main="Isolation Forest")
plot(Z2, col="gray80", pch=16, xlab="PC1", ylab="PC2", main="Dónde coinciden")
points(Z2[es_lof & es_if, , drop=FALSE],  col="red",       pch=16, cex=1.4)
points(Z2[es_lof & !es_if, , drop=FALSE], col="blue",      pch=17, cex=1.4)
points(Z2[!es_lof & es_if, , drop=FALSE], col="darkgreen", pch=15, cex=1.4)
legend("topright", c("ambos","solo LOF","solo iForest"),
       col=c("red","blue","darkgreen"), pch=c(16,17,15), cex=.8)
par(mfrow=c(1,1))

cat("\nListo. Revisa la carpeta figuras/\n")
