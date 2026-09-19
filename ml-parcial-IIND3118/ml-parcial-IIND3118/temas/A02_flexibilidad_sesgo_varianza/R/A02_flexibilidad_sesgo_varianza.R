# ############################################################################
# A2. FLEXIBILIDAD
# Tema A02_flexibilidad_sesgo_varianza — Introducción al Machine Learning (IIND3118)
#
# CÓMO USARLO
#   1. Cambia el bloque marcado con ✏️ por tu dataset.
#   2. Corre:  Rscript A02_flexibilidad_sesgo_varianza.R     (o ábrelo en RStudio)
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
cat("A2. FLEXIBILIDAD\n")
cat(strrep("=",74), "\n")
cat(sprintf("datos: %d obs. x %d predictores\n\n", nrow(X), ncol(X)))

# ############################################################################
#  CÓDIGO DEL MÉTODO
# ############################################################################

# ============================================================================
# A2. FLEXIBILIDAD Y SESGO-VARIANZA
# ----------------------------------------------------------------------------
# QUÉ ES: la flexibilidad es cuánto se puede "doblar" el modelo.
#   POCA  -> modelo rígido, no captura la forma real  -> SESGO (underfitting)
#   MUCHA -> persigue hasta el ruido                  -> VARIANZA (overfitting)
#
#   EPE = error_irreducible + Sesgo^2 + Varianza
#         (el irreducible es el ruido: NINGÚN modelo lo elimina)
#
# LAS CURVAS (esto sale en el parcial)
#   MSE de TRAIN: baja SIEMPRE. No sirve para calibrar.
#   MSE de TEST : forma de U. El óptimo es el MÍNIMO de la U.
#
# CÓMO SE CONTROLA LA FLEXIBILIDAD
#   KNN            : k       -> MÁS flexible con k PEQUEÑO
#   kernel regr.   : h       -> MÁS flexible con h PEQUEÑO
#   regresión      : nº vars -> más flexible con MÁS variables
#   ridge/lasso    : lambda  -> MÁS flexible con lambda PEQUEÑO
# ============================================================================

library(class); library(caret)

ks <- 1:40
mse_tr <- numeric(length(ks)); mse_te <- numeric(length(ks))

for (i in seq_along(ks)) {
  m_tr <- knnreg(Xs_train, ytrain, k = ks[i])
  mse_tr[i] <- mean((ytrain - predict(m_tr, Xs_train))^2)
  mse_te[i] <- mean((ytest  - predict(m_tr, Xs_test ))^2)
}
k_opt <- ks[which.min(mse_te)]

# Eje x invertido: izquierda = rígido, derecha = flexible (como en clase)
plot(ks, mse_tr, type="l", col="red", lwd=2, xlim=rev(range(ks)),
     ylim=range(c(mse_tr,mse_te)),
     xlab="k  (<- menos flexible  |  más flexible ->)", ylab="MSE",
     main="Sesgo-varianza: la curva en U")
lines(ks, mse_te, col="blue", lwd=2)
abline(v=k_opt, lty=2)
legend("topright", c("MSE TRAIN (siempre baja)","MSE TEST (forma de U)"),
       col=c("red","blue"), lwd=2)
cat(sprintf("k óptimo = %d  (MSE test = %.3f)\n", k_opt, min(mse_te)))

cat("\nListo. Revisa la carpeta figuras/\n")
