# ############################################################################
# A7. REGRESIÓN LINEAL
# Tema A07_regresion_lineal — Introducción al Machine Learning (IIND3118)
#
# CÓMO USARLO
#   1. Cambia el bloque marcado con ✏️ por tu dataset.
#   2. Corre:  Rscript A07_regresion_lineal.R     (o ábrelo en RStudio)
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
cat("A7. REGRESIÓN LINEAL\n")
cat(strrep("=",74), "\n")
cat(sprintf("datos: %d obs. x %d predictores\n\n", nrow(X), ncol(X)))

# ############################################################################
#  CÓDIGO DEL MÉTODO
# ############################################################################

# ============================================================================
# A7. REGRESIÓN LINEAL
# ----------------------------------------------------------------------------
# QUÉ HACE: busca la recta (o plano) que mejor pasa por los datos,
# minimizando la suma de errores al cuadrado.
#
#   beta = argmin suma( (Y_i - beta'X_i)^2 )
#
# CÓMO SE INTERPRETA (esto siempre lo preguntan)
#   "Si X_j aumenta UNA UNIDAD, el valor esperado de Y cambia en beta_j
#    unidades, MANTENIENDO TODAS LAS DEMÁS VARIABLES CONSTANTES."
#
# ✅ El más interpretable · rápido · da valores p e intervalos
# ❌ Solo relaciones LINEALES · sensible a outliers · sufre con
#    MULTICOLINEALIDAD (variables muy correlacionadas -> coeficientes
#    inestables y enormes) -> eso es justo lo que arreglan Ridge/Lasso
#
# SUPUESTOS: linealidad, errores independientes, varianza constante, normalidad
# ============================================================================

if (!ES_CLASIFICACION) {
  dtr <- data.frame(Xtrain, ytrain=ytrain)
  reg <- lm(ytrain ~ ., data=dtr)
  print(summary(reg))
  cat("\nP>|t| < 0.05 -> la variable es estadísticamente significativa.\n")

  pred_lm <- predict(reg, data.frame(Xtest))
  mse_lm  <- mean((ytest - pred_lm)^2)
  cat(sprintf("MSE test: %.3f\n", mse_lm))

  # GRÁFICA de coeficientes
  co <- coef(reg)[-1]
  barplot(sort(co), horiz=TRUE, las=1, cex.names=.7,
          col=ifelse(sort(co)>0,"steelblue","indianred"),
          main="Coeficientes de la regresión lineal", xlab="coeficiente")
  abline(v=0)

  # MULTICOLINEALIDAD: VIF > 10 es problema serio
  if (requireNamespace("car", quietly=TRUE) && ncol(Xtrain) > 1) {
    v <- car::vif(reg)
    print(sort(v, decreasing=TRUE))
    cat("VIF > 10 -> multicolinealidad seria. Usa Ridge/Lasso (A9).\n")
  }
}

cat("\nListo. Revisa la carpeta figuras/\n")
