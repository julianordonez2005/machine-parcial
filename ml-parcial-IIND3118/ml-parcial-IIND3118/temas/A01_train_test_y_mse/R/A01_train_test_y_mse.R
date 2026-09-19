# ############################################################################
# A1. PARTICIÓN
# Tema A01_train_test_y_mse — Introducción al Machine Learning (IIND3118)
#
# CÓMO USARLO
#   1. Cambia el bloque marcado con ✏️ por tu dataset.
#   2. Corre:  Rscript A01_train_test_y_mse.R     (o ábrelo en RStudio)
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
cat("A1. PARTICIÓN\n")
cat(strrep("=",74), "\n")
cat(sprintf("datos: %d obs. x %d predictores\n\n", nrow(X), ncol(X)))

# ############################################################################
#  CÓDIGO DEL MÉTODO
# ############################################################################

# ============================================================================
# A1. PARTICIÓN TRAIN/TEST Y MSE
# ----------------------------------------------------------------------------
# QUÉ HACE: parte los datos en dos. Con una parte entrenas, con la otra
# calificas. Es como estudiar con unos ejercicios y presentar el examen con
# otros: si te calificas con los mismos que estudiaste, la nota no dice nada.
#
# POR QUÉ: el error en TRAIN siempre baja si haces el modelo más flexible.
# Un modelo que memoriza tiene error de train = 0 y no sirve. El que importa
# es el MSE de TEST.
#
#   MSE_test = (1/n_test) * suma( (y_pred - y_real)^2 )
#
# PARÁMETROS
#   proporción train : entre 0.50 y 0.90; lo típico es 0.70 o 0.67
#   set.seed(n)      : PONLO SIEMPRE, para que el resultado se repita
# ============================================================================

# La partición ya está hecha arriba. Manualmente sería:
set.seed(0)
idx2   <- sample(seq_len(nrow(X)), size = floor(0.70*nrow(X)))   # 70% train
Xtr <- X[idx2,]; Xte <- X[-idx2,]; ytr <- y[idx2]; yte <- y[-idx2]

# Cómo calcular el MSE de cualquier modelo:
modelo <- lm(ytrain ~ ., data = data.frame(Xtrain, ytrain=ytrain))  # 1. entrenar
pred   <- predict(modelo, newdata = data.frame(Xtest))              # 2. predecir
mse    <- mean((ytest - pred)^2)                                    # 3. comparar

cat(sprintf("MSE en test : %.3f\n", mse))
cat(sprintf("RMSE en test: %.3f  (en las unidades de y)\n", sqrt(mse)))

# GRÁFICA estándar: predicho vs. real
plot(ytest, pred, pch=16, col=rgb(0,0,1,.5),
     xlab="y real (test)", ylab="y predicho",
     main=sprintf("Predicho vs. real — RMSE = %.2f", sqrt(mse)))
abline(0, 1, col="red", lwd=2, lty=2)
# Entre más pegados los puntos a la línea roja, mejor el modelo.

cat("\nListo. Revisa la carpeta figuras/\n")
