# ############################################################################
# A3. KNN Y KERNEL
# Tema A03_knn_y_kernel_regression — Introducción al Machine Learning (IIND3118)
#
# CÓMO USARLO
#   1. Cambia el bloque marcado con ✏️ por tu dataset.
#   2. Corre:  Rscript A03_knn_y_kernel_regression.R     (o ábrelo en RStudio)
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
cat("A3. KNN Y KERNEL\n")
cat(strrep("=",74), "\n")
cat(sprintf("datos: %d obs. x %d predictores\n\n", nrow(X), ncol(X)))

# ############################################################################
#  CÓDIGO DEL MÉTODO
# ############################################################################

# ============================================================================
# A3. KNN Y KERNEL REGRESSION
# ----------------------------------------------------------------------------
# KNN, EN SIMPLE: para predecir un punto nuevo, busca los k puntos más
# parecidos y devuelve el PROMEDIO de sus y (regresión) o la clase más
# VOTADA (clasificación). No aprende ninguna fórmula: se guarda los datos.
#
# KERNEL REGRESSION: igual, pero en vez de "los k más cercanos" usa todos
# los que caigan dentro de una VENTANA de ancho h.
#   f(X*) = promedio de los Y_i cuyos X_i cumplen |X_i - X*| <= h
#
# ⚠️ HAY QUE ESTANDARIZAR (usa Xs): ambos usan distancias. Si una variable
#    está en millones y otra en decenas, la primera domina todo.
#
# PARÁMETROS
#   k    : el que se calibra. k PEQUEÑO = flexible. Prueba 1 a 40.
#   h    : ancho de ventana. h PEQUEÑO = flexible.
# ============================================================================

# --- KNN: calibrar k --------------------------------------------------------
errores <- numeric(40)
for (k in 1:40) {
  if (ES_CLASIFICACION) {
    pr <- knn(Xs_train, Xs_test, ytrain, k=k)      # class::knn
    errores[k] <- mean(pr != ytest)                # error de clasificación
  } else {
    pr <- predict(knnreg(Xs_train, ytrain, k=k), Xs_test)
    errores[k] <- mean((ytest - pr)^2)             # MSE
  }
}
k_opt <- which.min(errores)

plot(1:40, errores, type="b", pch=16, xlab="k (número de vecinos)",
     ylab=ifelse(ES_CLASIFICACION,"error de clasificación","MSE test"),
     main="Calibración de k")
abline(v=k_opt, col="red", lty=2)
cat(sprintf("k óptimo = %d\n", k_opt))

# --- Modelo final -----------------------------------------------------------
if (ES_CLASIFICACION) {
  pred_knn <- knn(Xs_train, Xs_test, ytrain, k=k_opt)
  cat(sprintf("Accuracy: %.4f\n", mean(pred_knn == ytest)))
} else {
  pred_knn <- predict(knnreg(Xs_train, ytrain, k=k_opt), Xs_test)
  mse_knn  <- mean((ytest - pred_knn)^2)
  cat(sprintf("MSE test: %.3f\n", mse_knn))
}

# --- Kernel regression: implementación de clase ----------------------------
kernelreg <- function(xstar, X_, Y_, h) {
  # h GRANDE = rígido (sesgo). h PEQUEÑO = flexible (varianza).
  sapply(xstar, function(xs) {
    dentro <- abs(xs - X_) <= h
    if (sum(dentro) == 0) 0 else mean(Y_[dentro])
  })
}

if (!ES_CLASIFICACION) {
  j     <- which.max(abs(cor(X, y)))     # variable más correlacionada con y
  x1    <- Xs[, j]
  malla <- seq(min(x1), max(x1), length.out=200)

  plot(x1, y, pch=16, cex=.5, col="gray",
       xlab=paste(nombres_X[j],"(estandarizada)"), ylab=TARGET,
       main="Kernel regression: efecto de h")
  for (hh in c(0.05, 0.3, 1.5)) {
    lines(malla, kernelreg(malla, x1, y, hh), lwd=2,
          col=c("green","blue","red")[match(hh, c(0.05,0.3,1.5))])
  }
  legend("topright", c("h=0.05 (varianza)","h=0.3","h=1.5 (sesgo)"),
         col=c("green","blue","red"), lwd=2)
}

cat("\nListo. Revisa la carpeta figuras/\n")
