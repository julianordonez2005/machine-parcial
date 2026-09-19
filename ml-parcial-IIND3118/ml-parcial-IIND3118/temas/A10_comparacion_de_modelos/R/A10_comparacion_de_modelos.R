# ############################################################################
# A10. COMPARACIÓN FINAL
# Tema A10_comparacion_de_modelos — Introducción al Machine Learning (IIND3118)
#
# CÓMO USARLO
#   1. Cambia el bloque marcado con ✏️ por tu dataset.
#   2. Corre:  Rscript A10_comparacion_de_modelos.R     (o ábrelo en RStudio)
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
cat("A10. COMPARACIÓN FINAL\n")
cat(strrep("=",74), "\n")
cat(sprintf("datos: %d obs. x %d predictores\n\n", nrow(X), ncol(X)))

# ############################################################################
#  CÓDIGO DEL MÉTODO
# ############################################################################

# ============================================================================
# A10. COMPARACIÓN FINAL DE MODELOS
# ----------------------------------------------------------------------------
# Siempre se reporta el error en TEST (nunca en train), y se menciona qué
# parámetro se calibró en cada modelo y cómo.
# ============================================================================

if (!ES_CLASIFICACION) {
  resultados <- c()

  # Lineal
  m1 <- lm(ytrain ~ ., data=data.frame(Xtrain, ytrain))
  resultados["Lineal (OLS)"] <- mean((ytest - predict(m1, data.frame(Xtest)))^2)

  # KNN calibrado
  resultados[paste0("KNN (k=",k_opt,")")] <-
    mean((ytest - predict(knnreg(Xs_train, ytrain, k=k_opt), Xs_test))^2)

  # Penalizados
  for (info in list(list(0,"Ridge"), list(1,"Lasso"), list(0.3,"Elastic Net"))) {
    cvm <- cv.glmnet(Xs_train, ytrain, alpha=info[[1]], nfolds=10)
    mm  <- glmnet(Xs_train, ytrain, alpha=info[[1]], lambda=cvm$lambda.min)
    resultados[info[[2]]] <- mean((ytest - as.numeric(predict(mm, Xs_test)))^2)
  }

  resultados <- sort(resultados)
  print(round(resultados, 3))
  cat(sprintf("\nMejor modelo: %s (MSE = %.3f)\n",
              names(resultados)[1], resultados[1]))

  barplot(rev(resultados), horiz=TRUE, las=1, cex.names=.8, col="steelblue",
          xlab="MSE en test (menor = mejor)", main="Comparación de modelos")
}



# ############################################################################
# PARTE B — APRENDIZAJE NO SUPERVISADO
# ############################################################################
# Aquí NO se usa y. Todo trabaja con Xs (estandarizado).

# Proyección PCA en 2D: se usa para graficar TODOS los métodos de esta parte
pca_plot <- prcomp(Xs, center=TRUE, scale.=FALSE)
Z2 <- pca_plot$x[, 1:2]

cat("\nListo. Revisa la carpeta figuras/\n")
