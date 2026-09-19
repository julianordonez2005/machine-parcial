# ############################################################################
# A6. VALIDACIÓN CRUZADA
# Tema A06_validacion_cruzada — Introducción al Machine Learning (IIND3118)
#
# CÓMO USARLO
#   1. Cambia el bloque marcado con ✏️ por tu dataset.
#   2. Corre:  Rscript A06_validacion_cruzada.R     (o ábrelo en RStudio)
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
cat("A6. VALIDACIÓN CRUZADA\n")
cat(strrep("=",74), "\n")
cat(sprintf("datos: %d obs. x %d predictores\n\n", nrow(X), ncol(X)))

# ############################################################################
#  CÓDIGO DEL MÉTODO
# ############################################################################

# ============================================================================
# A6. VALIDACIÓN CRUZADA (k-fold)
# ----------------------------------------------------------------------------
# QUÉ HACE: partir train/test una sola vez depende del azar. La validación
# cruzada parte los datos en k PEDAZOS (folds), entrena k veces dejando un
# pedazo distinto fuera cada vez, y PROMEDIA los k errores.
# Así TODOS los datos sirven para entrenar y para evaluar, en rondas distintas.
#
#   Ronda 1: [TEST ][train][train][train][train]
#   Ronda 2: [train][TEST ][train][train][train]   ...
#            -> MSE_CV = promedio de los k MSE
#
# PARÁMETROS
#   k (number)  : 5 o 10 son los estándar. k grande = menos sesgo, más lento
#   LOOCV       : k = n. Sin sesgo pero lento y con alta varianza
#
# ✅ Usa todos los datos, resultado estable
# ❌ k veces más lento; NO sirve tal cual con series de tiempo
# ============================================================================

library(caret)

ctrl <- trainControl(method="cv", number=10)     # ⬅ number = nº de folds

modelo_cv <- train(
  x = Xs, y = y,
  method = "knn",                                # ⬅ "knn","lm","glmnet","rf"...
  trControl = ctrl,
  tuneGrid = data.frame(k = 1:40),               # ⬅ valores del parámetro a probar
  metric = ifelse(ES_CLASIFICACION, "Accuracy", "RMSE")
)
print(modelo_cv)
plot(modelo_cv, main="Calibración de k con 10-fold CV")
cat(sprintf("Mejor k por CV: %d\n", modelo_cv$bestTune$k))

# Bucle manual, por si piden mostrar el procedimiento:
set.seed(0)
folds <- sample(rep(1:10, length.out=nrow(Xs)))
err <- numeric(10)
for (f in 1:10) {
  m <- knnreg(Xs[folds!=f, ], y[folds!=f], k=5)
  err[f] <- mean((y[folds==f] - predict(m, Xs[folds==f, ]))^2)
}
cat(sprintf("MSE promedio (10-fold, manual): %.3f ± %.3f\n", mean(err), sd(err)))

cat("\nListo. Revisa la carpeta figuras/\n")
