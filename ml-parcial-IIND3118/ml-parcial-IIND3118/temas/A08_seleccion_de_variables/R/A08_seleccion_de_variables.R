# ############################################################################
# A8. SELECCIÓN DE VARIABLES
# Tema A08_seleccion_de_variables — Introducción al Machine Learning (IIND3118)
#
# CÓMO USARLO
#   1. Cambia el bloque marcado con ✏️ por tu dataset.
#   2. Corre:  Rscript A08_seleccion_de_variables.R     (o ábrelo en RStudio)
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
cat("A8. SELECCIÓN DE VARIABLES\n")
cat(strrep("=",74), "\n")
cat(sprintf("datos: %d obs. x %d predictores\n\n", nrow(X), ncol(X)))

# ############################################################################
#  CÓDIGO DEL MÉTODO
# ############################################################################

# ============================================================================
# A8. SELECCIÓN DE VARIABLES
# ----------------------------------------------------------------------------
# EL PROBLEMA: con p variables hay 2^p - 1 modelos posibles. Con 20 variables
# son más de un millón. Meter todas no es la respuesta: las irrelevantes
# AGREGAN VARIANZA sin agregar señal.
#
# MÉTODO EXHAUSTIVO ("exhaustive")
#   Prueba TODOS los 2^p - 1 modelos.
#   ✅ GARANTIZA encontrar el mejor    ❌ imposible con p > ~20
#
# MÉTODO FORWARD ("forward")
#   Empieza vacío. Agrega la variable que más mejore. Repite.
#   Compara solo p(p+1)/2 modelos (con p=20: 210, no un millón).
#   ✅ rápido, sirve con p grande
#   ❌ ES HEURÍSTICA: NO garantiza el mejor. Una variable escogida al
#      principio nunca se vuelve a quitar.
#
# MÉTODO BACKWARD ("backward"): empieza con todas y va quitando. Requiere n>p.
#
# PARÁMETROS de regsubsets()
#   nvmax  : máximo de variables a considerar
#   method : "exhaustive" | "forward" | "backward"
# ============================================================================

if (!ES_CLASIFICACION) {
  dtr <- data.frame(Xtrain, ytrain=ytrain)

  # --- FORWARD -------------------------------------------------------------
  regfit_fwd <- regsubsets(ytrain ~ ., data=dtr,
                           nvmax = ncol(Xtrain),   # ⬅ máximo de variables
                           method = "forward")     # ⬅ "forward"|"backward"|"exhaustive"
  sum_fwd <- summary(regfit_fwd)

  par(mfrow=c(1,3))
  plot(sum_fwd$cp,    type="b", pch=16, xlab="nº variables", main="Cp Mallows (MIN)")
  abline(v=which.min(sum_fwd$cp), col="red", lty=2)
  plot(sum_fwd$bic,   type="b", pch=16, col="red",  xlab="nº variables", main="BIC (MIN)")
  abline(v=which.min(sum_fwd$bic), col="red", lty=2)
  plot(sum_fwd$adjr2, type="b", pch=16, col="blue", xlab="nº variables", main="R2 ajustado (MAX)")
  abline(v=which.max(sum_fwd$adjr2), col="red", lty=2)
  par(mfrow=c(1,1))

  k_best <- which.min(sum_fwd$cp)          # ⬅ o which.max(sum_fwd$adjr2)
  vars_sel <- names(coef(regfit_fwd, k_best))[-1]
  cat(sprintf("Forward: %d variables -> %s\n", k_best, paste(vars_sel, collapse=", ")))

  # Modelo final con las variables seleccionadas
  f_sel <- as.formula(paste("ytrain ~", paste(vars_sel, collapse="+")))
  mod_fwd <- lm(f_sel, data=dtr)
  mse_fwd <- mean((ytest - predict(mod_fwd, data.frame(Xtest)))^2)
  cat(sprintf("MSE test (forward): %.3f\n", mse_fwd))

  # --- EXHAUSTIVO (solo si p es pequeño) ----------------------------------
  if (ncol(Xtrain) <= 15) {
    regfit_exh <- regsubsets(ytrain ~ ., data=dtr,
                             nvmax=ncol(Xtrain), method="exhaustive")
    sum_exh <- summary(regfit_exh)
    k_exh <- which.min(sum_exh$cp)
    vars_exh <- names(coef(regfit_exh, k_exh))[-1]
    f_exh <- as.formula(paste("ytrain ~", paste(vars_exh, collapse="+")))
    mod_exh <- lm(f_exh, data=dtr)
    mse_exh <- mean((ytest - predict(mod_exh, data.frame(Xtest)))^2)
    cat(sprintf("Exhaustivo: %d variables -> MSE test = %.3f\n", k_exh, mse_exh))

    barplot(c(mse_fwd, mse_exh), names.arg=c("Forward","Exhaustivo"),
            col=c("steelblue","indianred"), ylab="MSE en test",
            main="Comparación de métodos de selección")
  } else {
    cat(sprintf("p=%d es demasiado para el exhaustivo (2^%d-1 modelos).\n",
                ncol(Xtrain), ncol(Xtrain)))
  }
}

cat("\nListo. Revisa la carpeta figuras/\n")
