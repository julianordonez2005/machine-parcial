# ############################################################################
# A9. RIDGE, LASSO
# Tema A09_ridge_lasso_elasticnet — Introducción al Machine Learning (IIND3118)
#
# CÓMO USARLO
#   1. Cambia el bloque marcado con ✏️ por tu dataset.
#   2. Corre:  Rscript A09_ridge_lasso_elasticnet.R     (o ábrelo en RStudio)
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
cat("A9. RIDGE, LASSO\n")
cat(strrep("=",74), "\n")
cat(sprintf("datos: %d obs. x %d predictores\n\n", nrow(X), ncol(X)))

# ############################################################################
#  CÓDIGO DEL MÉTODO
# ############################################################################

# ============================================================================
# A9. RIDGE, LASSO Y ELASTIC NET
# ----------------------------------------------------------------------------
# QUÉ HACEN: la regresión normal solo minimiza el error. Estas minimizan
# el error MÁS UN CASTIGO por tener coeficientes grandes:
#
#   beta = argmin  suma((Y - beta'X)^2)  +  lambda * castigo(beta)
#
# Si un coeficiente es enorme, el modelo probablemente sobreajusta.
# Encogerlos da estabilidad a cambio de un poquito de sesgo.
#
#   RIDGE  (alpha=0)   castigo = lambda*suma(beta^2)
#      -> ENCOGE los coeficientes pero NUNCA los deja en 0 exacto
#   LASSO  (alpha=1)   castigo = lambda*suma(|beta|)
#      -> puede dejarlos EXACTAMENTE en 0 => SELECCIONA VARIABLES
#   ELASTIC NET (0<alpha<1) mezcla de los dos
#
# ⚠️ LA DIFERENCIA QUE PREGUNTAN: Lasso hace selección (pone ceros),
#    Ridge no, solo encoge.
#
# EL PARÁMETRO lambda
#   lambda = 0        -> es la regresión lineal normal
#   lambda pequeño    -> modelo flexible
#   lambda grande     -> todos los coeficientes tienden a 0 -> modelo rígido
#   Se calibra con VALIDACIÓN CRUZADA: cv.glmnet()
#
# ⚠️ ESTANDARIZAR ES OBLIGATORIO (glmnet lo hace solo con standardize=TRUE)
#
# CUÁNDO USAR CUÁL
#   muchas variables correlacionadas, todas aportan  -> RIDGE
#   muchas variables pero pocas importan             -> LASSO
#   p > n                                            -> Lasso o Elastic Net
#   correlación Y quieres selección                  -> ELASTIC NET
#
# PARÁMETROS de glmnet()
#   alpha       : 0=Ridge, 1=Lasso, 0.3=Elastic Net
#   lambda      : el castigo. Usa cv.glmnet()$lambda.min
#   standardize : TRUE (por defecto). DÉJALO EN TRUE.
#   family      : "gaussian" (regresión) | "binomial" (clasif. binaria)
# ============================================================================

library(glmnet)

fam <- ifelse(ES_CLASIFICACION, "binomial", "gaussian")

for (info in list(list(0.0,"RIDGE"), list(1.0,"LASSO"), list(0.3,"ELASTIC NET"))) {
  a <- info[[1]]; nom <- info[[2]]

  # 1. Calibrar lambda con validación cruzada
  cvmod <- cv.glmnet(Xs_train, ytrain, alpha=a, family=fam, nfolds=10)

  # 2. Modelo final con el lambda óptimo
  mod <- glmnet(Xs_train, ytrain, alpha=a, lambda=cvmod$lambda.min, family=fam)

  pr  <- predict(mod, Xs_test)
  mse_p <- mean((as.numeric(ytest) - as.numeric(pr))^2)
  n_cero <- sum(coef(mod)[-1] == 0)

  cat(sprintf("\n%-12s lambda=%.5f  MSE=%.3f  coefs en cero: %d/%d\n",
              nom, cvmod$lambda.min, mse_p, n_cero, ncol(Xs_train)))
  if (a==0) cat("             (Ridge NUNCA da ceros)\n")
  if (a==1) cat(sprintf("             variables que sobreviven: %s\n",
                        paste(nombres_X[coef(mod)[-1]!=0], collapse=", ")))

  # GRÁFICA 1: curva de validación cruzada
  plot(cvmod); title(paste("CV -", nom), line=2.5)

  # GRÁFICA 2: caminos de los coeficientes (la clave)
  mod_path <- glmnet(Xs_train, ytrain, alpha=a, family=fam)
  plot(mod_path, xvar="lambda", label=TRUE)
  title(paste("Caminos de coeficientes -", nom), line=2.5)
  # En LASSO las líneas TOCAN el cero y se quedan ahí: eso es selección.
  # En RIDGE se acercan pero nunca lo tocan.
}

cat("\nListo. Revisa la carpeta figuras/\n")
