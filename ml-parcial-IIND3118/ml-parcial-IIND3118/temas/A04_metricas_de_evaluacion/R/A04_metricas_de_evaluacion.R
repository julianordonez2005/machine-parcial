# ############################################################################
# A4. MÉTRICAS DE EVALUACIÓN
# Tema A04_metricas_de_evaluacion — Introducción al Machine Learning (IIND3118)
#
# CÓMO USARLO
#   1. Cambia el bloque marcado con ✏️ por tu dataset.
#   2. Corre:  Rscript A04_metricas_de_evaluacion.R     (o ábrelo en RStudio)
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
cat("A4. MÉTRICAS DE EVALUACIÓN\n")
cat(strrep("=",74), "\n")
cat(sprintf("datos: %d obs. x %d predictores\n\n", nrow(X), ncol(X)))

# ############################################################################
#  CÓDIGO DEL MÉTODO
# ############################################################################

# ============================================================================
# A4. MÉTRICAS DE EVALUACIÓN
# ----------------------------------------------------------------------------
# REGRESIÓN
#   MSE   = mean((pred-real)^2)          el estándar. MINIMIZAR
#   RMSE  = sqrt(MSE)                    igual pero EN LAS UNIDADES DE y
#   MAE   = mean(abs(pred-real))         más ROBUSTO a outliers
#   MAPE  = mean(abs((pred-real)/real))  error en %. No sirve si y puede ser 0
#   R2    = 1 - SSres/SStot              % de varianza explicada. MAXIMIZAR
#
# CLASIFICACIÓN — la matriz de confusión es la base
#                  Predicho 0   Predicho 1
#      Real 0         TN            FP
#      Real 1         FN            TP
#
#   Accuracy      = (TP+TN)/total     ⚠️ ENGAÑA con clases desbalanceadas
#   Sensibilidad  = TP/(TP+FN)        de los que SÍ eran, ¿cuántos detecté?
#   Especificidad = TN/(TN+FP)        de los que NO eran, ¿cuántos descarté?
#   Precisión     = TP/(TP+FP)        de los que predije SÍ, ¿cuántos acerté?
#   AUC-ROC                           0.5 = azar, 1 = perfecto
#
# ⚠️ LA TRAMPA DEL ACCURACY: si el 99% son clase 0, decir siempre "0" da
#    99% de accuracy y es inútil. Mira SIEMPRE la matriz de confusión.
# ============================================================================

evaluar <- function(real, pred, nombre="modelo") {
  cat(sprintf("-- %s --\n", nombre))
  if (ES_CLASIFICACION) {
    cm <- table(real=real, predicho=pred)
    print(cm)
    cat(sprintf("\nAccuracy: %.4f\n", sum(diag(cm))/sum(cm)))
    if (all(dim(cm)==c(2,2))) {
      tn<-cm[1,1]; fp<-cm[1,2]; fn<-cm[2,1]; tp<-cm[2,2]
      cat(sprintf("Sensibilidad : %.4f\n", tp/(tp+fn)))
      cat(sprintf("Especificidad: %.4f\n", tn/(tn+fp)))
      cat(sprintf("Precisión    : %.4f\n", tp/(tp+fp)))
    }
  } else {
    mse <- mean((real-pred)^2)
    cat(sprintf("MSE  : %.4f\n", mse))
    cat(sprintf("RMSE : %.4f  (unidades de y)\n", sqrt(mse)))
    cat(sprintf("MAE  : %.4f  (robusto a outliers)\n", mean(abs(real-pred))))
    cat(sprintf("R2   : %.4f\n", 1 - sum((real-pred)^2)/sum((real-mean(real))^2)))
  }
}

if (!ES_CLASIFICACION) {
  mod <- lm(ytrain ~ ., data=data.frame(Xtrain, ytrain=ytrain))
  pr  <- predict(mod, data.frame(Xtest))
  evaluar(ytest, pr, "Regresión lineal")

  # GRÁFICA de residuales
  par(mfrow=c(1,2))
  plot(pr, ytest-pr, pch=16, col=rgb(0,0,1,.5),
       xlab="predicho", ylab="residual", main="Residuales: SIN patrón")
  abline(h=0, col="red", lty=2)
  hist(ytest-pr, breaks=25, col="steelblue",
       main="Histograma de residuales", xlab="residual")
  par(mfrow=c(1,1))
}

# CURVA ROC (solo clasificación binaria)
# library(pROC)
# modlog <- glm(ytrain ~ ., data=data.frame(Xs_train, ytrain), family="binomial")
# prob   <- predict(modlog, data.frame(Xs_test), type="response")
# roc_obj <- pROC::roc(ytest, prob); plot(roc_obj); pROC::auc(roc_obj)

cat("\nListo. Revisa la carpeta figuras/\n")
