# ############################################################################
# CHEATSHEET — Introducción al Machine Learning (IIND3118)
# R · listo para usar en el parcial
# ############################################################################
#
# CÓMO USAR ESTE ARCHIVO
#   1. Corre el bloque SETUP (librerías). Una sola vez.
#   2. Ve a CARGAR DATOS y cambia SOLO el bloque marcado con ✏️.
#   3. Salta al método que necesites. Todos usan los mismos nombres:
#        df, X, y, Xs, nombres_X, Xtrain, Xtest, ytrain, ytest, Xs_train, Xs_test
#
# REGLA PRÁCTICA
#   Si el método usa DISTANCIAS (kmeans, knn, dbscan, PCA, LOF) -> usa Xs
#   Si es lineal con PENALIZACIÓN (ridge/lasso)                 -> usa Xs
#   Regresión lineal simple                                      -> X está bien
#
# ÍNDICE
#   PARTE A — SUPERVISADO
#     A1  Partición train/test y MSE
#     A2  Flexibilidad y sesgo-varianza
#     A3  KNN y kernel regression
#     A4  Métricas (regresión y clasificación)
#     A5  AIC, BIC, Cp, R2 ajustado
#     A6  Validación cruzada
#     A7  Regresión lineal
#     A8  Selección de variables (forward / exhaustiva)
#     A9  Ridge, Lasso, Elastic Net
#     A10 Comparación de modelos
#   PARTE B — NO SUPERVISADO
#     B1  K-Means          B8   PCA
#     B2  K-Medoides       B9   MDS
#     B3  Jerárquico       B10  Análisis Factorial
#     B4  DBSCAN           B11  Kernel PCA
#     B5  Métricas cluster B12  SVD
#     B6  Z-Score/Mahalan. B13  Manifold (Isomap, t-SNE)
#     B7  LOF / iForest
#
# NOTA: este archivo NO se pudo ejecutar en el entorno donde se generó
# (no había R instalado). Córrelo una vez completo ANTES del parcial para
# confirmar que las librerías están instaladas.
# ############################################################################


# ============================================================================
# SETUP — instala lo que falte y carga librerías
# ============================================================================
paquetes <- c("MASS","cluster","dbscan","glmnet","leaps","caret","class",
              "fpc","clusterCrit","isotree","kernlab","Rtsne","psych","car")
faltan <- paquetes[!(paquetes %in% installed.packages()[,"Package"])]
if (length(faltan)) install.packages(faltan, repos="https://cloud.r-project.org")

library(MASS); library(cluster); library(dbscan); library(glmnet); library(leaps)

set.seed(0)


# ############################################################################
# ⬇️ CARGAR DATOS — EL ÚNICO BLOQUE QUE DEBES EDITAR
# ############################################################################

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  ✏️  CAMBIA SOLO ESTO                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# --- Opción 1: CSV (lo más común) ------------------------------------------
df     <- read.csv("data/ejemplo_regresion.csv")
TARGET <- "target"        # nombre EXACTO de la columna respuesta. NULL si no hay.

# --- Opción 2: Excel -------------------------------------------------------
# library(readxl); df <- read_excel("data/mis_datos.xlsx"); TARGET <- "y"

# --- Opción 3: texto separado por espacios --------------------------------
# df <- read.table("data/measure.txt", header=TRUE); TARGET <- NULL

# --- Opción 4: dataset de un paquete ---------------------------------------
# library(ISLR); df <- na.omit(Hitters); TARGET <- "Salary"
# df <- iris; TARGET <- "Species"

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  ⬇️  DE AQUÍ PARA ABAJO NO TOQUES NADA                                ║
# ╚═══════════════════════════════════════════════════════════════════════╝

df <- na.omit(df)

if (!is.null(TARGET)) {
  y_raw <- df[[TARGET]]
  X_df  <- df[, setdiff(names(df), TARGET), drop=FALSE]
} else {
  y_raw <- NULL
  X_df  <- df
}

# Las columnas de texto se convierten a variables dummy automáticamente
X_df <- as.data.frame(model.matrix(~ . - 1, data = X_df))

nombres_X <- names(X_df)
X  <- as.matrix(X_df)
Xs <- scale(X)                       # estandarizado: media 0, sd 1

if (!is.null(y_raw)) {
  ES_CLASIFICACION <- is.factor(y_raw) || is.character(y_raw) ||
                      length(unique(y_raw)) <= 10
  y <- if (ES_CLASIFICACION) as.factor(y_raw) else as.numeric(y_raw)

  n     <- nrow(X)
  idx   <- sample(seq_len(n), size = floor(0.70 * n))   # 70% train
  Xtrain <- X[idx, ];  Xtest <- X[-idx, ]
  ytrain <- y[idx];    ytest <- y[-idx]
  Xs_train <- Xs[idx, ]; Xs_test <- Xs[-idx, ]
  df_train <- df[idx, ]; df_test <- df[-idx, ]
} else {
  y <- NULL; ES_CLASIFICACION <- FALSE
}

cat(sprintf("df        : %d filas x %d columnas\n", nrow(df), ncol(df)))
cat(sprintf("X         : %d obs. x %d predictores\n", nrow(X), ncol(X)))
cat("predictores:", paste(nombres_X, collapse=", "), "\n")
if (!is.null(y)) {
  cat(sprintf("y ('%s') : %s\n", TARGET,
              ifelse(ES_CLASIFICACION,"CLASIFICACIÓN","REGRESIÓN")))
  cat(sprintf("train/test: %d / %d\n", length(ytrain), length(ytest)))
} else cat("y         : ninguna (problema NO SUPERVISADO)\n")

head(df)


# ############################################################################
# PARTE A — APRENDIZAJE SUPERVISADO
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


# ============================================================================
# A5. AIC, BIC, Cp DE MALLOWS Y R2 AJUSTADO
# ----------------------------------------------------------------------------
# EL PROBLEMA: si agregas variables, el R2 SIEMPRE sube y el MSE de train
# SIEMPRE baja, aunque las variables sean basura. Hace falta una métrica que
# CASTIGUE la complejidad.
#
#   R2 ajustado = 1 - (1-R2)(n-1)/(n-p-1)     MAXIMIZAR
#   AIC = -2logL + 2p                         MINIMIZAR
#   BIC = -2logL + p*log(n)                   MINIMIZAR
#   Cp  = (SSE + 2p*sigma^2)/n                MINIMIZAR
#
# ⚠️ AIC vs BIC (la pregunta clásica)
#   El castigo del BIC es p*log(n); el del AIC es 2p. Como log(n) > 2 cuando
#   n > 7, el BIC CASTIGA MÁS la complejidad => elige modelos MÁS PEQUEÑOS.
#   AIC: busca el que mejor predice.  BIC: más conservador.
# ============================================================================

if (!ES_CLASIFICACION) {
  dtr <- data.frame(Xtrain, ytrain=ytrain)
  p_max <- ncol(Xtrain)
  res <- data.frame()
  for (k in 1:p_max) {
    f <- as.formula(paste("ytrain ~", paste(nombres_X[1:k], collapse="+")))
    m <- lm(f, data=dtr)
    sse <- sum(resid(m)^2); nn <- nrow(dtr)
    sigma2 <- sse/(nn-k-1)
    res <- rbind(res, data.frame(p=k, r2adj=summary(m)$adj.r.squared,
                                 AIC=AIC(m), BIC=BIC(m), Cp=(sse+2*k*sigma2)/nn))
  }
  print(round(res,3))

  par(mfrow=c(1,4))
  plot(res$p, res$r2adj, type="b", pch=16, xlab="nº variables", main="R2 ajustado (MAX)")
  abline(v=res$p[which.max(res$r2adj)], col="red", lty=2)
  plot(res$p, res$AIC, type="b", pch=16, xlab="nº variables", main="AIC (MIN)")
  abline(v=res$p[which.min(res$AIC)], col="red", lty=2)
  plot(res$p, res$BIC, type="b", pch=16, xlab="nº variables", main="BIC (MIN)")
  abline(v=res$p[which.min(res$BIC)], col="red", lty=2)
  plot(res$p, res$Cp, type="b", pch=16, xlab="nº variables", main="Cp Mallows (MIN)")
  abline(v=res$p[which.min(res$Cp)], col="red", lty=2)
  par(mfrow=c(1,1))

  cat(sprintf("Mejor por R2adj: %d | AIC: %d | BIC: %d | Cp: %d variables\n",
              which.max(res$r2adj), which.min(res$AIC),
              which.min(res$BIC), which.min(res$Cp)))
  cat("El BIC suele elegir MENOS variables porque castiga más.\n")
}


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


# ============================================================================
# B1. K-MEANS
# ----------------------------------------------------------------------------
# QUÉ HACE: parte los datos en K grupos. Cada grupo tiene un CENTRO (el
# promedio de sus puntos) y cada punto va AL CENTRO MÁS CERCANO. Repite:
# recalcula centros -> reasigna puntos, hasta que nada cambia.
#
#   min  suma_k suma_{x en C_k} ||x - centro_k||^2      (WSS)
#
# IDENTIDAD ANOVA: SST = BSS + WSS, con SST FIJO
#   => minimizar WSS  ES LO MISMO QUE  maximizar BSS
#      (coherencia interna y separación son el MISMO objetivo)
#
# PARÁMETROS de kmeans()
#   centers  : K, el número de grupos. Calibrar con codo o Silhouette (B5)
#   nstart   : ⚠️ PON 10 O MÁS. Repite con distintos arranques y se queda con
#              el mejor. Es la defensa contra óptimos locales.
#   iter.max : sube a 50 si no converge
#
# ✅ Rapidísimo · centroides interpretables · asigna puntos nuevos
# ❌ HAY QUE FIJAR K · sensible a inicialización, escala y outliers ·
#    solo clusters REDONDOS y de tamaño parecido ·
#    SIEMPRE devuelve K grupos aunque no haya estructura
# ============================================================================

K <- 3                                    # ⬅ cambia el número de grupos
km <- kmeans(Xs, centers=K, nstart=10, iter.max=50)

cat(sprintf("WSS total: %.2f\n", km$tot.withinss))
cat(sprintf("SST = %.1f = BSS (%.1f) + WSS (%.1f)\n",
            km$totss, km$betweenss, km$tot.withinss))
cat(sprintf("varianza explicada (BSS/SST): %.3f\n", km$betweenss/km$totss))
cat("tamaño de cada grupo:", table(km$cluster), "\n")
cat("\nCentroides:\n"); print(round(km$centers, 2))

plot(Z2, col=km$cluster, pch=16, xlab="PC1", ylab="PC2",
     main=sprintf("K-Means con K=%d (visto en el plano de PCA)", K))


# ============================================================================
# B2. K-MEDOIDES (PAM) Y DISTANCIA DE GOWER
# ----------------------------------------------------------------------------
# QUÉ HACE: igual que K-Means, pero el centro de cada grupo es UNA
# OBSERVACIÓN REAL (el medoide), no un promedio inventado.
#
#   medoide_k = la observación de C_k que minimiza la suma de distancias
#               a las demás del grupo
#
# POR QUÉ IMPORTA
#   1. ROBUSTO a outliers: un punto extremo arrastra un promedio, pero no
#      puede "sacar" al medoide de los datos.
#   2. Solo necesita una MATRIZ DE DISTANCIAS => sirve con variables
#      CATEGÓRICAS o MIXTAS usando GOWER.
#
# GOWER, EN SIMPLE: compara dos observaciones variable por variable. Si es
# numérica, la diferencia escalada por el rango; si es categórica, 0 si son
# iguales y 1 si no. Luego PROMEDIA. Da un número entre 0 y 1.
#
# PARÁMETROS de pam()
#   k     : número de grupos
#   diss  : TRUE si el primer argumento ya es una matriz de distancias
#   La DECISIÓN MÁS IMPORTANTE es la distancia, no k.
#
# ❌ COSTOSO: la matriz es n x n. Con n > 10.000 ya pesa -> usa clara()
# ============================================================================

library(cluster)
K <- 3

# Opción A: distancia euclidiana (variables numéricas)
D <- dist(Xs)

# Opción B: distancia de GOWER (variables mixtas / categóricas)
#   D <- daisy(X_df, metric="gower")

pam_res <- pam(D, k=K, diss=TRUE)

cat("medoides (filas del dataset):", pam_res$id.med, "\n")
cat("tamaño de cada grupo:", table(pam_res$clustering), "\n")
cat("\nLos medoides son observaciones REALES:\n")
print(round(X_df[pam_res$id.med, ], 2))

plot(Z2, col=pam_res$clustering, pch=16, xlab="PC1", ylab="PC2",
     main="K-Medoides")
points(Z2[pam_res$id.med, , drop=FALSE], col="red", pch=4, cex=3, lwd=4)
legend("topright", "medoides", col="red", pch=4)

# Silueta (funciona con CUALQUIER distancia)
sil <- silhouette(pam_res)
plot(sil, main="Silueta - K-Medoides")
cat(sprintf("ancho promedio de silueta: %.3f\n", mean(sil[,"sil_width"])))

# Para n grande: clara(X_df, k=3, samples=50)


# ============================================================================
# B3. CLUSTERING JERÁRQUICO
# ----------------------------------------------------------------------------
# QUÉ HACE: empieza con cada punto solo. En cada paso UNE los dos grupos más
# cercanos. Sigue hasta que queda uno. El resultado es un ÁRBOL (dendrograma)
# que muestra todo el proceso. Para obtener grupos, CORTAS el árbol.
#
# ⚠️ EL LINKAGE ES LO QUE DE VERDAD DECIDE EL RESULTADO
#   "single"   = distancia MÍNIMA entre puntos -> formas alargadas; sufre
#                CHAINING (une grupos por una cadena de puntos intermedios)
#   "complete" = distancia MÁXIMA -> grupos compactos; sensible a outliers
#   "average"  = PROMEDIO -> término medio, buen default
#   "ward.D2"  = menor aumento de WSS -> balanceado, lo más parecido a K-Means
#
# ⚠️ EN R USA "ward.D2", NO "ward.D" (este último existe por
#    retrocompatibilidad y no implementa el criterio original)
#
# PARÁMETROS
#   hclust(d, method=)  : d es una MATRIZ DE DISTANCIAS (no los datos)
#   cutree(hc, k=3)     : corta para obtener 3 grupos
#   cutree(hc, h=5.0)   : corta a la altura 5.0
#
# ✅ No fija K de antemano · determinista · da estructura de parentesco
# ❌ FUSIONES IRREVERSIBLES (greedy, no optimiza nada global) ·
#    O(n^2)-O(n^3) · MUY sensible al linkage ·
#    siempre dibuja un árbol, haya estructura o no
# ============================================================================

K <- 3
par(mfrow=c(2,4))
for (m in c("single","complete","average","ward.D2")) {
  hc <- hclust(dist(Xs), method=m)          # ⬅ cambia el linkage aquí
  et <- cutree(hc, k=K)                     # ⬅ corta para obtener K grupos
  cat(sprintf("%-9s -> tamaños %s\n", m, paste(table(et), collapse=" ")))
  plot(hc, labels=FALSE, hang=-1, main=m, xlab="", sub="", ylab="distancia")
  rect.hclust(hc, k=K, border="red")
}
for (m in c("single","complete","average","ward.D2")) {
  et <- cutree(hclust(dist(Xs), method=m), k=K)
  plot(Z2, col=et, pch=16, xlab="PC1", ylab="PC2", main=m)
}
par(mfrow=c(1,1))


# ============================================================================
# B4. DBSCAN
# ----------------------------------------------------------------------------
# QUÉ HACE: busca ZONAS DENSAS. Un cluster es un montón de puntos pegados.
# No exige que todos estén cerca del centro, solo que haya una CADENA de
# vecinos densos. Por eso encuentra FORMAS RARAS (espirales, anillos).
# Además marca como RUIDO los puntos que no caen en ninguna zona densa.
# Es el ÚNICO método de clustering que dice "este punto no va en ningún lado".
#
# LOS 3 TIPOS DE PUNTO
#   core   : tiene al menos minPts vecinos dentro del radio eps
#   border : no es core, pero está dentro del radio de un core
#   noise  : ninguna de las dos -> etiqueta 0 EN R  (⚠️ en Python es -1)
#
# CÓMO ELEGIR eps: LA CURVA k-dist
#   1. distancia de cada punto a su k-ésimo vecino
#   2. ordenar y graficar  ->  kNNdistplot()
#   3. el CODO de la curva es un buen eps
#
# PARÁMETROS
#   eps    : ⚠️ EL CRÍTICO. Radio de la vecindad. Se lee del codo.
#   minPts : regla práctica >= p+1, o 2p si hay ruido
#
# DIAGNÓSTICO: casi todo es ruido -> eps muy chico.
#              un solo cluster gigante -> eps muy grande.
#
# ✅ Formas arbitrarias · no fija K · detecta outliers
# ❌ Calibración difícil y MUY sensible ·
#    FALLA si los clusters tienen densidades distintas (usa hdbscan) ·
#    se degrada en dimensión alta
# ============================================================================

library(dbscan)
k_vec <- min(5, nrow(Xs)-1)

# Paso 1: la curva k-dist
kNNdistplot(Xs, k=k_vec)
title("Curva k-dist: busca el CODO para elegir eps")
d_k <- sort(as.vector(kNNdist(Xs, k=k_vec)))
abline(h=quantile(d_k, .90), col="red", lty=2)
cat(sprintf("Sugerencia: eps entre %.2f (p80) y %.2f (p95)\n",
            quantile(d_k,.80), quantile(d_k,.95)))

# Paso 2: probar varios eps
cat("\nSensibilidad a eps:\n")
for (eps in round(quantile(d_k, c(.5,.7,.8,.9,.95)), 2)) {
  lab <- dbscan(Xs, eps=eps, minPts=k_vec+1)$cluster
  cat(sprintf("  eps=%5.2f -> %d clusters, %d puntos de ruido\n",
              eps, length(setdiff(unique(lab),0)), sum(lab==0)))
}

# Paso 3: modelo final
EPS <- as.numeric(quantile(d_k, .90))     # ⬅ ajusta según la curva
db  <- dbscan(Xs, eps=EPS, minPts=k_vec+1)
cat(sprintf("\nFINAL: eps=%.2f -> %d clusters, %d ruido\n",
            EPS, length(setdiff(unique(db$cluster),0)), sum(db$cluster==0)))

plot(Z2, col=ifelse(db$cluster==0, "red", db$cluster+1),
     pch=ifelse(db$cluster==0, 4, 16),
     xlab="PC1", ylab="PC2", main=sprintf("DBSCAN (eps=%.2f)", EPS))
legend("topright", "ruido (0)", col="red", pch=4)

# Si las densidades son muy distintas: hdbscan(Xs, minPts=5)


# ============================================================================
# B5. MÉTRICAS DE CLUSTERING — CÓMO ELEGIR K
# ----------------------------------------------------------------------------
# MÉTODO DEL CODO: grafica WSS vs K. El WSS SIEMPRE baja (con K=n es 0), así
# que no se minimiza: se busca DÓNDE DEJA DE BAJAR RÁPIDO.
#
# SILHOUETTE (la más usada), por observación:
#   s(i) = (b(i) - a(i)) / max(a(i), b(i))    en [-1, 1]
#     a(i) = distancia promedio a los de SU grupo       (cohesión)
#     b(i) = distancia promedio al grupo VECINO         (separación)
#   s ~ 1  bien asignada | s ~ 0  en la frontera | s < 0  MAL ASIGNADA
#   Promedio: >0.7 fuerte · 0.5-0.7 razonable · 0.25-0.5 débil · <0.25 nada
#
# CALINSKI-HARABASZ = [BSS/(K-1)] / [WSS/(n-K)]    MAXIMIZAR
# DAVIES-BOULDIN                                   MINIMIZAR
#
# ⚠️ LA TRAMPA (esto lo preguntan)
#   Las tres asumen clusters REDONDOS separados por sus centros. Con anillos
#   concéntricos premian a K-Means aunque esté visiblemente MAL, porque los
#   dos anillos COMPARTEN CENTROIDE. NO las uses para comparar K-Means
#   contra DBSCAN en formas irregulares.
#   Y si las tres discrepan: NO las promedies. Significa que no hay
#   estructura clara y K se decide por criterio del problema.
# ============================================================================

library(fpc); library(clusterCrit)
D <- dist(Xs)
Ks <- 2:10
res_c <- data.frame()
for (k in Ks) {
  kmk <- kmeans(Xs, centers=k, nstart=10)
  res_c <- rbind(res_c, data.frame(
    K   = k,
    WSS = kmk$tot.withinss,
    sil = mean(silhouette(kmk$cluster, D)[,"sil_width"]),        # MAX
    ch  = calinhara(Xs, kmk$cluster, cn=k),                      # MAX
    db  = intCriteria(Xs, as.integer(kmk$cluster), "Davies_Bouldin")$davies_bouldin  # MIN
  ))
}
print(round(res_c, 3))
cat(sprintf("\nMejor K: Silhouette=%d | Calinski=%d | Davies-Bouldin=%d\n",
            res_c$K[which.max(res_c$sil)], res_c$K[which.max(res_c$ch)],
            res_c$K[which.min(res_c$db)]))

par(mfrow=c(1,4))
plot(res_c$K, res_c$WSS, type="b", pch=16, xlab="K", main="Codo: WSS")
plot(res_c$K, res_c$sil, type="b", pch=16, xlab="K", main="Silhouette (MAX)")
abline(v=res_c$K[which.max(res_c$sil)], col="red", lty=2)
plot(res_c$K, res_c$ch,  type="b", pch=16, xlab="K", main="Calinski-H (MAX)")
abline(v=res_c$K[which.max(res_c$ch)], col="red", lty=2)
plot(res_c$K, res_c$db,  type="b", pch=16, xlab="K", main="Davies-B (MIN)")
abline(v=res_c$K[which.min(res_c$db)], col="red", lty=2)
par(mfrow=c(1,1))

# Gráfico de silueta por cluster (más informativo que el promedio)
k_sil <- res_c$K[which.max(res_c$sil)]
plot(silhouette(kmeans(Xs, centers=k_sil, nstart=10)$cluster, D),
     main=paste("Silueta por cluster (K =", k_sil, ")"))


# ============================================================================
# B6. OUTLIERS: Z-SCORE Y MAHALANOBIS
# ----------------------------------------------------------------------------
# Z-SCORE (una variable a la vez)
#   Z = (x - media) / sd      Regla: |Z| > 3 es atípico (~0.27%)
#
# ⚠️ EL ENMASCARAMIENTO
#   Los outliers INFLAN la media y la sd, y así SE ESCONDEN A SÍ MISMOS.
#   Con varios outliers juntos, sus propios Z caen bajo 3 y no se detectan.
#   Además el Z tiene un TECHO: |Z| <= (n-1)/sqrt(n).
#   SOLUCIÓN: Z MODIFICADO con mediana y MAD (aguanta 50% de contaminación)
#     Z_mod = 0.6745 * (x - mediana) / MAD
#
# MAHALANOBIS (todas las variables juntas)
#   d^2 = (x-media)' S^-1 (x-media)  ~  chi2 con p grados de libertad
#   Corrige por ESCALA Y CORRELACIÓN. Sus fronteras son ELIPSES, no círculos.
#   Umbral: qchisq(0.975, df=p)   con p = nº de VARIABLES
#
# ⚠️ Mirar variable por variable NO ES LO MISMO que Mahalanobis. Un punto
#    puede ser normal en cada variable y atípico en conjunto por ROMPER LA
#    CORRELACIÓN.
#
# ⚠️ CASO HADLUM (lo preguntan): un embarazo de 349 días es improbable pero
#    posible. UN OUTLIER ESTADÍSTICO NO ES AUTOMÁTICAMENTE UN ERROR NI UN
#    FRAUDE: es una observación improbable BAJO EL MODELO ASUMIDO. Con n
#    grande siempre habrá puntos en las colas.
#
# VERSIÓN ROBUSTA: cov.rob(method="mcd") estima con la mayoría limpia de los
# datos, así el grupo contaminante no puede esconderse a sí mismo.
# ============================================================================

# Z-Score clásico y modificado, por variable
z     <- abs(scale(X))
z_mod <- abs(0.6745 * sweep(X, 2, apply(X,2,median)) /
             rep(apply(X, 2, mad, constant=1), each=nrow(X)))

cat("Outliers por variable (|Z| > 3):\n")
for (j in seq_along(nombres_X))
  cat(sprintf("  %-22s Z clásico: %3d   Z modificado: %3d\n",
              nombres_X[j], sum(z[,j]>3), sum(z_mod[,j]>3)))
cat("El Z modificado detecta MÁS porque no se deja enmascarar.\n")

# Mahalanobis clásica y robusta
p_var  <- ncol(X)
umbral <- qchisq(0.975, df=p_var)          # ⬅ df = nº de VARIABLES

d2     <- mahalanobis(X, colMeans(X), cov(X))
rob    <- cov.rob(X, method="mcd")          # MCD: versión robusta
d2_rob <- mahalanobis(X, rob$center, rob$cov)

cat(sprintf("\numbral chi2(0.975, df=%d) = %.2f\n", p_var, umbral))
cat(sprintf("atípicos, Mahalanobis clásica: %d\n", sum(d2 > umbral)))
cat(sprintf("atípicos, Mahalanobis robusta: %d  <- detecta más\n", sum(d2_rob > umbral)))

par(mfrow=c(1,2))
plot(Z2, col=ifelse(d2>umbral,"red","steelblue"), pch=16,
     xlab="PC1", ylab="PC2", main="Mahalanobis clásica")
plot(Z2, col=ifelse(d2_rob>umbral,"red","steelblue"), pch=16,
     xlab="PC1", ylab="PC2", main="Mahalanobis robusta (MCD)")
par(mfrow=c(1,1))

cat("Filas más atípicas:", order(d2_rob, decreasing=TRUE)[1:8], "\n")


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


# ============================================================================
# B8. PCA
# ----------------------------------------------------------------------------
# QUÉ HACE: tienes muchas variables correlacionadas (redundantes). PCA las
# RESUME en unas pocas variables nuevas (los componentes), que son
# COMBINACIONES LINEALES de las originales y NO están correlacionadas entre
# sí. El primer componente es LA DIRECCIÓN DONDE LOS DATOS MÁS SE DISPERSAN.
#
#   S = W L W'    (descomposición espectral de la covarianza)
#     w_j = autovector j = componente j (los LOADINGS)
#     l_j = autovalor    = varianza en esa dirección
#   Scores (coordenadas nuevas): Z = (X - media) W
#   Varianza explicada del componente j: l_j / suma(l)
#
# LOS COEFICIENTES DE CADA COMPONENTE están en pca$rotation:
#   una COLUMNA por componente, una FILA por variable. Norma 1 cada columna.
#   ⚠️ EL SIGNO ES ARBITRARIO (-w también es autovector). R y Python pueden
#      dar signos opuestos: NO es un error.
#
# CÓMO ELEGIR CUÁNTOS COMPONENTES
#   Scree plot (codo) | varianza acumulada (~90%) | Kaiser (l_j > 1)
#
# PARÁMETROS de prcomp()
#   center = TRUE   siempre
#   scale. = TRUE   usa la matriz de CORRELACIÓN (equivale a estandarizar)
#            FALSE  usa la de COVARIANZA
#            ⚠️ PON TRUE si las variables están en unidades distintas
#   prcomp (SVD, divisor n-1) es MÁS ESTABLE que princomp (espectral, div. n)
#
# ✅ Óptimo entre proyecciones lineales · solución cerrada ·
#    elimina multicolinealidad · proyecta datos nuevos
# ❌ SOLO ESTRUCTURA LINEAL · componentes difíciles de interpretar
#    (mezclan todas las variables) · sensible a escala y outliers ·
#    MAXIMIZAR VARIANZA NO ES LO MISMO QUE MAXIMIZAR INFORMACIÓN ÚTIL
# ============================================================================

pca <- prcomp(X, center=TRUE, scale.=TRUE)     # ⬅ scale.=TRUE casi siempre
print(summary(pca))

varexp <- pca$sdev^2 / sum(pca$sdev^2)
cat(sprintf("\nComponentes para el 90%% de varianza: %d\n",
            which(cumsum(varexp) >= .90)[1]))
cat(sprintf("Criterio de Kaiser (lambda>1): %d componentes\n", sum(pca$sdev^2 > 1)))

cat("\nLOADINGS (coeficientes de cada componente):\n")
print(round(pca$rotation[, 1:min(4, ncol(pca$rotation))], 3))
cat("Lee cada COLUMNA: qué variables pesan más en ese componente.\n")

par(mfrow=c(1,3))
plot(pca$sdev^2, type="b", pch=16, xlab="componente", ylab="varianza (lambda)",
     main="Scree plot (codo)")
abline(h=1, col="red", lty=2)
plot(cumsum(varexp), type="b", pch=16, xlab="nº componentes",
     ylab="varianza acumulada", main="Varianza acumulada")
abline(h=.90, col="red", lty=2)
plot(pca$x[,1], pca$x[,2], pch=16, col=if(!is.null(y)) as.integer(as.factor(y)) else "steelblue",
     xlab=sprintf("PC1 (%.1f%%)", 100*varexp[1]),
     ylab=sprintf("PC2 (%.1f%%)", 100*varexp[2]), main="Datos en 2D")
par(mfrow=c(1,1))

biplot(pca, main="Biplot: observaciones + variables")


# ============================================================================
# B9. MDS (MULTIDIMENSIONAL SCALING)
# ----------------------------------------------------------------------------
# QUÉ HACE: tienes SOLO LAS DISTANCIAS entre observaciones, no sus
# coordenadas. MDS RECONSTRUYE UN MAPA de puntos que respete esas distancias.
# El ejemplo clásico: con las distancias por carretera entre ciudades
# europeas, MDS DIBUJA EL MAPA DE EUROPA sin haber visto una latitud.
#
#   Doble centrado: B = -1/2 C D^2 C   con C = I - (1/n)11'
#   Luego descomposición espectral de B.
#
# ⚠️ EQUIVALENCIA CLAVE (la preguntan)
#   MDS CLÁSICO CON DISTANCIA EUCLIDIANA = PCA
#   Mismo procedimiento visto desde dos lados: PCA descompone la covarianza
#   (p x p), MDS la de productos punto (n x n).
#
# ⚠️ AUTOVALORES NEGATIVOS = DIAGNÓSTICO
#   Si aparecen y son grandes, la matriz D NO ES EUCLIDIANA: ninguna
#   configuración de puntos puede reproducir esas distancias exactamente.
#   Pasa con GOWER y con distancias por carretera.
#
# MDS NO MÉTRICO (Kruskal, isoMDS): solo respeta el ORDEN de las
#   disimilitudes. Para datos ordinales (Likert, rankings).
#   STRESS: <5% excelente · <10% bueno · <20% aceptable
#
# PARÁMETROS
#   cmdscale(d, k=2, eig=TRUE)   k = dimensión; eig=TRUE devuelve autovalores
#   isoMDS(d, k=2)               versión no métrica
#
# ❌ O(n^3) · NO PROYECTA DATOS NUEVOS (PCA sí) ·
#    los ejes NO TIENEN INTERPRETACIÓN (se puede rotar/reflejar libremente)
# ============================================================================

D <- dist(Xs)                                  # ⬅ cambia la métrica si quieres
mds <- cmdscale(D, k=2, eig=TRUE)              # ⬅ eig=TRUE para ver autovalores

cat("autovalores (primeros 5):", round(mds$eig[1:5], 3), "\n")
cat(sprintf("negativos relevantes: %d  (0 => D SÍ es euclidiana)\n",
            sum(mds$eig < -1e-8)))
cat(sprintf("varianza explicada 2D: %.2f%%\n",
            100*sum(mds$eig[1:2])/sum(mds$eig[mds$eig>0])))

# Verificación: MDS clásico == PCA
cat(sprintf("MDS vs PCA, diferencia máxima: %.2e  <- deben ser iguales\n",
            max(abs(abs(mds$points) - abs(pca$x[,1:2])))))

# MDS no métrico (solo usa el ORDEN)
mds_nm <- isoMDS(D, k=2, trace=FALSE)
cat(sprintf("STRESS (no métrico): %.2f%%  (<10%% es bueno)\n", mds_nm$stress))

par(mfrow=c(1,3))
plot(mds$points, pch=16, col=if(!is.null(y)) as.integer(as.factor(y)) else "steelblue",
     xlab="Dim 1", ylab="Dim 2", main="MDS clásico (= PCA)")
plot(mds_nm$points, pch=16, col=if(!is.null(y)) as.integer(as.factor(y)) else "steelblue",
     xlab="Dim 1", ylab="Dim 2", main="MDS no métrico")
sh <- Shepard(D, mds$points)
plot(sh, pch=16, cex=.4, col=rgb(0,0,0,.3),
     xlab="disimilitud original", ylab="distancia en el mapa",
     main="Diagrama de Shepard")
abline(0, 1, col="red", lty=2)
par(mfrow=c(1,1))
# Shepard: entre más pegados a la diagonal, mejor preserva las distancias.


# ============================================================================
# B10. ANÁLISIS FACTORIAL
# ----------------------------------------------------------------------------
# QUÉ HACE: PCA resume variables. El análisis factorial va más allá: supone
# que hay CAUSAS OCULTAS (factores latentes) que PRODUCEN lo que observas.
# Ejemplo clásico: 9 pruebas de habilidad mental; detrás hay 3 factores no
# observables (verbal, espacial, numérico) que causan los puntajes.
#
#   X = L F + e        con    Sigma = L L' + Psi
#     L   = CARGAS: cuánto pesa cada factor sobre cada variable
#     Psi = UNICIDADES: la parte de cada variable que no comparte con nadie
#
# ⚠️ LA DIFERENCIA CON PCA (lo preguntan)
#              PCA                     ANÁLISIS FACTORIAL
#   Dirección  componentes = CONSECUENCIA   factores = CAUSA
#   Varianza   modela TODA                  solo la COMPARTIDA (separa ruido)
#   Solución   única y cerrada              iterativa, NO ÚNICA
#   Objetivo   comprimir                    INTERPRETAR
#
# LA ROTACIÓN: la solución no es única (si L sirve, L*R también). Se
# aprovecha para hacer los factores interpretables.
#   "varimax" (ortogonal): factores NO correlacionados. El default.
#   "promax"  (oblicua)  : permite factores correlacionados.
#   LA ROTACIÓN NO CAMBIA EL AJUSTE, solo el sistema de ejes.
#
# ⚠️ LA PRUEBA CHI-CUADRADO
#   H0: m factores bastan.
#   P-VALOR ALTO = EL MODELO CON m FACTORES SÍ ES ADECUADO.
#   Es AL REVÉS de la lectura habitual de un p-valor.
#
# ⚠️ LÍMITE DE IDENTIFICABILIDAD: m < (2p + 1 - sqrt(8p+1))/2
#   Con p=4 el máximo es 1 factor. Por eso factanal(iris,2) da ERROR:
#   NO es un bug, es una restricción del modelo.
#
# VERIFICACIÓN PREVIA
#   Bartlett: H0 = no hay nada que factorizar. Se quiere RECHAZAR.
#   KMO: >0.8 excelente · >0.6 aceptable · <0.5 inadecuado
#
# PARÁMETROS de factanal()
#   factors  : número de factores m
#   rotation : "varimax" | "promax" | "none"
#   scores   : "regression" o "Bartlett" para obtener los scores
#   covmat + n.obs : permite ajustar SOLO con la matriz de correlación
# ============================================================================

library(psych)

p_var <- ncol(X); n_obs <- nrow(X)
m_max <- floor((2*p_var + 1 - sqrt(8*p_var + 1))/2 - 1e-9)
cat(sprintf("Máximo de factores identificables: %d (con p=%d variables)\n", m_max, p_var))

# Verificación previa
print(cortest.bartlett(cor(X), n=n_obs))    # se quiere p-valor PEQUEÑO
cat(sprintf("KMO: %.3f\n", KMO(cor(X))$MSA))

# Prueba chi2 para elegir m
cat("\nDeterminando el número de factores:\n")
for (m in 1:min(m_max, 5)) {
  ff <- try(factanal(covmat=cor(X), factors=m, n.obs=n_obs), silent=TRUE)
  if (!inherits(ff, "try-error"))
    cat(sprintf("  m=%d  chi2=%8.2f  gl=%3d  p=%.4f  %s\n", m, ff$STATISTIC,
                ff$dof, ff$PVAL, ifelse(ff$PVAL>=.05,"ADECUADO","insuficiente")))
}
cat("⚠️ p-valor ALTO = el modelo SÍ es adecuado (al revés de lo habitual)\n")

# Ajuste final con rotación
M <- min(2, m_max)                          # ⬅ elige m según la tabla
fa <- factanal(X, factors=M, rotation="varimax", scores="regression")
print(fa)
cat("\ncomunalidad + unicidad = 1. Cargas |·|>0.3 definen el factor.\n")

# Compara con rotación promax (oblicua)
# fa2 <- factanal(X, factors=M, rotation="promax"); print(fa2)


# ============================================================================
# B11. KERNEL PCA
# ----------------------------------------------------------------------------
# QUÉ HACE: PCA solo encuentra estructura RECTA. Si los datos están sobre una
# curva o en círculos concéntricos, ninguna proyección lineal los separa.
# KPCA los manda a un espacio de MÁS dimensiones donde SÍ son separables,
# hace PCA ahí, y regresa. Lo notable: NUNCA CONSTRUYE ese espacio, usa el
# TRUCO DEL KERNEL: K(x,y) = <phi(x), phi(y)>
#
# KERNELS
#   "vanilladot" (lineal)  -> reproduce PCA exactamente
#   "polydot"    (polinomial) -> interacciones
#   "rbfdot"     (gaussiano)  -> EL MÁS USADO: exp(-sigma*||x-y||^2)
#
# ⚠️ SIGMA LO DECIDE TODO
#   muy GRANDE  -> cada punto solo se parece a sí mismo -> SIN estructura
#   muy PEQUEÑO -> todos se parecen -> CONVERGE A PCA LINEAL
#   Heurística de la mediana: sigma ~ 1/mediana(||xi-xj||^2)
#
# ⚠️ CONVENCIÓN: kernlab usa "sigma", sklearn usa "gamma" -> SON LO MISMO.
#    Pero otras librerías usan exp(-||x-y||^2/(2 sigma^2)), que NO lo es.
#
# PARÁMETROS de kpca()
#   kernel   : "rbfdot" | "polydot" | "vanilladot"
#   kpar     : list(sigma=...)  ⬅ EL parámetro crítico
#   features : número de componentes
#
# ✅ Captura no linealidad · SÍ proyecta datos nuevos (MDS y t-SNE no)
# ❌ O(n^3) · COMPONENTES SIN INTERPRETACIÓN · sin reconstrucción directa
#    (problema de la preimagen) · ES FÁCIL "ENCONTRAR" ESTRUCTURA ajustando
#    sigma hasta que el gráfico se vea bien
#
# ⚠️ REGLA: verifica siempre contra PCA lineal primero. Si PCA ya resuelve,
#    KPCA solo agrega opacidad.
# ============================================================================

library(kernlab)

sigma_sug <- 1/median(dist(Xs)^2)          # heurística de la mediana
cat(sprintf("sigma sugerido: %.4f\n", sigma_sug))

# Barrido de sigma: el parámetro que lo decide todo
sigmas <- sigma_sug * c(0.01, 0.1, 1, 10, 100)
par(mfrow=c(1, length(sigmas)+1))
plot(pca$x[,1:2], pch=16, cex=.6, main="PCA lineal\n(compara contra esto)",
     xlab="", ylab="", axes=FALSE); box()
for (s in sigmas) {
  kp <- kpca(~., data=as.data.frame(Xs), kernel="rbfdot",
             kpar=list(sigma=s), features=2)
  plot(rotated(kp), pch=16, cex=.6, main=sprintf("sigma=%.4f", s),
       xlab="", ylab="", axes=FALSE); box()
}
par(mfrow=c(1,1))

# Modelo final
kpc <- kpca(~., data=as.data.frame(Xs), kernel="rbfdot",
            kpar=list(sigma=sigma_sug),   # ⬅ EL parámetro crítico
            features=2)
plot(rotated(kpc), pch=16, col=if(!is.null(y)) as.integer(as.factor(y)) else "steelblue",
     xlab="KPC1", ylab="KPC2",
     main=sprintf("Kernel PCA (rbf, sigma=%.4f)", sigma_sug))

# KPCA SÍ puede proyectar datos nuevos: predict(kpc, datos_nuevos)


# ============================================================================
# B12. SVD (SINGULAR VALUE DECOMPOSITION)
# ----------------------------------------------------------------------------
# QUÉ HACE: descompone CUALQUIER matriz (no tiene que ser cuadrada ni
# simétrica) en tres piezas:   A = U D V'
# Los VALORES SINGULARES d1 >= d2 >= ... dicen cuánta "información" hay en
# cada dirección. Quedándote con los primeros k obtienes una VERSIÓN
# COMPRIMIDA de la matriz.
#
# ⚠️ ECKART-YOUNG (el teorema clave)
#   Truncar a los primeros k valores singulares da la MEJOR APROXIMACIÓN
#   POSIBLE DE RANGO k. Ninguna otra matriz de rango k se acerca más.
#   Y el error se sabe EN FÓRMULA CERRADA:
#       ||A - A_k||_F = sqrt( suma de los d_i^2 descartados )
#   -> los valores singulares descartados dicen EXACTAMENTE cuánto pierdes.
#
# ⚠️ RELACIÓN CON PCA
#   PCA = SVD SOBRE LOS DATOS CENTRADOS
#     V = los componentes | U*D = los scores | lambda_i = d_i^2/(n-1)
#   Por eso prcomp usa SVD: es MÁS ESTABLE que formar X'X.
#
# APLICACIÓN — LSI (recuperación de información)
#   Matriz término-documento -> SVD -> espacio SEMÁNTICO LATENTE.
#   Resuelve la SINONIMIA: buscar "petroleum" recupera documentos que solo
#   dicen "crude". Ranking por SIMILITUD COSENO (no euclidiana, porque los
#   documentos largos tendrían ventaja injusta).
#
# ⚠️ svd() de R devuelve v.  np.linalg.svd de Python devuelve Vt (TRANSPUESTA).
#    Al traducir, transponer.
#
# CÓMO ELEGIR k: la ENERGÍA ACUMULADA  cumsum(d^2)/sum(d^2)
# ============================================================================

Xc <- scale(Xs, center=TRUE, scale=FALSE)
sv <- svd(Xc)

cat(sprintf("u: %d x %d   d: %d   v: %d x %d\n",
            nrow(sv$u), ncol(sv$u), length(sv$d), nrow(sv$v), ncol(sv$v)))
cat("valores singulares:", round(sv$d, 3), "\n")
cat(sprintf("error de la descomposición: %.2e\n",
            max(abs(Xc - sv$u %*% diag(sv$d) %*% t(sv$v)))))

# SVD == PCA
cat(sprintf("\nd^2/(n-1) == varianzas de PCA: %s\n",
            isTRUE(all.equal(sv$d^2/(nrow(Xc)-1), pca$sdev^2, tolerance=1e-6))))

# Eckart-Young: el error se sabe en fórmula cerrada
energia <- cumsum(sv$d^2)/sum(sv$d^2)
cat(sprintf("\n%3s | %9s | %11s | %13s\n", "k", "energía", "error real", "error teórico"))
for (k in 1:min(6, length(sv$d))) {
  Ak <- sv$u[,1:k,drop=FALSE] %*% diag(sv$d[1:k], nrow=k) %*% t(sv$v)[1:k,,drop=FALSE]
  cat(sprintf("%3d | %9.4f | %11.4f | %13.4f\n", k, energia[k],
              sqrt(sum((Xc-Ak)^2)), sqrt(sum(sv$d[(k+1):length(sv$d)]^2))))
}

par(mfrow=c(1,2))
plot(sv$d, type="b", pch=16, log="y", xlab="i", ylab="valor singular",
     main="Valores singulares (log)")
plot(energia, type="b", pch=16, xlab="k", ylab="energía acumulada",
     main="Información retenida")
abline(h=.95, col="red", lty=2)
par(mfrow=c(1,1))


# ============================================================================
# B13. MANIFOLD LEARNING (ISOMAP, LLE, t-SNE)
# ----------------------------------------------------------------------------
# LA IDEA: a veces los datos viven sobre una SUPERFICIE DOBLADA dentro del
# espacio. Imagina una hoja de papel enrollada en 3D: en realidad es 2D,
# solo está torcida. PCA la APLASTA y pone juntos puntos que sobre la hoja
# están lejísimos. Estos métodos la DESENROLLAN.
# Clave: distinguir la distancia EN LÍNEA RECTA (a través del aire) de la
# distancia CAMINANDO SOBRE LA SUPERFICIE (geodésica).
#
# ISOMAP = MDS CON DISTANCIAS GEODÉSICAS
#   Construye un grafo de k vecinos, calcula caminos más cortos (eso aproxima
#   la geodésica) y aplica MDS.
#   ✅ Preserva estructura GLOBAL
#   ❌ MUY sensible a k: k grande -> crea ATAJOS que saltan de un pliegue a
#      otro y destruyen todo; k chico -> el grafo se desconecta · O(n^3)
#
# LLE: preserva RELACIONES LINEALES LOCALES.
#   ❌ No preserva distancias globales · tiende a COLAPSAR regiones
#
# t-SNE: convierte distancias en PROBABILIDADES y minimiza la divergencia KL.
#   ⚠️ LO QUE NO SE PUEDE LEER DE UN t-SNE (esto lo preguntan)
#      - LOS TAMAÑOS DE LOS CLUSTERS NO SIGNIFICAN NADA
#      - LAS DISTANCIAS ENTRE CLUSTERS NO SIGNIFICAN NADA
#      - Puede INVENTAR clusters en datos sin estructura
#      - NO es determinista · NO proyecta datos nuevos
#
# PARÁMETROS
#   Isomap(k=)      : ⚠️ EL CRÍTICO. 5-15 típico
#   Rtsne(perplexity=) : 5-50. Baja -> grumos falsos; alta -> se aplana
#   Rtsne(dims=2, max_iter=)
#
# ⚠️ EL CONTROL OBLIGATORIO
#   Estos métodos SIEMPRE dibujan algo bonito, incluso con RUIDO PURO.
#   Antes de interpretar, corre el mismo método sobre datos permutados.
# ============================================================================

library(Rtsne)
K_VEC <- min(10, nrow(Xs)-1)
PERP  <- min(30, floor((nrow(Xs)-1)/3))

col_pt <- if(!is.null(y)) as.integer(as.factor(y)) else "steelblue"

# t-SNE
set.seed(0)
ts <- Rtsne(Xs, dims=2, perplexity=PERP,   # ⬅ perplexity: 5-50
            max_iter=500, check_duplicates=FALSE, verbose=FALSE)

# Isomap (RDRToolbox está en Bioconductor; vegan es la alternativa en CRAN)
emb_iso <- NULL
if (requireNamespace("vegan", quietly=TRUE)) {
  emb_iso <- vegan::isomap(dist(Xs), k=K_VEC, ndim=2)$points[,1:2]
} else if (requireNamespace("RDRToolbox", quietly=TRUE)) {
  emb_iso <- RDRToolbox::Isomap(data=Xs, dims=2, k=K_VEC)[[1]]
}

par(mfrow=c(1,3))
plot(pca$x[,1:2], col=col_pt, pch=16, main="PCA (lineal)", xlab="", ylab="")
if (!is.null(emb_iso))
  plot(emb_iso, col=col_pt, pch=16, main=sprintf("Isomap (k=%d)", K_VEC), xlab="", ylab="")
else plot.new(); title("Isomap: instala vegan")
plot(ts$Y, col=col_pt, pch=16, main=sprintf("t-SNE (perp=%d)", PERP), xlab="", ylab="")
par(mfrow=c(1,1))

# ⚠️ EL CONTROL: ¿qué sale con datos SIN estructura?
set.seed(0)
ruido <- matrix(rnorm(nrow(Xs)*ncol(Xs)), nrow=nrow(Xs))
ts_r  <- Rtsne(ruido, dims=2, perplexity=5, check_duplicates=FALSE, verbose=FALSE)

par(mfrow=c(1,2))
plot(prcomp(ruido)$x[,1:2], pch=16, cex=.6, main="PCA sobre RUIDO", xlab="", ylab="")
plot(ts_r$Y, pch=16, cex=.6, main="t-SNE sobre RUIDO (perp=5)", xlab="", ylab="")
par(mfrow=c(1,1))
cat("Con perplexity baja, t-SNE FABRICA grumos convincentes sobre ruido puro.\n")
cat("MORALEJA: un gráfico bonito NO es evidencia de estructura.\n")


# ############################################################################
# TABLA FINAL — QUÉ MÉTODO USAR
# ############################################################################
#
# CON VARIABLE RESPUESTA y (SUPERVISADO)
#   Punto de partida siempre              -> Regresión lineal      (A7)
#   Relación no lineal, pocas variables   -> KNN / kernel          (A3)
#   Muchas variables correlacionadas      -> RIDGE                 (A9)
#   Muchas variables, pocas importan      -> LASSO (da ceros)      (A9)
#   Correlación Y quieres selección       -> ELASTIC NET           (A9)
#   p pequeño, quieres el mejor subconj.  -> Exhaustivo            (A8)
#   p grande, seleccionar rápido          -> Forward               (A8)
#   Comparar sin partir la muestra        -> AIC/BIC/Cp/R2adj      (A5)
#   Calibrar cualquier parámetro          -> Validación cruzada    (A6)
#
# SIN y (NO SUPERVISADO)
#   Agrupar, clusters redondos, n grande  -> K-Means               (B1)
#   Agrupar con outliers o categóricas    -> K-Medoides + Gower    (B2)
#   Ver jerarquía, no fijar K             -> Jerárquico            (B3)
#   Formas raras + detectar ruido         -> DBSCAN                (B4)
#   Elegir K                              -> Codo / Silhouette     (B5)
#   Outliers, pocas variables             -> Mahalanobis           (B6)
#   Outliers, muchas variables            -> Isolation Forest      (B7)
#   Outliers locales                      -> LOF                   (B7)
#   Reducir dimensión, visualizar         -> PCA                   (B8)
#   Solo tengo distancias                 -> MDS                   (B9)
#   Factores latentes interpretables      -> Análisis Factorial    (B10)
#   No lineal + proyectar datos nuevos    -> Kernel PCA            (B11)
#   Comprimir matriz / LSI                -> SVD                   (B12)
#   Solo visualizar estructura curva      -> Isomap / t-SNE        (B13)
#
#
# ############################################################################
# ⚠️ LAS 15 TRAMPAS DEL CURSO
# ############################################################################
#  1. El MSE de TRAIN siempre baja con más flexibilidad. Calibra con test o CV.
#  2. La curva del MSE de TEST tiene forma de U. El óptimo es el mínimo.
#  3. En KNN, Ridge y kernel el PARÁMETRO GRANDE = MODELO RÍGIDO (al revés).
#  4. BIC castiga más que AIC (p*log n vs 2p) => BIC elige modelos más chicos.
#  5. Accuracy ENGAÑA con clases desbalanceadas. Mira la matriz de confusión.
#  6. Lasso da CEROS exactos (selecciona); Ridge NUNCA. La diferencia clave.
#  7. HAY QUE ESTANDARIZAR en KNN, Ridge/Lasso, K-Means, PCA, DBSCAN.
#  8. Forward NO garantiza el mejor modelo (heurística); el exhaustivo sí.
#  9. K-Means SIEMPRE devuelve K grupos, aunque no haya estructura.
# 10. Silhouette/CH/DB premian a K-Means sobre DBSCAN en anillos concéntricos.
# 11. El Z-Score se ENMASCARA con grupos de outliers. Usa Z modificado.
# 12. Mirar variable por variable NO ES Mahalanobis (rompe la correlación).
# 13. p-valor ALTO en análisis factorial = modelo ADECUADO (al revés).
# 14. En t-SNE, tamaños y distancias ENTRE clusters NO significan nada.
# 15. Manifold learning SIEMPRE dibuja algo bonito, aunque sea ruido puro.
#
# ############################################################################
# 🔗 EQUIVALENCIAS QUE PREGUNTAN
# ############################################################################
#   MDS clásico (euclidiano)        = PCA
#   PCA                             = SVD sobre datos centrados
#   Isomap                          = MDS con distancias geodésicas
#   Kernel PCA con kernel lineal    = PCA
#   min WSS                         = max BSS  (ANOVA, solo K-Means euclidiano)
#   Elastic Net alpha=0 = Ridge  |  alpha=1 = Lasso
#
# ############################################################################
# DIFERENCIAS R vs PYTHON (para no confundirse)
# ############################################################################
#   Ruido en DBSCAN     R: 0            Python: -1
#   LOF                 R: mayor=atípico  Python: negative_outlier_factor_ al revés
#   Isolation Forest    R: mayor=anómalo  Python: decision_function al revés
#   SVD                 R: devuelve v     Python: devuelve Vt (transpuesta)
#   Ward                R: "ward.D2"      Python: "ward"
#   Reinicios K-Means   R: nstart         Python: n_init
#   Índices             R: desde 1        Python: desde 0
# ############################################################################
