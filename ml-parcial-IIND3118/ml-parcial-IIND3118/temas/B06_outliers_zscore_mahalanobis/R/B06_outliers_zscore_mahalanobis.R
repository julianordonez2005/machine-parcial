# ############################################################################
# B6. OUTLIERS: Z-SCORE
# Tema B06_outliers_zscore_mahalanobis — Introducción al Machine Learning (IIND3118)
#
# CÓMO USARLO
#   1. Cambia el bloque marcado con ✏️ por tu dataset.
#   2. Corre:  Rscript B06_outliers_zscore_mahalanobis.R     (o ábrelo en RStudio)
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
cat("B6. OUTLIERS: Z-SCORE\n")
cat(strrep("=",74), "\n")
cat(sprintf("datos: %d obs. x %d predictores\n\n", nrow(X), ncol(X)))

# ############################################################################
#  CÓDIGO DEL MÉTODO
# ############################################################################

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

cat("\nListo. Revisa la carpeta figuras/\n")
