# ############################################################################
# B12. SVD
# Tema B12_svd — Introducción al Machine Learning (IIND3118)
#
# CÓMO USARLO
#   1. Cambia el bloque marcado con ✏️ por tu dataset.
#   2. Corre:  Rscript B12_svd.R     (o ábrelo en RStudio)
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
cat("B12. SVD\n")
cat(strrep("=",74), "\n")
cat(sprintf("datos: %d obs. x %d predictores\n\n", nrow(X), ncol(X)))

# ############################################################################
#  CÓDIGO DEL MÉTODO
# ############################################################################

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

cat("\nListo. Revisa la carpeta figuras/\n")
