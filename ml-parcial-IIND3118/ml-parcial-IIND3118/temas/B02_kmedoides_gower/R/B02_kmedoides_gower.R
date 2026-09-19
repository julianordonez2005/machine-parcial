# ############################################################################
# B2. K-MEDOIDES
# Tema B02_kmedoides_gower — Introducción al Machine Learning (IIND3118)
#
# CÓMO USARLO
#   1. Cambia el bloque marcado con ✏️ por tu dataset.
#   2. Corre:  Rscript B02_kmedoides_gower.R     (o ábrelo en RStudio)
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
cat("B2. K-MEDOIDES\n")
cat(strrep("=",74), "\n")
cat(sprintf("datos: %d obs. x %d predictores\n\n", nrow(X), ncol(X)))

# ############################################################################
#  CÓDIGO DEL MÉTODO
# ############################################################################

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

cat("\nListo. Revisa la carpeta figuras/\n")
