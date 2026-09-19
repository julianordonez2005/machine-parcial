# ############################################################################
# B9. MDS
# Tema B09_mds — Introducción al Machine Learning (IIND3118)
#
# CÓMO USARLO
#   1. Cambia el bloque marcado con ✏️ por tu dataset.
#   2. Corre:  Rscript B09_mds.R     (o ábrelo en RStudio)
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
cat("B9. MDS\n")
cat(strrep("=",74), "\n")
cat(sprintf("datos: %d obs. x %d predictores\n\n", nrow(X), ncol(X)))

# ############################################################################
#  CÓDIGO DEL MÉTODO
# ############################################################################

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

cat("\nListo. Revisa la carpeta figuras/\n")
