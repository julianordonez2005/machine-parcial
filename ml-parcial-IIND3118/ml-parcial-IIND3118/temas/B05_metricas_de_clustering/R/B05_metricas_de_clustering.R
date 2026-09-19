# ############################################################################
# B5. MÉTRICAS DE CLUSTERING
# Tema B05_metricas_de_clustering — Introducción al Machine Learning (IIND3118)
#
# CÓMO USARLO
#   1. Cambia el bloque marcado con ✏️ por tu dataset.
#   2. Corre:  Rscript B05_metricas_de_clustering.R     (o ábrelo en RStudio)
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
cat("B5. MÉTRICAS DE CLUSTERING\n")
cat(strrep("=",74), "\n")
cat(sprintf("datos: %d obs. x %d predictores\n\n", nrow(X), ncol(X)))

# ############################################################################
#  CÓDIGO DEL MÉTODO
# ############################################################################

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

cat("\nListo. Revisa la carpeta figuras/\n")
