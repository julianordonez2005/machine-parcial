# ############################################################################
# B3. CLUSTERING JERÁRQUICO
# Tema B03_clustering_jerarquico — Introducción al Machine Learning (IIND3118)
#
# CÓMO USARLO
#   1. Cambia el bloque marcado con ✏️ por tu dataset.
#   2. Corre:  Rscript B03_clustering_jerarquico.R     (o ábrelo en RStudio)
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
cat("B3. CLUSTERING JERÁRQUICO\n")
cat(strrep("=",74), "\n")
cat(sprintf("datos: %d obs. x %d predictores\n\n", nrow(X), ncol(X)))

# ############################################################################
#  CÓDIGO DEL MÉTODO
# ############################################################################

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

cat("\nListo. Revisa la carpeta figuras/\n")
