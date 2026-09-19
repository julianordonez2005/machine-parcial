# ############################################################################
# B13. MANIFOLD LEARNING
# Tema B13_manifold_learning — Introducción al Machine Learning (IIND3118)
#
# CÓMO USARLO
#   1. Cambia el bloque marcado con ✏️ por tu dataset.
#   2. Corre:  Rscript B13_manifold_learning.R     (o ábrelo en RStudio)
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
cat("B13. MANIFOLD LEARNING\n")
cat(strrep("=",74), "\n")
cat(sprintf("datos: %d obs. x %d predictores\n\n", nrow(X), ncol(X)))

# ############################################################################
#  CÓDIGO DEL MÉTODO
# ############################################################################

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

cat("\nListo. Revisa la carpeta figuras/\n")
