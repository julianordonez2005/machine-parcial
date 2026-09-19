# ############################################################################
# B11. KERNEL PCA
# Tema B11_kernel_pca — Introducción al Machine Learning (IIND3118)
#
# CÓMO USARLO
#   1. Cambia el bloque marcado con ✏️ por tu dataset.
#   2. Corre:  Rscript B11_kernel_pca.R     (o ábrelo en RStudio)
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
cat("B11. KERNEL PCA\n")
cat(strrep("=",74), "\n")
cat(sprintf("datos: %d obs. x %d predictores\n\n", nrow(X), ncol(X)))

# ############################################################################
#  CÓDIGO DEL MÉTODO
# ############################################################################

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

cat("\nListo. Revisa la carpeta figuras/\n")
