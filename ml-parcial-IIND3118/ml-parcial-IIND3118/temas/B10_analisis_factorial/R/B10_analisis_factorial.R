# ############################################################################
# B10. ANÁLISIS FACTORIAL
# Tema B10_analisis_factorial — Introducción al Machine Learning (IIND3118)
#
# CÓMO USARLO
#   1. Cambia el bloque marcado con ✏️ por tu dataset.
#   2. Corre:  Rscript B10_analisis_factorial.R     (o ábrelo en RStudio)
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
cat("B10. ANÁLISIS FACTORIAL\n")
cat(strrep("=",74), "\n")
cat(sprintf("datos: %d obs. x %d predictores\n\n", nrow(X), ncol(X)))

# ############################################################################
#  CÓDIGO DEL MÉTODO
# ############################################################################

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

cat("\nListo. Revisa la carpeta figuras/\n")
