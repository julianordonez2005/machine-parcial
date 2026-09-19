# Instalación de dependencias — Parcial IIND3118
paquetes <- c(
  "MASS", "cluster", "caret", "class",        # base, KNN, CV
  "glmnet",                                   # Ridge, Lasso, Elastic Net
  "leaps",                                    # selección de variables
  "car",                                      # VIF (multicolinealidad)
  "dbscan",                                   # DBSCAN y LOF
  "fpc", "clusterCrit",                       # métricas de clustering
  "isotree",                                  # Isolation Forest
  "kernlab",                                  # Kernel PCA
  "Rtsne",                                    # t-SNE
  "psych",                                    # análisis factorial, KMO
  "vegan",                                    # Isomap (alternativa CRAN)
  "ISLR",                                     # datasets del libro
  "pROC"                                      # curva ROC
)
faltan <- paquetes[!(paquetes %in% installed.packages()[, "Package"])]
if (length(faltan)) install.packages(faltan, repos = "https://cloud.r-project.org")
cat("Listo.\n")
