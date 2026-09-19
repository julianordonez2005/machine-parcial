# ============================================================================
# PLANTILLA: PROBLEMA DE REGRESIÓN en R — copiar y pegar
# Cambia solo la primera línea y el nombre de la columna respuesta.
# ============================================================================
library(glmnet); library(caret); library(leaps)

# ── ✏️ CAMBIA SOLO ESTO ─────────────────────────────────────────────────
df     <- read.csv("data/ejemplo_regresion.csv")
TARGET <- "target"
# ────────────────────────────────────────────────────────────────────────

df <- na.omit(df)
X_df <- as.data.frame(model.matrix(~ . - 1, data = df[, setdiff(names(df), TARGET), drop=FALSE]))
X  <- as.matrix(X_df); y <- as.numeric(df[[TARGET]])
Xs <- scale(X)

set.seed(0)
idx <- sample(seq_len(nrow(X)), floor(.70*nrow(X)))     # 70% train
Xtrain <- X[idx,]; Xtest <- X[-idx,]; ytrain <- y[idx]; ytest <- y[-idx]
Xs_train <- Xs[idx,]; Xs_test <- Xs[-idx,]

res <- c()

# 1. Regresión lineal
m1 <- lm(ytrain ~ ., data=data.frame(Xtrain, ytrain))
res["Lineal"] <- mean((ytest - predict(m1, data.frame(Xtest)))^2)

# 2. KNN calibrado por CV   ⚠️ con Xs
cv <- train(x=Xs_train, y=ytrain, method="knn", trControl=trainControl(method="cv", number=10),
            tuneGrid=data.frame(k=1:40))
res[paste0("KNN (k=", cv$bestTune$k, ")")] <- mean((ytest - predict(cv, Xs_test))^2)

# 3. Ridge / Lasso / Elastic Net   ⚠️ con Xs
#    alpha: 0 = Ridge, 1 = Lasso, 0.3 = Elastic Net
for (info in list(list(0,"Ridge"), list(1,"Lasso"), list(0.3,"ElasticNet"))) {
  cvm <- cv.glmnet(Xs_train, ytrain, alpha=info[[1]], nfolds=10)
  mm  <- glmnet(Xs_train, ytrain, alpha=info[[1]], lambda=cvm$lambda.min)
  res[info[[2]]] <- mean((ytest - as.numeric(predict(mm, Xs_test)))^2)
  cat(sprintf("%-12s lambda=%.5f  coefs en cero: %d\n",
              info[[2]], cvm$lambda.min, sum(coef(mm)[-1]==0)))
}

# 4. Selección de variables (forward)
regfit <- regsubsets(ytrain ~ ., data=data.frame(Xtrain, ytrain),
                     nvmax=ncol(Xtrain), method="forward")
k_best <- which.min(summary(regfit)$cp)
vars   <- names(coef(regfit, k_best))[-1]
mf <- lm(as.formula(paste("ytrain ~", paste(vars, collapse="+"))),
         data=data.frame(Xtrain, ytrain))
res[paste0("Forward (", k_best, " vars)")] <- mean((ytest - predict(mf, data.frame(Xtest)))^2)

# ── Resultados ──────────────────────────────────────────────────────────
res <- sort(res)
print(round(res, 3))
cat(sprintf("\nMejor: %s (MSE = %.3f)\n", names(res)[1], res[1]))

barplot(rev(res), horiz=TRUE, las=1, cex.names=.8, col="steelblue",
        xlab="MSE en test", main="Comparación de modelos")
