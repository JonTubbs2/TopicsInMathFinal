library(plyr)
library(readr)
library(dplyr)
library(caret)
library(ggplot2)
library(repr)

dat = read.csv("Life_expectancy.csv")
dat = na.omit(dat)
glimpse(dat)


# Data Portioning
set.seed(100) 

index = sample(1:nrow(dat), 0.7*nrow(dat)) 

train = dat[index,] # Create the training data 
test = dat[-index,] # Create the test data

dim(train)
dim(test)

# Scaling Numeric Features
cols = c('Life.expectancy', 'Alcohol', 'BMI', 'Total.expenditure', 'GDP')

pre_proc_val = preProcess(train[,cols], method = c("center", "scale"))

train[,cols] = predict(pre_proc_val, train[,cols])
test[,cols] = predict(pre_proc_val, test[,cols])


summary(train)


# Regularization
cols_reg = c('Life.expectancy', 'Alcohol', 'BMI', 'Total.expenditure', 'GDP')

dummies = dummyVars(Life.expectancy ~ ., data = dat[,cols_reg])

train_dummies = predict(dummies, newdata = train[,cols_reg])

test_dummies = predict(dummies, newdata = test[,cols_reg])

print(dim(train_dummies)); print(dim(test_dummies))


# Ridge Regression
library(glmnet)

x = as.matrix(train_dummies)
y_train = train$Life.expectancy

x_test = as.matrix(test_dummies)
y_test = test$Life.expectancy

lambdas = 10^seq(2, -3, by = -.1)
ridge_reg = glmnet(x, y_train, nlambda = 25, alpha = 0, family = 'gaussian', lambda = lambdas)

summary(ridge_reg)

cv_ridge = cv.glmnet(x, y_train, alpha = 0, lambda = lambdas)
optimal_lambda = cv_ridge$lambda.min
optimal_lambda

# Compute R^2 from true and predicted values
eval_results = function(true, predicted, df) {
  SSE = sum((predicted - true)^2)
  SST = sum((true - mean(true))^2)
  R_square = 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
  
  # Model performance metrics
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
  
}

# Prediction and evaluation on train data
predictions_train = predict(ridge_reg, s = optimal_lambda, newx = x)
eval_results(y_train, predictions_train, train)

# Prediction and evaluation on test data
predictions_test <- predict(ridge_reg, s = optimal_lambda, newx = x_test)
eval_results(y_test, predictions_test, test)


# Lasso Regression
lambdas = 10^seq(2, -3, by = -.1)

# Setting alpha = 1 implements lasso regression
lasso_reg = cv.glmnet(x, y_train, alpha = 1, lambda = lambdas, standardize = TRUE, nfolds = 5)

# Best 
lambda_best = lasso_reg$lambda.min 
lambda_best

lasso_model = glmnet(x, y_train, alpha = 1, lambda = lambda_best, standardize = TRUE)

predictions_train = predict(lasso_model, s = lambda_best, newx = x)
eval_results(y_train, predictions_train, train)

predictions_test = predict(lasso_model, s = lambda_best, newx = x_test)
eval_results(y_test, predictions_test, test)


# Elastic Net Regression

# Set training control
train_cont = trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 5,
                           search = "random",
                           verboseIter = TRUE)

# Train the model
elastic_reg = train(Life.expectancy ~ .,
                     data = train,
                     method = "glmnet",
                     preProcess = c("center", "scale"),
                     tuneLength = 10,
                     trControl = train_cont)


# Best tuning parameter
elastic_reg$bestTune

# Make predictions on training set
predictions_train = predict(elastic_reg, x)
eval_results(y_train, predictions_train, train) 

# Make predictions on test set
predictions_test = predict(elastic_reg, x_test)
eval_results(y_test, predictions_test, test)

