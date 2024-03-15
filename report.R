
library(ncvreg)
library(tree)
library(randomForest)
library(ROCR)
library(pROC)

heart_data <- read.csv("https://www.louisaslett.com/Courses/MISCADA/heart_failure.csv")
head(heart_data)


dim(heart_data)

table(is.na(heart_data))

heart_data$fatal_mi <- as.factor(heart_data$fatal_mi)


set.seed(6)
train_indices <- sample(nrow(heart_data), nrow(heart_data) * 0.8)
train_data <- heart_data[train_indices,]
test_data <- heart_data[-train_indices,]


set.seed(6)
logistic_model <- glm(fatal_mi ~ ., family=binomial(link="logit"), data = train_data)

logistic_predictions <- predict(logistic_model, test_data, type="response")
logistic_classified <- ifelse(logistic_predictions > 0.5, 1, 0)
table(logistic_classified, test_data$fatal_mi)


lr_test <- anova(logistic_model, test = "Chisq")
print(lr_test)


logistic_predicted_class <- ifelse(logistic_predictions > 0.5, 1, 0)


confusion_matrix <- table(Predicted = logistic_predicted_class, Actual = test_data$fatal_mi)


print(confusion_matrix)


accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy: ", accuracy, "\n")


sensitivity <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
specificity <- confusion_matrix[1, 1] / sum(confusion_matrix[1, ])
cat("Sensitivity: ", sensitivity, "\n")
cat("Specificity: ", specificity, "\n")


decision_tree <- tree(fatal_mi ~ ., train_data)
summary(decision_tree)

plot(decision_tree)
text(decision_tree, pretty=0)


dt_predictions <- predict(decision_tree, test_data, type = "class")
table(dt_predictions, test_data$fatal_mi)


set.seed(6)
cv_decision_tree <- cv.tree(decision_tree, FUN = prune.misclass)
cv_decision_tree
plot(cv_decision_tree$size, cv_decision_tree$dev, type="b", xlab="Tree size", ylab="Misclassification Error")


pruned_tree <- prune.misclass(decision_tree, best=5)
plot(pruned_tree)
text(pruned_tree, pretty=0)


pruned_tree_predictions <- predict(pruned_tree, test_data, type = "class")
table(pruned_tree_predictions, test_data$fatal_mi)


confusion_matrix_pruned_tree <- table(Predicted = pruned_tree_predictions, Actual = test_data$fatal_mi)


print(confusion_matrix_pruned_tree)


accuracy_pruned_tree <- sum(diag(confusion_matrix_pruned_tree)) / sum(confusion_matrix_pruned_tree)
cat("Accuracy of Pruned Decision Tree: ", accuracy_pruned_tree, "\n")


sensitivity_pruned_tree <- confusion_matrix_pruned_tree[2, 2] / sum(confusion_matrix_pruned_tree[2, ])
specificity_pruned_tree <- confusion_matrix_pruned_tree[1, 1] / sum(confusion_matrix_pruned_tree[1, ])
cat("Sensitivity of Pruned Decision Tree: ", sensitivity_pruned_tree, "\n")
cat("Specificity of Pruned Decision Tree: ", specificity_pruned_tree, "\n")


set.seed(6)
rf_model <- randomForest(fatal_mi ~ ., data = train_data, importance = TRUE)

rf_predictions <- predict(rf_model, newdata = test_data, type="class")
table(rf_predictions, test_data$fatal_mi)


confusion_matrix_rf <- table(Predicted = rf_predictions, Actual = test_data$fatal_mi)


print(confusion_matrix_rf)


accuracy_rf <- sum(diag(confusion_matrix_rf)) / sum(confusion_matrix_rf)
cat("Accuracy of Random Forest: ", accuracy_rf, "\n")


sensitivity_rf <- confusion_matrix_rf[2, 2] / sum(confusion_matrix_rf[2, ])
specificity_rf <- confusion_matrix_rf[1, 1] / sum(confusion_matrix_rf[1, ])
cat("Sensitivity of Random Forest: ", sensitivity_rf, "\n")
cat("Specificity of Random Forest: ", specificity_rf, "\n")


roc_curve_logistic <- roc(response = as.numeric(test_data$fatal_mi) - 1, predictor = as.numeric(logistic_predictions))
roc_curve_rf <- roc(response = as.numeric(test_data$fatal_mi) - 1, predictor = as.numeric(rf_predictions))


par(mfrow = c(1, 2))
plot(roc_curve_logistic, main="Logistic Regression ROC")
plot(roc_curve_rf, main="Random Forest ROC")


rocplot <- function(pred, truth, ...) {
  predob = prediction(pred, truth)
  perf = performance(predob, "tpr", "fpr")
  plot(perf, ...)
  auc = performance(predob, "auc")
  auc = unlist(slot(auc, "y.values"))
  auc = round(auc, 4) 
  text(x = 0.8, y = 0.1, labels = paste("AUC =", auc))
}


set.seed(1)
logistic_pred_prob = predict(logistic_model, test_data, type="response")
pruned_tree_pred_prob = predict(pruned_tree, test_data, type="prob")[,2]
rf_pred_prob = predict(rf_model, newdata = test_data, type="prob")[,2]


logistic_roc = roc(response = as.numeric(test_data$fatal_mi) - 1, predictor = logistic_pred_prob)
pruned_tree_roc = roc(response = as.numeric(test_data$fatal_mi) - 1, predictor = pruned_tree_pred_prob)
rf_roc = roc(response = as.numeric(test_data$fatal_mi) - 1, predictor = rf_pred_prob)


par(mfrow = c(2, 2))
rocplot(logistic_pred_prob, as.numeric(test_data$fatal_mi) - 1, main="Logistic Regression ROC")
rocplot(pruned_tree_pred_prob, as.numeric(test_data$fatal_mi) - 1, main="Pruned Decision Tree ROC")
rocplot(rf_pred_prob, as.numeric(test_data$fatal_mi) - 1, main="Random Forest ROC")


logistic_auc = round(auc(logistic_roc), 4)
pruned_tree_auc = round(auc(pruned_tree_roc), 4)
rf_auc = round(auc(rf_roc), 4)


model_aucs <- data.frame(
  Model = c("Logistic Regression", "Pruned Decision Tree", "Random Forest"),
  AUC = c(logistic_auc, pruned_tree_auc, rf_auc)
)


best_model <- model_aucs[which.max(model_aucs$AUC), ]


cat("Best model based on AUC:\n")
print(best_model)


if(best_model$Model == "Logistic Regression") {

  print(summary(logistic_model))
  

  coef_data <- as.data.frame(summary(logistic_model)$coefficients)
  coef_data$Variable <- rownames(coef_data)
  ggplot(coef_data, aes(x = Variable, y = Estimate)) +
    geom_bar(stat = "identity") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ylab("Coefficient Estimate") +
    ggtitle("Effect of Variables on Model")
}


for(var in names(train_data)[-which(names(train_data) == "fatal_mi")]) {

  ggplot(train_data, aes_string(x = var)) +
    geom_histogram(binwidth = 1, fill = "blue", color = "black") +
    ggtitle(paste("Histogram of", var))
  

  if(is.factor(train_data[[var]])) {
    ggplot(train_data, aes_string(x = var)) +
      geom_bar(fill = "orange", color = "black") +
      ggtitle(paste("Bar Plot of", var))
  }
  

  cat(paste("Plots for", var, "\n"))
  Sys.sleep(1)
}


cat("Based on the model, to reduce the risk of heart failure, consider the following factors:\n")
cat("1. Maintain a healthy weight.\n")
cat("2. Control blood pressure and cholesterol levels.\n")
cat("3. Avoid smoking and limit alcohol consumption.\n")
cat("4. Exercise regularly.\n")
cat("5. Monitor and manage diabetes if applicable.\n")


example_data <- test_data[1:5, ] 
predictions <- predict(logistic_model, example_data, type = "response")
cat("Predictions for example cases:\n")
print(predictions)

