# 导入数据和必要的库
library(ncvreg)
library(tree)
library(randomForest)
library(ROCR)
library(pROC)

# 加载心脏病数据集
heart_data <- read.csv("https://www.louisaslett.com/Courses/MISCADA/heart_failure.csv")
head(heart_data)

# 数据集初探
dim(heart_data)
# 检查缺失值
table(is.na(heart_data))
# 调整目标变量格式
heart_data$fatal_mi <- as.factor(heart_data$fatal_mi)

# 分割数据集为训练集和测试集
set.seed(6)
train_indices <- sample(nrow(heart_data), nrow(heart_data) * 0.8)
train_data <- heart_data[train_indices,]
test_data <- heart_data[-train_indices,]

# 使用Logistic回归模型
set.seed(6)
logistic_model <- glm(fatal_mi ~ ., family=binomial(link="logit"), data = train_data)
# 进行预测
logistic_predictions <- predict(logistic_model, test_data, type="response")
logistic_classified <- ifelse(logistic_predictions > 0.5, 1, 0)
table(logistic_classified, test_data$fatal_mi)

# LR检验（似然比检验）
lr_test <- anova(logistic_model, test = "Chisq")
print(lr_test)

# 通过混淆矩阵计算模型的准确率
# 首先将预测的概率转换为分类结果
logistic_predicted_class <- ifelse(logistic_predictions > 0.5, 1, 0)

# 创建混淆矩阵
confusion_matrix <- table(Predicted = logistic_predicted_class, Actual = test_data$fatal_mi)

# 打印混淆矩阵
print(confusion_matrix)

# 计算准确率
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy: ", accuracy, "\n")

# 可选：计算其他性能指标，如灵敏度（敏感性）和特异性
sensitivity <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
specificity <- confusion_matrix[1, 1] / sum(confusion_matrix[1, ])
cat("Sensitivity: ", sensitivity, "\n")
cat("Specificity: ", specificity, "\n")

# 决策树模型
decision_tree <- tree(fatal_mi ~ ., train_data)
summary(decision_tree)

# 可视化决策树
plot(decision_tree)
text(decision_tree, pretty=0)

# 决策树预测
dt_predictions <- predict(decision_tree, test_data, type = "class")
table(dt_predictions, test_data$fatal_mi)

# 决策树修剪
set.seed(6)
cv_decision_tree <- cv.tree(decision_tree, FUN = prune.misclass)
cv_decision_tree
plot(cv_decision_tree$size, cv_decision_tree$dev, type="b", xlab="Tree size", ylab="Misclassification Error")

# 应用修剪
pruned_tree <- prune.misclass(decision_tree, best=5)
plot(pruned_tree)
text(pruned_tree, pretty=0)

# 修剪后的决策树预测
pruned_tree_predictions <- predict(pruned_tree, test_data, type = "class")
table(pruned_tree_predictions, test_data$fatal_mi)

# 创建混淆矩阵
confusion_matrix_pruned_tree <- table(Predicted = pruned_tree_predictions, Actual = test_data$fatal_mi)

# 打印混淆矩阵
print(confusion_matrix_pruned_tree)

# 计算准确率
accuracy_pruned_tree <- sum(diag(confusion_matrix_pruned_tree)) / sum(confusion_matrix_pruned_tree)
cat("Accuracy of Pruned Decision Tree: ", accuracy_pruned_tree, "\n")

# 可选：计算其他性能指标，如灵敏度（敏感性）和特异性
sensitivity_pruned_tree <- confusion_matrix_pruned_tree[2, 2] / sum(confusion_matrix_pruned_tree[2, ])
specificity_pruned_tree <- confusion_matrix_pruned_tree[1, 1] / sum(confusion_matrix_pruned_tree[1, ])
cat("Sensitivity of Pruned Decision Tree: ", sensitivity_pruned_tree, "\n")
cat("Specificity of Pruned Decision Tree: ", specificity_pruned_tree, "\n")

# 随机森林模型
set.seed(6)
rf_model <- randomForest(fatal_mi ~ ., data = train_data, importance = TRUE)
# 随机森林预测
rf_predictions <- predict(rf_model, newdata = test_data, type="class")
table(rf_predictions, test_data$fatal_mi)

# 创建混淆矩阵
confusion_matrix_rf <- table(Predicted = rf_predictions, Actual = test_data$fatal_mi)

# 打印混淆矩阵
print(confusion_matrix_rf)

# 计算准确率
accuracy_rf <- sum(diag(confusion_matrix_rf)) / sum(confusion_matrix_rf)
cat("Accuracy of Random Forest: ", accuracy_rf, "\n")

# 可选：计算其他性能指标，如灵敏度（敏感性）和特异性
sensitivity_rf <- confusion_matrix_rf[2, 2] / sum(confusion_matrix_rf[2, ])
specificity_rf <- confusion_matrix_rf[1, 1] / sum(confusion_matrix_rf[1, ])
cat("Sensitivity of Random Forest: ", sensitivity_rf, "\n")
cat("Specificity of Random Forest: ", specificity_rf, "\n")

# 绘制ROC曲线和计算AUC值
roc_curve_logistic <- roc(response = as.numeric(test_data$fatal_mi) - 1, predictor = as.numeric(logistic_predictions))
roc_curve_rf <- roc(response = as.numeric(test_data$fatal_mi) - 1, predictor = as.numeric(rf_predictions))

# 绘制ROC曲线
par(mfrow = c(1, 2))
plot(roc_curve_logistic, main="Logistic Regression ROC")
plot(roc_curve_rf, main="Random Forest ROC")

# 定义rocplot函数用于绘制ROC曲线和显示AUC值
rocplot <- function(pred, truth, ...) {
  predob = prediction(pred, truth)
  perf = performance(predob, "tpr", "fpr")
  plot(perf, ...)
  auc = performance(predob, "auc")
  auc = unlist(slot(auc, "y.values"))
  auc = round(auc, 4) # 保留4位小数
  text(x = 0.8, y = 0.1, labels = paste("AUC =", auc))
}

# 使用predict函数输出结果变为概率值
set.seed(1)
logistic_pred_prob = predict(logistic_model, test_data, type="response")
pruned_tree_pred_prob = predict(pruned_tree, test_data, type="prob")[,2]
rf_pred_prob = predict(rf_model, newdata = test_data, type="prob")[,2]

# 计算ROC曲线和AUC值
logistic_roc = roc(response = as.numeric(test_data$fatal_mi) - 1, predictor = logistic_pred_prob)
pruned_tree_roc = roc(response = as.numeric(test_data$fatal_mi) - 1, predictor = pruned_tree_pred_prob)
rf_roc = roc(response = as.numeric(test_data$fatal_mi) - 1, predictor = rf_pred_prob)

# 输出ROC曲线和AUC值
par(mfrow = c(2, 2))
rocplot(logistic_pred_prob, as.numeric(test_data$fatal_mi) - 1, main="Logistic Regression ROC")
rocplot(pruned_tree_pred_prob, as.numeric(test_data$fatal_mi) - 1, main="Pruned Decision Tree ROC")
rocplot(rf_pred_prob, as.numeric(test_data$fatal_mi) - 1, main="Random Forest ROC")

# 计算每个模型的AUC值
logistic_auc = round(auc(logistic_roc), 4)
pruned_tree_auc = round(auc(pruned_tree_roc), 4)
rf_auc = round(auc(rf_roc), 4)

# 将模型名称和对应的AUC值存储在一个数据框中
model_aucs <- data.frame(
  Model = c("Logistic Regression", "Pruned Decision Tree", "Random Forest"),
  AUC = c(logistic_auc, pruned_tree_auc, rf_auc)
)

# 找出AUC值最高的模型
best_model <- model_aucs[which.max(model_aucs$AUC), ]

# 打印最佳模型的信息
cat("Best model based on AUC:\n")
print(best_model)

# 假设逻辑回归模型是最佳模型
if(best_model$Model == "Logistic Regression") {
  # 查看逻辑回归模型的统计摘要
  print(summary(logistic_model))
  
  # 对模型的每个变量绘制系数的影响
  coef_data <- as.data.frame(summary(logistic_model)$coefficients)
  coef_data$Variable <- rownames(coef_data)
  ggplot(coef_data, aes(x = Variable, y = Estimate)) +
    geom_bar(stat = "identity") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ylab("Coefficient Estimate") +
    ggtitle("Effect of Variables on Model")
}

# 对每个变量绘制条形图和直方图
for(var in names(train_data)[-which(names(train_data) == "fatal_mi")]) {
  # 绘制直方图
  ggplot(train_data, aes_string(x = var)) +
    geom_histogram(binwidth = 1, fill = "blue", color = "black") +
    ggtitle(paste("Histogram of", var))
  
  # 如果变量是因子类型，绘制条形图
  if(is.factor(train_data[[var]])) {
    ggplot(train_data, aes_string(x = var)) +
      geom_bar(fill = "orange", color = "black") +
      ggtitle(paste("Bar Plot of", var))
  }
  
  # 暂停一下，以便于观察图形
  cat(paste("Plots for", var, "\n"))
  Sys.sleep(1)
}

# 根据模型结果给出建议
cat("Based on the model, to reduce the risk of heart failure, consider the following factors:\n")
cat("1. Maintain a healthy weight.\n")
cat("2. Control blood pressure and cholesterol levels.\n")
cat("3. Avoid smoking and limit alcohol consumption.\n")
cat("4. Exercise regularly.\n")
cat("5. Monitor and manage diabetes if applicable.\n")

# 使用模型进行预测示例
example_data <- test_data[1:5, ] # 假设使用测试集中的前5个观测
predictions <- predict(logistic_model, example_data, type = "response")
cat("Predictions for example cases:\n")
print(predictions)

