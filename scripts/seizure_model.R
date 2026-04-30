# LOAD LIBRARIES
packages <- c(
  "tidyverse",
  "caret",
  "ranger",
  "xgboost",
  "pROC",
  "knitr",
  "here"
)

for(p in packages){
  if (!require(p, character.only = TRUE)) {
    install.packages(p, dependencies = TRUE)
  }
  library(p, character.only = TRUE)
}

#  LOAD DATA
df <- read.csv(here::here("data", "BEED_Data.csv"))

#  SAFETY CHECK
if (!exists("df") || nrow(df) == 0) {
  stop("Data load failed. Ensure BEED_Data.csv is in the /data folder!")
}

#  Preprocessing
df <- na.omit(df)

# Convert to binary (Seizure vs Non-Seizure)
df$y <- ifelse(df$y == 1, 1, 0)
df$y <- as.factor(df$y)

#  EDA (Data Visualization)
#  Class Distribution
ggplot(df, aes(x = y)) +
  geom_bar(fill = "steelblue") +
  ggtitle("Class Distribution (Seizure vs Non-Seizure)")

# Feature Distribution
ggplot(df, aes(x = X1, fill = y)) +
  geom_histogram(bins = 30, alpha = 0.6, position = "identity") +
  ggtitle("Distribution of Feature X1")

#  Boxplot
ggplot(df, aes(x = y, y = X2, fill = y)) +
  geom_boxplot() +
  ggtitle("Boxplot of Feature X2")

#  Correlation Heatmap
cor_matrix <- cor(df[sapply(df, is.numeric)])

# Convert to dataframe
cor_df <- as.data.frame(as.table(cor_matrix))

# Plot heatmap with values
heatmap_plot <- ggplot(cor_df, aes(Var1, Var2, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = round(Freq, 2)), size = 3) +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  ggtitle("Correlation Heatmap with Values") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Display
heatmap_plot

# Save
ggsave("visuals/heatmap.png", plot = heatmap_plot, width = 8, height = 6)

#  Train-Test Split
set.seed(123)
trainIndex <- createDataPartition(df$y, p = 0.8, list = FALSE)
train <- df[trainIndex,]
test <- df[-trainIndex,]

#  Logistic Regression
model_lr <- train(y ~ ., data = train, method = "glm", family = "binomial")
pred_lr <- predict(model_lr, test)
cm_lr <- confusionMatrix(pred_lr, test$y)
print(cm_lr)

#  Random Forest
model_rf <- ranger(y ~ ., data = train, num.trees = 100)
pred_rf <- predict(model_rf, test)$predictions
cm_rf <- confusionMatrix(pred_rf, test$y)
print(cm_rf)

# SAVE CONFUSION MATRIX 
png("visuals/confusion_matrix.png", width=800, height=600)

fourfoldplot(cm_rf$table,
             color = c("red","green"),
             main = "Confusion Matrix - Random Forest")

dev.off()

# XGBoost
# XGBoost (Correct Version)

train_matrix <- model.matrix(y ~ . -1, train)
test_matrix  <- model.matrix(y ~ . -1, test)

train_label <- as.numeric(train$y) - 1

# Convert to DMatrix
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest  <- xgb.DMatrix(data = test_matrix)

# Parameters
params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss"
)

# Train model
model_xgb <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 50,
  verbose = 0
)

# Prediction
pred_probs <- predict(model_xgb, dtest)
pred_xgb <- as.factor(ifelse(pred_probs > 0.5, 1, 0))

# Confusion Matrix
cm_xgb <- confusionMatrix(pred_xgb, test$y)
print(cm_xgb)

#  Metrics Function
get_metrics <- function(cm){
  c(
    Accuracy = as.numeric(cm$overall["Accuracy"]),
    Precision = as.numeric(cm$byClass["Precision"]),
    Recall = as.numeric(cm$byClass["Recall"]),
    F1 = as.numeric(cm$byClass["F1"])
  )
}

#  Comparison Table
results <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "XGBoost"),
  rbind(get_metrics(cm_lr), get_metrics(cm_rf), get_metrics(cm_xgb))
)

kable(results, caption = "Model Comparison Table")

#  Result Visualization
recall_plot <- ggplot(results, aes(x = Model, y = Recall)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  ggtitle("Model Comparison (Recall)")

ggsave("visuals/model_performance.png", plot = recall_plot, width = 8, height = 5)

# SAVE ACCURACY PLOT
accuracy_plot <- ggplot(results, aes(x = Model, y = Accuracy)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  ggtitle("Model Accuracy Comparison")

accuracy_plot

ggsave("visuals/accuracy_plot.png", plot = accuracy_plot, width = 8, height = 5)

# SAVE RESULTS TABLE
write.csv(results, "visuals/model_results.csv", row.names = FALSE)
