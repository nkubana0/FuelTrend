# Fuel Price Category Prediction

## 1. Project Overview

This project aims to predict the future price category of wholesale gasoline in Toronto based on historical gasoline and diesel prices. By analyzing trends and relationships between these fuel types, the project develops and evaluates various machine learning models, including Logistic Regression and several Neural Network configurations, to classify gasoline prices into low, medium, or high categories. The goal is to identify the most effective model for predicting price shifts.

## 2. Dataset Used

The dataset utilized in this project is derived from publicly available wholesale gasoline and diesel price data for Toronto. Specifically, it combines:
- Wholesale Gasoline Prices: `https://prod-energy-fuel-prices.s3.amazonaws.com/wholesalegasolineprices.csv`
- Wholesale Diesel Prices: `https://prod-energy-fuel-prices.s3.amazonaws.com/wholesaledieselprices.csv`

This dataset is suitable for classification as it allows for the categorization of gasoline prices into distinct levels. It is not generic, focusing on specific fuel price data, and is aligned with the task of forecasting fuel price trends. The features engineered (`month`, `year`, and `diesel_price`) directly relate to and influence the target variable, `gasoline_price` converted into `price_category`. The dataset contains sufficient volume and variety to train robust models.

## 3. Discussion of Findings

### Model Training and Optimization Results

This section details the performance of the implemented models, including a classical Machine Learning algorithm (Logistic Regression), a simple Neural Network, and several optimized Neural Network configurations. Metrics reported include Accuracy, Precision (macro), Recall (macro), and F1-score (macro) on the validation set.

**Note on Metrics**: Precision, Recall, and F1-score are reported as macro averages to account for class imbalance, ensuring that performance across all categories is considered equally important. `UndefinedMetricWarning` for Precision indicates that for some classes, there were no predicted samples, leading to a precision of 0.0 for that specific class.

| Training Instance           | Optimizer Used   | Regularizer Used | Epochs | Early Stopping | Hidden Layers (units) | Learning Rate | Accuracy | F1-score | Recall  | Precision |
|----------------------------|------------------|-----------------|--------|----------------|----------------------|---------------|----------|----------|---------|-----------|
| **Logistic Regression**     | N/A (lbfgs)      | L2 (C=1.0)      | 500    | No             | N/A                  | N/A           | 0.7246   | 0.7231   | 0.7225  | 0.7239    |
| **NN Simple**               | Adam (default)   | None            | 10     | No             | 3 (128, 64, 32)      | Default       | 0.4407   | 0.3441   | 0.4417  | 0.4571    |
| **NN Optimized Instance 1** | Adam (default)   | None            | 20     | No             | 3 (256, 128, 64)     | Default       | 0.6398   | 0.5148   | 0.6371  | 0.4357    |
| **NN Optimized Instance 2** | Adam             | None            | 50     | Yes (patience=5) | 3 (256, 128, 64)   | 0.001         | 0.3305   | 0.1656   | 0.3333  | 0.1102    |
| **NN Optimized Instance 3** | RMSprop          | L2 (0.001)      | 75     | Yes (patience=5) | 3 (256, 128, 64)   | 0.0005        | 0.6144   | 0.5854   | 0.6113  | 0.6509    |
| **NN Optimized Instance 4** | Adam             | L2 (0.01)       | 100    | Yes (patience=5) | 3 (256, 128, 64)   | 0.0001        | 0.3305   | 0.1656   | 0.3333  | 0.1102    |

### Summary of Which Combination Worked Better

Based on the validation metrics, the **Logistic Regression model** showed the best overall performance with an accuracy of approximately 0.7246. Among the Neural Network models, **NN Optimized Instance 3** performed the best, achieving an accuracy of 0.6144. This instance utilized the RMSprop optimizer with a learning rate of 0.0005, L2 regularization (0.001), and early stopping.

Several optimized NN instances (2 and 4) performed poorly, likely due to aggressive regularization or suboptimal learning rates, causing underfitting or getting stuck in local minima. The simple NN also performed significantly worse, emphasizing the importance of optimization.

### Which Implementation Worked Better (ML Algorithm vs. Neural Network)?

In this case, the **Logistic Regression** model outperformed all Neural Network models, achieving higher accuracy, precision, recall, and F1-score. The logistic regression's hyperparameters were:

- `C=1.0` (inverse regularization strength)
- `penalty='l2'`
- `solver='lbfgs'`
- `max_iter=500`

This suggests a simpler linear model fits the data well, possibly due to the inherent linearity of features with the target classes. Neural networks require careful tuning and can sometimes underperform on tabular data like this.

## 4. Instructions for Running the Notebook and Loading the Best Model

### Running the Notebook (`notebook.ipynb`)

1. **Environment Setup**:  
   Ensure you have a Python environment with required libraries:
   ```bash
   pip install pandas numpy scikit-learn tensorflow joblib
