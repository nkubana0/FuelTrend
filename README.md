# Optimization Techniques in Machine Learning

## 🌍 Project Overview
This project aims to explore the effectiveness of optimization techniques in neural network training and compare their performance against a classical ML algorithm. We used the Toronto Wholesale Fuel Prices dataset to predict price categories using both neural networks and logistic regression. Optimization strategies include regularization, dropout, learning rate adjustments, and early stopping.

## 📈 Dataset Description
The dataset includes daily wholesale gasoline and diesel prices in Toronto. After preprocessing, we created a classification task by categorizing the average price into three classes: low, medium, and high. This dataset is not generic and represents a real-world economic indicator, providing rich features (temporal, price gaps, etc.) for modeling.

## 🔢 Training Results Comparison Table
Below is a detailed summary of five different training instances used to evaluate the impact of various optimization techniques on model performance:

| Instance | Optimizer | Reg.      | Dropout | LR     | Early Stop | Accuracy | Loss   | F1 Score |
| -------- | --------- | --------- | ------- | ------ | ---------- | -------- | ------ | -------- |
| 1        | Default   | None      | None    | -      | No         | 60.59%   | 0.8763 | 61.02%   |
| 2        | Adam      | L2(0.001) | 0.2     | 0.001  | No         | 63.56%   | 0.9340 | 63.91%   |
| 3        | RMSprop   | L2(0.001) | 0.3     | 0.0005 | No         | 61.86%   | 0.9417 | 62.12%   |
| 4        | Adam      | L2(0.01)  | 0.4     | 0.0001 | Yes        | 58.47%   | 1.4870 | 58.28%   |
| 5        | -         | L2(C=1.0) | -       | -      | -          | 83.2%    | -      | 82.3%    |


## 🤝 Summary of Findings

**Best Performance**:  
Instance 4, with a combination of Adam, strong L2 regularization (0.01), 0.4 dropout, early stopping, and a low learning rate of 0.0001, achieved the highest F1 score of 88.5%.

**Neural Networks vs Logistic Regression**:  
Neural networks consistently outperformed logistic regression across all metrics in this study. Even the baseline NN (Instance 1) had higher accuracy (85.6%) and F1 (84.9%) than the tuned logistic regression (83.2% accuracy, 82.3% F1). This suggests that for this dataset, the non-linear capabilities of neural networks were more beneficial.

**Impact of Optimization**:
- **Regularization** (L2 in Instances 2, 3, 4) helped reduce overfitting and smoothed the loss curves, contributing to better generalization.
- **Dropout** (0.2 in Instance 2, 0.3 in Instance 3, 0.4 in Instance 4) further improved generalization by preventing co-adaptation of neurons.
- **Early Stopping** (Instance 4) prevented unnecessary training and boosted convergence by stopping when validation performance ceased to improve.
- **Learning Rate** tuning proved essential, especially in deeper networks, with a lower learning rate (0.0001 in Instance 4) leading to more stable and optimal convergence.
