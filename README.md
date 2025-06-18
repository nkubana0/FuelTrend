# Optimization Techniques in Machine Learning

## 🌍 Project Overview
This project aims to explore the effectiveness of optimization techniques in neural network training and compare their performance against a classical ML algorithm. We used the Toronto Wholesale Fuel Prices dataset to predict price categories using both neural networks and logistic regression. Optimization strategies include regularization, dropout, learning rate adjustments, and early stopping.

## 📈 Dataset Description
The dataset includes daily wholesale gasoline and diesel prices in Toronto. After preprocessing, we created a classification task by categorizing the average price into three classes: low, medium, and high. This dataset is not generic and represents a real-world economic indicator, providing rich features (temporal, price gaps, etc.) for modeling.

## 🔢 Training Results Comparison Table

| Instance              | Optimizer   | Regularizer | Dropout | LR      | Early Stopping | Layers (Hidden) | Accuracy | Precision | Recall | F1-Score | Loss Curve |
| :-------------------- | :---------- | :---------- | :------ | :------ | :------------- | :-------------- | :------- | :-------- | :----- | :------- | :--------- |
| **1 (Simple NN)**     | Default     | None        | None    | -       | No             | 3               | 85.6%    | 85.1%     | 84.7%  | 84.9%    | Curved     |
| **2 (Optimized NN)**  | Adam        | L2(0.001)   | 0.2     | 0.001   | No             | 4               | 87.3%    | 86.9%     | 86.7%  | 86.8%    | Smoother   |
| **3 (Optimized NN)**  | RMSprop     | L2(0.001)   | 0.3     | 0.0005  | No             | 4               | 88.0%    | 87.5%     | 87.3%  | 87.4%    | Smooth     |
| **4 (Optimized NN)**  | Adam        | L2(0.01)    | 0.4     | 0.0001  | Yes            | 4               | 89.1%    | 88.6%     | 88.4%  | 88.5%    | Optimal    |
| **5 (LogReg)**        | -           | L2(C=1.0)   | -       | -       | -              | -               | 83.2%    | 82.7%     | 82.0%  | 82.3%    | -          |

*Note: Actual values taken from validation set performance metrics.*

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

## 🎯 Instructions to Run the Notebook

1. **Clone or download** the GitHub repository.
2. **Install required libraries**: Ensure you have the following packages installed:
   ```bash
   pip install tensorflow scikit-learn pandas numpy matplotlib seaborn joblib
