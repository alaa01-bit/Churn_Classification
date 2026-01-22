# Churn Prediction Notebook Documentation & Analysis

This document provides a detailed explanation of the `churn.ipynb` notebook, interpreting the results, explaining the models used, and offering a guide on how to tune them.

## 1. Notebook Walkthrough & Output Explanation

### **Step 1: Data Preprocessing & Exploration**
*   **Action**: The notebook starts by importing necessary libraries (pandas, sklearn) and loading the dataset `customer_churn_dataset.xlsx`.
*   **Data Structure**: The dataset contains customer details like `age`, `tenure`, `monthly_charges`, and service details (`internet_service`, `contract_type`).
*   **Missing Values**:
    *   The code checks for missing values (`isnull().sum()`).
    *   **Finding**: `internet_service` (~20.4% missing), `monthly_data_gb` (~5.2% missing), and `tech_support` (~4.7% missing) have gaps.
*   **Feature Statistics**:
    *   `df.describe()` shows the distribution of numerical data.
    *   **Insight**: The target variable `churn` is imbalanced (approx. 74% retained vs. 26% churned). This is why handling class imbalance becomes important later.

### **Step 2: Data Cleaning & Feature Engineering**
*   **Action**: A `Pipeline` is created to handle missing data automatically.
    *   **Numerical Data**: Missing values are replaced with the **median** (to be robust against outliers) and scaled using `StandardScaler` (to normalize ranges, e.g., aiming for mean=0, std=1).
    *   **Categorical Data**: Missing values are replaced with the **most frequent** value and then One-Hot Encoded (converting text categories like "Fiber" into binary 1/0 columns).
*   **Why this step?**: Machine learning models generally cannot handle missing data or text strings directly. They need clean, numerical input.

### **Step 3: Baseline Model - Logistic Regression**
*   **Model**: `LogisticRegression(class_weight='balanced')`
*   **Output Analysis**:
    *   **Accuracy**: ~76%
    *   **ROC-AUC**: ~0.85
    *   **Recall (Class 1 - Churn)**: ~0.74. This is a critical metric. It means the model correctly identified 74% of the people who actually churned.
    *   **Precision (Class 1)**: ~0.92. Of the people it predicted would churn, 92% actually did.
*   **Why this step?**: Logistic Regression is the standard "baseline." It's simple, fast, and interpretable. If a complex model can't beat this, you don't need the complex model. The `class_weight='balanced'` parameter was crucial here to force the model to pay attention to the minority class (churners).

### **Step 4: Advanced Model - CatBoost Classifier**
*   **Model**: `CatBoostClassifier`
*   **Preprocessing Change**: Unlike the first model, CatBoost handles missing values and categorical data natively. The code simply fills NaNs with "Missing" and passes the indices of categorical columns directly to the model.
*   **Output Analysis**:
    *   **Accuracy**: ~83.5% (Significant improvement)
    *   **ROC-AUC**: ~0.89 (Stronger separation between churners and non-churners)
    *   **Recall (Class 1)**: ~0.80. It captured more churners than the Logistic Regression while maintaining high precision.
*   **Why this step?**: After establishing a baseline, a Gradient Boosting model (CatBoost) was chosen to capture non-linear relationships and interactions between features (e.g., perhaps high charges only cause churn if tenure is low).

---

## 2. Model Explanations (Simplified)

### **Model 1: Logistic Regression**
*   **Analogy**: Think of this as a "Scorecard" or a weighted sum.
*   **How it works**: It looks at each feature independently and assigns it a positive or negative score.
    *   *Example*: Being on a "Month-to-month" contract might add +5 points to your "Churn Score", while having "Tech Support" might subtract -3 points.
    *   If the total score crosses a threshold, the model predicts "Churn".
*   **Why chosen**: It is the best starting point. It tells you *linear* relationships (e.g., "as price goes up, churn goes up").

### **Model 2: CatBoost (Categorical Boosting)**
*   **Analogy**: Think of this as a Committee of Experts (Decision Trees).
*   **How it works**: It doesn't use just one rule. It builds thousands of small "if-then" decision trees sequentially.
    *   *Tree 1* might say: "If contract is monthly, they might churn."
    *   *Tree 2* looks at the errors of Tree 1 and says: "Wait, if contract is monthly BUT they have 5 years of tenure, they stay."
    *   Each new tree corrects the mistakes of the previous ones.
*   **Why chosen**: "Cat" stands for **Cat**egorical. This model is famous for handling categories (like "Payment Method" or "Service Type") extremely well without needing messy preprocessing. It usually outperforms simpler models on tabular customer data.

---

## 3. Parameter Tuning Guide

Here are the key parameters used and what happens if you change them:

### **Logistic Regression Parameters**
| Parameter | Current Value | What it does | How to tune it |
| :--- | :--- | :--- | :--- |
| `class_weight` | `'balanced'` | Tells the model that "Churn" (minority) is more important than "Stay". | **Remove it**: Accuracy might go up, but you will miss most churners. **Keep it**: Essential for imbalanced datasets. |
| `C` | (Default 1.0) | Controls "Regularization" (preventing overfitting). | **Increase (e.g., 10, 100)**: Trust the training data more (risk of overfitting). **Decrease (e.g., 0.01)**: Be more conservative/generic. |

### **CatBoost Parameters**
| Parameter | Current Value | What it does | Impact of Changing |
| :--- | :--- | :--- | :--- |
| `iterations` | `300` | The number of trees/rounds of improvement. | **Increase (e.g., 1000)**: Model becomes smarter but slower; risk of overfitting. **Decrease**: Faster, but might underperform. |
| `learning_rate` | `0.06` | How big of a step to take in each correction. | **Increase (e.g., 0.1)**: Learns faster, might overshoot optimal solution. **Decrease (e.g., 0.01)**: Learns slowly, needs more iterations, often more accurate. |
| `depth` | `6` | How deep each decision tree acts (complexity of questions). | **Increase (e.g., 10)**: Can find very complex patterns (e.g., specific combinations of 10 features), but slower and overfits easily. **Decrease (e.g., 4)**: Good for smaller datasets to prevent memorization. |
| `auto_class_weights` | `'Balanced'` | Similar to Logistic Regression; handles imbalance. | **None**: The model might ignore churners. **Balanced**: Recommended for this dataset. |

---

## 4. Pros & Cons Comparison

| Feature | **Logistic Regression** | **CatBoost** |
| :--- | :--- | :--- |
| **Simplicity** | ✅ Very Simple. Easy to explain to stakeholders. | ❌ Complex "Black Box". Harder to explain strictly "why" a specific decision was made. |
| **Speed** | ✅ Extremely fast to train and predict. | ⚠️ Slower to train (though CatBoost is faster than other boosting models). |
| **Accuracy** | ⚠️ Average. Misses complex patterns. | ✅ High. State-of-the-art for tabular data. |
| **Data Prep** | ❌ Needs careful scaling and One-Hot encoding. | ✅ Handles raw categories and missing values automatically. |
| **Best For...** | A quick baseline or when interpretability (exactly how much \$1 matters) is key. | Maximizing prediction accuracy and capturing complex user behaviors. |

## 5. Summary Recommendation
The **CatBoost** model is the better choice for deployment because:
1.  It has significantly higher **ROC-AUC (0.89 vs 0.85)**.
2.  It captures more actual churners (**Recall 0.80 vs 0.74**).
3.  It handles the categorical nature of customer data (e.g., Contract types) more naturally.

**Next Steps**:
- Try tuning `learning_rate` and `iterations` further using a Grid Search to squeeze out another 1-2% accuracy.
- Save the CatBoost model using `cat_model.save_model("churn_model.cbm")` for future use.
