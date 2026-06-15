# Predictive Modeling Using Machine Learning: Bank Marketing Response Prediction

## Introduction

Predictive modeling is a fundamental machine learning technique used to forecast future outcomes based on historical data. This project focuses on predicting whether a customer will subscribe to a term deposit offered through a bank marketing campaign.

The primary objective of this project is to build and evaluate supervised machine learning models capable of predicting customer responses using demographic, financial, and campaign-related information.

The project demonstrates key machine learning concepts including:

* Data preprocessing and feature engineering
* Handling missing values
* Encoding categorical variables
* Training supervised learning models
* Evaluating model performance
* Comparing classification algorithms
* Visualizing results using Confusion Matrices and ROC Curves
* Analyzing feature importance

By comparing Decision Tree and Random Forest classifiers, the project aims to identify the most effective model for predicting customer subscription behavior.

---

## Dataset Information

**Source:** UCI Machine Learning Repository

The dataset contains information collected from direct marketing campaigns conducted by a Portuguese banking institution.

### Target Variable

| Value | Description                           |
| ----- | ------------------------------------- |
| Yes   | Customer subscribed to a term deposit |
| No    | Customer did not subscribe            |

### Dataset Overview

| Metric          | Value  |
| --------------- | ------ |
| Rows            | 45,211 |
| Columns         | 17     |
| Target Variable | y      |

### Features Included

| Feature     | Description                        |
| ----------- | ---------------------------------- |
| age         | Customer age                       |
| job         | Job type                           |
| marital     | Marital status                     |
| education   | Education level                    |
| default     | Credit default status              |
| balance     | Average yearly account balance     |
| housing     | Housing loan status                |
| loan        | Personal loan status               |
| contact     | Contact communication type         |
| day_of_week | Last contact day                   |
| month       | Last contact month                 |
| duration    | Last contact duration              |
| campaign    | Number of contacts during campaign |
| pdays       | Days since previous contact        |
| previous    | Number of previous contacts        |
| poutcome    | Outcome of previous campaign       |
| y           | Subscription outcome               |

---

## Tools and Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* UCI ML Repository API (ucimlrepo)

### Machine Learning Algorithms

* Decision Tree Classifier
* Random Forest Classifier

### Development Environment

* Jupyter Notebook

---

## Requirements

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn ucimlrepo
```

---

## Project Structure

```text
Predictive_Modeling_Bank_Marketing_Response_Prediction/
│
├── Predictive_Modeling_Bank_Marketing_Response_Prediction.ipynb
├── README.md
├── images/
│   ├── decision_tree_confusion_matrix.png
│   ├── random_forest_confusion_matrix.png
│   ├── confusion_matrix_comparison.png
│   ├── decision_tree_roc_curve.png
│   ├── random_forest_roc_curve.png
│   ├── roc_curve_comparison.png
└──── feature_importance.png 
```

---

## Data Preprocessing

### Step 1: Loading the Dataset

The dataset was fetched directly from the UCI Machine Learning Repository using:

```python
from ucimlrepo import fetch_ucirepo

bank_marketing = fetch_ucirepo(id=222)
```

### Initial Dataset Dimensions

| Metric  | Value  |
| ------- | ------ |
| Rows    | 45,211 |
| Columns | 17     |

---

### Step 2: Missing Value Analysis

Missing values were identified using:

```python
df.isnull().sum()
```

### Missing Value Summary

| Column    | Missing Values |
| --------- | -------------- |
| job       | 288            |
| education | 1857           |
| contact   | 13020          |
| poutcome  | 36959          |

### Observation

Several categorical features contained missing values, with `poutcome` and `contact` having the highest number of missing entries.

### Insight

Proper handling of missing values is essential to prevent information loss and improve model performance.

---

### Step 3: Target Encoding

The target variable (`y`) was encoded using LabelEncoder.

### Class Distribution

| Class | Count  |
| ----- | ------ |
| No    | 39,922 |
| Yes   | 5,289  |

### Observation

The dataset is imbalanced, with significantly more customers declining the term deposit than subscribing.

### Insight

Class imbalance can affect model performance and should be considered during evaluation.

---

### Step 4: Feature Encoding

Categorical variables were converted into numerical format using one-hot encoding.

### Dataset Dimensions After Encoding

| Metric   | Value  |
| -------- | ------ |
| Rows     | 45,211 |
| Features | 38     |

### Observation

One-hot encoding increased the number of usable features from 17 to 38.

---

### Step 5: Train-Test Split

The dataset was split into training and testing sets using an 80:20 ratio.

### Dataset Split

| Dataset      | Shape       |
| ------------ | ----------- |
| Training Set | (36168, 38) |
| Testing Set  | (9043, 38)  |

Stratified sampling was used to preserve class distribution.

---

## Model Training

### Decision Tree Classifier

A Decision Tree model was trained using the training dataset.

### Random Forest Classifier

A Random Forest model consisting of multiple decision trees was trained and evaluated.

---

## Model Evaluation

### Accuracy Comparison

| Model         | Accuracy |
| ------------- | -------- |
| Decision Tree | 86.66%   |
| Random Forest | 90.46%   |

### Observation

Random Forest achieved the highest prediction accuracy.

### Insight

Combining multiple trees allowed Random Forest to generalize better and reduce overfitting compared to a single Decision Tree.

---

## Classification Report Analysis

### Decision Tree Results

| Metric    | Class 0 | Class 1 |
| --------- | ------- | ------- |
| Precision | 0.93    | 0.43    |
| Recall    | 0.92    | 0.46    |
| F1-Score  | 0.92    | 0.45    |

### Random Forest Results

| Metric    | Class 0 | Class 1 |
| --------- | ------- | ------- |
| Precision | 0.92    | 0.65    |
| Recall    | 0.97    | 0.39    |
| F1-Score  | 0.95    | 0.49    |

### Observation

Random Forest significantly improved precision for predicting customer subscriptions.

### Insight

Although the dataset is imbalanced, Random Forest demonstrated stronger predictive capability for identifying positive responses.

---

## Confusion Matrix Analysis

### Visualizations

* Decision Tree Confusion Matrix
* Random Forest Confusion Matrix
* Side-by-Side Confusion Matrix Comparison

### Observation

The Random Forest model produced fewer overall classification errors compared to the Decision Tree model.
<img width="1025" height="495" alt="confusion_matrix_comparison" src="https://github.com/user-attachments/assets/c431eb30-bd8d-4f7f-98af-df903be27bc3" />


### Insight

Confusion matrices provide a deeper understanding of model performance beyond accuracy alone.

---

## ROC Curve Analysis

### Visualizations

* Decision Tree ROC Curve
* Random Forest ROC Curve
* ROC Curve Comparison

### Observation

The Random Forest ROC curve showed stronger discrimination capability between the two classes.
<img width="691" height="547" alt="roc_curve_comparison" src="https://github.com/user-attachments/assets/08b08c2d-aa46-404d-8610-05e9f9544df4" />


### Insight

ROC analysis confirms that Random Forest is the more effective classifier for this prediction task.

---

## Feature Importance Analysis

The best-performing model (Random Forest) was selected for feature importance analysis.

### Top 10 Important Features

| Feature             | Importance |
| ------------------- | ---------- |
| duration            | 0.278      |
| balance             | 0.106      |
| age                 | 0.100      |
| day_of_week         | 0.091      |
| poutcome_success    | 0.061      |
| pdays               | 0.044      |
| campaign            | 0.040      |
| previous            | 0.024      |
| housing_yes         | 0.020      |
| education_secondary | 0.014      |

### Observation

The duration of customer contact was the most influential factor in determining subscription outcomes.
<img width="972" height="547" alt="feature_importance" src="https://github.com/user-attachments/assets/b06b2ec9-4164-40a9-8090-3874a7f8748f" />


### Insight

Customer engagement during marketing calls appears to play a significant role in successful term deposit subscriptions.

---

## Key Findings

* The dataset contained 45,211 customer records and 17 original features.
* One-hot encoding expanded the feature space to 38 machine-learning-ready features.
* The target variable was highly imbalanced, with substantially more negative responses than positive responses.
* Random Forest achieved the highest accuracy of 90.46%.
* Decision Tree achieved an accuracy of 86.66%.
* Random Forest outperformed Decision Tree across most evaluation metrics.
* Confusion Matrix analysis revealed fewer classification errors for Random Forest.
* ROC Curve analysis confirmed stronger predictive capability for Random Forest.
* Contact duration was identified as the most influential feature affecting customer subscription decisions.
* Customer balance, age, previous campaign outcomes, and campaign-related features also contributed significantly to model predictions.

---

## Conclusion

This project successfully demonstrated the application of supervised machine learning techniques for predictive modeling using real-world banking data.

Two classification algorithms, Decision Tree and Random Forest, were trained and evaluated to predict customer subscription outcomes. The Random Forest model achieved the best overall performance with an accuracy of 90.46%, making it the most effective model for this dataset.

Through preprocessing, feature engineering, model training, evaluation, and feature importance analysis, the project provided practical experience in supervised learning and model assessment. The findings highlight the importance of customer engagement and campaign-related factors in influencing subscription decisions and demonstrate how machine learning can support data-driven decision-making in marketing applications.
