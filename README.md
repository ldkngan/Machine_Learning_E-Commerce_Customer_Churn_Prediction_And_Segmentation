# Machine Learning | E-Commerce Customer Churn Prediction & Segmentation
![churn prediction](https://github.com/user-attachments/assets/747e24ad-f63e-4b30-9639-efef4e6d2f5a)
This project addresses a real-world e-commerce problem: predicting customer churn and designing targeted promotional strategies to retain users.
- **Author:** Le Dang Kim Ngan
- **Tool Used:** Python - Machine Learning

---

## Table of Contents
1. Overview
2. Exploratory Data Analysis (EDA)
3. Churn Prediction Model
4. Customer Segmentation

---

## Overview
### Objective
- This project analyzes customer churn behavior for an e-commerce company and translates data into **clear business insights** that support customer retention and targeted marketing strategies.
- Instead of focusing only on model performance, the project emphasizes **understanding customer behavior**, **identifying churn drivers**, and **turning analysis results into actionable recommendations** for the business.

### This project aims to
- Analyze behavioral differences between churned and active customers using EDA
- Identify key churn drivers that impact customer retention
- Predict churn risk to help the business prioritize high-risk customers
- Segment churned customers into distinct groups based on behavior patterns
- Support data-driven decisions for personalized promotions and retention campaigns

### Main business questions addressed
- What behaviors are most commonly associated with customer churn?
- Which customer characteristics indicate a high risk of churn?
- How can churned customers be grouped to design more effective re-engagement strategies?

### This project is designed for
- Data Analysts & Business Analysts – To understand churn behavior, identify key drivers, and translate customer data into actionable retention insights.
- Marketing & CRM Teams – To design targeted promotions and re-engagement campaigns based on customer segments and churn risk.
- E-commerce Stakeholders – To support data-driven decisions that improve customer retention and lifetime value.

---

## Exploratory Data Analysis
### Data Preprocessing
This step ensures the dataset is clean, consistent, and suitable for modeling. The following preprocessing actions were performed:

**Handling Missing Information**
- Several customer behavior fields contained missing values, such as tenure, order frequency, app usage time, and recency of purchases.
- These missing values were filled using typical (median) values to maintain realistic customer patterns and avoid distorting the analysis.

**Duplicate Data Check**
- The dataset was reviewed for duplicated customer records.
- No duplicates were found, so the data did not require further correction at this step.

**Standardizing Customer Preferences**
- Some customer preference fields contained different labels with the same meaning (e.g., device type, payment method, order category).
- These values were unified into consistent categories to ensure accurate customer segmentation and modeling.

After completing the preprocessing steps, the final dataset contains **5,630 rows** and **20 features**.
The dataset is fully prepared for modeling, with no remaining missing values, duplicated records, or same-meaning categorical values, ensuring consistency and reliability for downstream analysis, churn prediction and churn segmentation.

### Feature Engineering
In this step, raw variables were transformed into a machine-learning-ready format to improve model compatibility and performance.

```python
# Encoding
df.drop(['CustomerID'], axis=1, inplace=True)
list_encode_columns = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus']
df_encoded = pd.get_dummies(df, columns=list_encode_columns, drop_first=True )
```

- **Removing Non-Informative Identifiers**: The CustomerID column was removed, as it is a unique identifier and does not provide predictive value for churn behavior.
- **Categorical Variable Encoding**: The following categorical features were converted into numerical format using one-hot encoding: PreferredLoginDevice, PreferredPaymentMode, Gender, PreferedOrderCat, MaritalStatus.
- To avoid multicollinearity and reduce redundancy, the first category of each feature was dropped (drop_first=True).
- **Resulting Dataset Structure**: After encoding, the dataset expanded from 20 to 27 features, while preserving the original 5,630 rows.

This feature engineering step enables machine learning models to interpret categorical customer attributes correctly and prepares the dataset for scaling and model training.

### Apply base Random Forest model
This section applies a baseline Random Forest model to understand key drivers of customer churn and extract actionable insights from feature importance.

**Split train/test set**
```python
from sklearn.model_selection import train_test_split

x = df_encoded.drop('Churn', axis = 1)
y = df_encoded[['Churn']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print(f"Number data of train set: {len(x_train)}")
print(f"Number data of test set: {len(x_test)}")
```

**Normalization**
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
```

**Apply Random Forest Model**
```python
from sklearn.ensemble import RandomForestClassifier

clf_rand = RandomForestClassifier(max_depth=15, random_state=0, n_estimators=100)

clf_rand.fit(x_train_scaled, y_train)
y_ranf_pre_train = clf_rand.predict(x_train_scaled)
y_ranf_pre_test = clf_rand.predict(x_test_scaled)
```

**Model Evaluation**
```python
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report, recall_score

print(f'Training Balanced Accuracy: {balanced_accuracy_score(y_train, y_ranf_pre_train)}')
print(f'Test Balanced Accuracy: {balanced_accuracy_score(y_test, y_ranf_pre_test)}')
print(f'\nTraining Recall: {recall_score(y_train, y_ranf_pre_train)}')
print(f'Test Recall: {recall_score(y_train, y_ranf_pre_train)}')
```
**Result**
```
Training Balanced Accuracy: 0.9984520123839009
Test Balanced Accuracy: 0.8590447246666062

Training Recall: 0.9969040247678018
Test Recall: 0.9969040247678018
```

Based on the strong recall performance and acceptable generalization on the test set, the Random Forest model was used to extract **feature importance** in order to identify the key factors driving customer churn.

### Analyse Top 5 Features from Random Forest model
**Show Feature Importance**
```python
feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(x_test.columns, clf_rand.feature_importances_):
    feats[feature] = importance # add the name/value pair

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances = importances.sort_values(by='Gini-importance', ascending=True)

importances = importances.reset_index()

# Create bar chart
plt.figure(figsize=(10, 10))
plt.barh(importances.tail(20)['index'][:20], importances.tail(20)['Gini-importance'])

plt.title('Feature Importance')

# Show plot
plt.show()
```
**Result**

<img width="959" height="749" alt="image" src="https://github.com/user-attachments/assets/80958448-269a-4af9-bb4b-118d3dc44213" />

- The feature importance results indicate that customer churn is primarily driven by engagement duration (Tenure), promotional sensitivity (CashbackAmount), service experience (Complain), logistical convenience (WarehoueToHome) and recent purchase behavior (DaySinceLastOrder).
- Based on these findings, the next step focuses on a deeper analysis of the **top 5 features** to understand how each factor influences churn behavior and to derive actionable recommendations for reducing customer attrition.

**Box Plot - Analyse Top Features**
<img width="1824" height="817" alt="box plot" src="https://github.com/user-attachments/assets/d2f8ff15-2065-4da5-a5c6-c2825c1f174d" />

### Key Insights & Recommendations to Reduce Churn
| Feature | Box Plot Observation | Key Insight | Recommendation |
|--------|----------------------|-------------|----------------|
| **Tenure (Engagement Duration)** | Non-churned users show significantly longer tenure with a wide distribution, while churned users are heavily concentrated at very low tenure (median ~0–2). | Churned users are predominantly **new users** who leave before forming usage habits or realizing the platform’s value. | **Strengthen early-stage user engagement (first 1–3 months):** Improve onboarding, provide clear usage guidance, and offer incentives for first and second purchases to help users build habits early. |
| **CashbackAmount (Promotional Sensitivity)** | Non-churned users have a higher median cashback and a wider upper range, while churned users show lower median cashback with a compressed distribution. | Higher cashback exposure is associated with better retention. Insufficient or poorly targeted incentives increase churn risk. | **Use targeted cashback as a retention lever:** Increase or personalize cashback for users with declining activity or low cumulative cashback to reinforce purchasing motivation. |
| **Complain (Service Experience)** | Non-churned users rarely submit complaints, whereas churned users almost always have complaint records. | A complaint is a **strong churn signal**. Unresolved service issues significantly increase the likelihood of churn. | **Treat complaints as high-priority churn risks:** Proactively flag users who submit complaints, respond quickly, and offer personalized compensation or recovery actions. |
| **WarehouseToHome (Logistical Convenience)** | Churned users have a higher median warehouse-to-home distance and a wider spread, while non-churned users are generally closer to warehouses. | Longer delivery distances introduce logistics friction (delivery delays, higher costs), increasing churn risk. | **Mitigate logistics-related churn risks:** Offer delivery fee subsidies, faster shipping options, or clearer delivery-time expectations for users located far from warehouses. |
| **DaySinceLastOrder (Recent Purchase Behavior)** | Churned users show long gaps or complete inactivity with extreme outliers, while non-churned users have shorter and more stable ordering intervals. | A **break in purchasing behavior** is a clear early warning sign of churn. | **Proactively re-engage inactive users:** Monitor days since last order and trigger reminders, personalized offers, or product recommendations before users disengage completely. |

---

## Churn Prediction Model
### Baseline Model
In this phase, a baseline machine learning model is built to establish an initial performance benchmark for churn prediction. A baseline model is trained using default or minimally adjusted parameters on the preprocessed and engineered dataset. Model performance is evaluated on a hold-out test set using appropriate metrics for imbalanced data, such as recall and balanced accuracy.
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score

models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "DecisionTree": DecisionTreeClassifier(max_depth=5),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

results = []

for name, model in models.items():
    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    results.append({
        "model": name,
        "precision": precision,
        "recall": recall,
        "balanced_accuracy": bal_acc
    })

results_df = pd.DataFrame(results).sort_values(
    "balanced_accuracy", ascending=False
)

print(results_df)
```
**Result**
```
                model  precision    recall  balanced_accuracy
2        RandomForest   0.923077  0.754967           0.870634
3    GradientBoosting   0.803653  0.582781           0.775890
0  LogisticRegression   0.748792  0.513245           0.737877
1        DecisionTree   0.801242  0.427152           0.702040
```
**Conclusion**

Among the baseline models, **Random Forest** clearly outperforms the others across all key evaluation metrics.
- It achieves the **highest balanced accuracy (0.87)**, indicating better overall performance on the imbalanced churn dataset.
- It also delivers the **highest recall (0.75)**, meaning it identifies a larger proportion of churned customers compared to other models.
- Precision remains strong (0.92), showing a reasonable trade-off between capturing churned users and limiting false positives.
- In contrast, Logistic Regression and Decision Tree show weaker recall, while Gradient Boosting underperforms Random Forest in both recall and balanced accuracy at the baseline stage.

Based on these results, **Random Forest** is selected as the **baseline model** for further improvement. The next step focuses on **hyperparameter tuning** to enhance recall and balanced accuracy while controlling overfitting.

### Hyperparameter Tuning
```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Hyperparameter Tuning
rf = RandomForestClassifier(random_state=42)

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Use GridSearchCV to find the best parameters
rf_grid = GridSearchCV(
    rf,
    param_grid,
    scoring='balanced_accuracy',
    cv=cv
)

# Fit the model
start_time = time.time()
rf_grid.fit(x_train_scaled, y_train)
rf_time = time.time() - start_time

# Print the best parameters
print("Best Parameters: ", rf_grid.best_params_)
print("Balanced Accuracy:", rf_grid.best_score_)
```

**Result**
```
Best Parameters:  {'bootstrap': False, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
Balanced Accuracy: 0.8980785925441322
```
### Model Evaluation
```python
best_model = rf_grid.best_estimator_

y_test_pred = best_model.predict(x_test_scaled)

print("Test Balanced Accuracy:", balanced_accuracy_score(y_test, y_test_pred))
print("Test Recall:", recall_score(y_test, y_test_pred))
```
**Result**
```
Test Balanced Accuracy: 0.8978451753988073
Test Recall: 0.8079470198675497
```
### Conclusion
After hyperparameter tuning, the churn prediction model shows a clear and meaningful performance improvement compared to the baseline.
- **Test Balanced Accuracy increased to 0.90**, indicating better overall discrimination between churned and non-churned users on an imbalanced dataset.
- **Test Recall improved to 0.81**, meaning the model successfully identifies more than 80% of churned customers, significantly reducing false negatives.

These results demonstrate that the tuned model generalizes well to unseen data while maintaining strong sensitivity to churned users. From a business perspective, this level of recall is particularly valuable, as it allows the company to **proactively target at-risk customers** before they leave. Overall, the final model achieves a **balanced trade-off between performance and practicality**, making it suitable for deployment in churn prediction and retention strategy design.

---

## Customer Segmentation
- In this phase, **unsupervised learning** is applied to gain deeper insights into **churned users only**, with the objective of identifying distinct behavioral segments within this high-risk group.
- Instead of predicting churn, this step focuses on **understanding *how* and *why* different groups of users leave**, enabling more targeted and effective retention strategies.
### Feature Engineering

### Apply K Means Model

### Model Evaluation

### Conclusion
