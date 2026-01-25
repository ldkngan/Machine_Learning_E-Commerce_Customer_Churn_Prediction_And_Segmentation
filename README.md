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

### Apply base Random Forest model

### Analyse Top 5 Features from Random Forest model

### Insights & Recommendations to Reduce Churn

---

## Churn Prediction Model
### Baseline Model

### Hyperparameter Tuning

### Model Evaluation

---

## Customer Segmentation
### Feature Engineering

### Apply K Means Model

### Model Evaluation

### Conclusion
