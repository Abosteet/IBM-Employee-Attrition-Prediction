# IBM Employee Attrition Prediction 👔

> A machine learning project that predicts employee attrition using IBM HR Analytics data, comparing nine classification approaches including K-Fold cross-validation, ensemble techniques (Voting, Bagging, Stacking), and hyperparameter tuning with Grid Search.

---

## 📌 Overview

Employee attrition is a costly challenge for organizations. This project builds and compares multiple classification models to predict whether an employee will leave the company, using the IBM HR Analytics Employee Attrition dataset.

The project goes beyond basic model training by implementing advanced ensemble techniques including Voting, Bagging, and Stacking classifiers, as well as a data-driven approach to selecting optimal hyperparameters and the best number of K-Fold cross-validation splits.

---

## 🔍 Key Results

| Model | Validation Method | Notes |
|-------|------------------|-------|
| SVC | 6-Fold Cross Validation | Optimal folds selected data-driven |
| KNN | 6-Fold Cross Validation | Best n_neighbors selected via error rate plot |
| Decision Tree | 6-Fold Cross Validation | Best max_depth selected via error rate plot |
| Logistic Regression | 6-Fold Cross Validation | C=1000, max_iter=10000 |
| Random Forest | 6-Fold Cross Validation | Best n_estimators selected via error rate plot |
| Voting Classifier | Train/Test Split (80/20) | Combines SVM, Decision Tree, LR, KNN |
| Bagging Classifier | Train/Test Split (80/20) | Best base estimator and n_estimators selected data-driven |
| Stacking Classifier | Train/Test Split (80/20) | Best final estimator selected data-driven |
| Naive Bayes | Train/Test Split (80/20) | GaussianNB |
| Random Forest + Grid Search | Train/Test Split (80/20) | n_estimators, max_depth, min_samples_split tuned |
| Gradient Boosting | Train/Test Split (80/20) | 1500 estimators, learning rate 0.25 |

---

## ⚙️ Project Pipeline

### 1. Data Cleaning
- Dropped `EmployeeNumber` column
- Verified zero null values and zero duplicate rows
- Dropped columns with a single unique value after outlier handling

### 2. Exploratory Data Analysis
- Univariate analysis with count plots and pie charts for all categorical features
- Categorical features vs Attrition count plots
- Monthly income vs Age, Marital Status, Department, Education, and Job Role scatter plots split by gender
- Distribution histograms for all features
- Correlation heatmap
- Pie charts for low-cardinality features
- Automated EDA report using pandas-profiling

### 3. Data Preprocessing
- Label Encoding for all categorical columns
- IQR-based outlier capping for all non-target columns
- Oversampling minority class (Attrition = Yes) using `resample` to balance the dataset
- Standard Scaling applied before model training

### 4. Optimal Fold Selection
- Tested K-Fold cross-validation from 2 to 10 folds using Random Forest
- Plotted overall accuracy per number of folds to determine the optimal split

### 5. Model Training and Evaluation
- Trained SVC, KNN, Decision Tree, Logistic Regression, and Random Forest with K-Fold cross-validation
- Tuned hyperparameters for KNN, Decision Tree, and Random Forest using error rate plots
- Decision Tree visualization using sklearn tree plotter
- Scatter plot showing Age vs Attrition for Logistic Regression interpretation
- Implemented Voting, Bagging, and Stacking ensemble classifiers with data-driven estimator selection
- Trained Naive Bayes with confusion matrix and classification report
- Trained Random Forest with Grid Search (n_estimators, max_depth, min_samples_split)
- Trained Gradient Boosting with confusion matrix and classification report

---

## 🛠️ Tools and Libraries

| Tool | Usage |
|------|-------|
| **pandas** | Data loading, cleaning, and manipulation |
| **numpy** | Numerical operations |
| **matplotlib / seaborn / plotly** | Data visualization |
| **scikit-learn** | Models, cross-validation, ensemble methods, scaling, and metrics |
| **pandas-profiling** | Automated EDA report generation |

---

## 📂 Dataset

This project uses the **IBM HR Analytics Employee Attrition and Performance** dataset from Kaggle.

🔗 [View Dataset on Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
