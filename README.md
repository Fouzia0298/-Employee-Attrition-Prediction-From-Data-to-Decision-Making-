# Employee Attrition Prediction From Data to Decision Making

This project aims to predict employee attrition using machine learning models and provide actionable insights to improve retention strategies. By leveraging the IBM HR Analytics Dataset, we build classification models such as Random Forest and Logistic Regression, interpret their predictions using SHAP (SHapley Additive exPlanations), and derive key factors influencing employee turnover.

---

## Table of Contents
1. [Objective](#objective)
2. [Dataset Description and Preprocessing Steps](#dataset-description-and-preprocessing-steps)
3. [Models Implemented with Rationale](#models-implemented-with-rationale)
4. [Key Insights and Visualizations](#key-insights-and-visualizations)
5. [Actionable Recommendations](#actionable-recommendations)
6. [Challenges Faced and Solutions](#challenges-faced-and-solutions)
7. [Tools and Libraries](#tools-and-libraries)
8. [How to Run the Code](#how-to-run-the-code)

---

## Objective
The primary goal of this project is to:
- Build a classification model to predict whether an employee will leave the company.
- Use explainable AI techniques (e.g., SHAP) to interpret model predictions.
- Provide actionable insights to HR teams for reducing employee attrition.

---

## Dataset Description and Preprocessing Steps

### Dataset Description
- **Dataset Name:** IBM HR Analytics Employee Attrition & Performance Dataset
- **Source:** [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Features:** The dataset contains various employee attributes such as:
  - **Categorical Features:** Department, Job Role, Marital Status, Gender, Overtime.
  - **Numerical Features:** Age, Monthly Income, Total Working Years, Years at Company, Distance from Home.
- **Target Variable:** `Attrition` (Yes/No).

### Preprocessing Steps
1. **Handling Missing Values:**
   - Checked for missing values and ensured the dataset was complete before analysis.
2. **Encoding Categorical Variables:**
   - Converted categorical features into numerical values using `LabelEncoder`.
3. **Feature Scaling:**
   - Normalized numerical features using `StandardScaler` to ensure all features were on the same scale.
4. **Train-Test Split:**
   - Split the dataset into training (80%) and testing (20%) sets using `train_test_split`.
5. **Balancing the Dataset:**
   - Addressed class imbalance in the target variable (`Attrition`) by considering techniques like SMOTE (if necessary).

---

## Models Implemented with Rationale

### Models Implemented
1. **Random Forest Classifier:**
   - A tree-based ensemble model capable of capturing non-linear relationships and interactions between features.
   - Selected for its robustness and ability to handle both categorical and numerical data effectively.
2. **Logistic Regression:**
   - A simple and interpretable linear model used as a baseline for comparison.
   - Selected to evaluate the performance of a basic model against more complex algorithms.

### Rationale for Selection
- **Random Forest:** Chosen for its high accuracy and ability to rank feature importance, making it ideal for interpretability.
- **Logistic Regression:** Used as a baseline to benchmark performance and ensure simpler models are not overlooked.

---

## Key Insights and Visualizations

### Key Insights
- **High-Impact Features:** Monthly income, overtime, and years at the company are the most significant predictors of attrition.
- **Trends Observed:**
  - Employees with lower salaries are more likely to leave.
  - Frequent overtime correlates strongly with higher attrition risk.
  - Employees in certain departments (e.g., Sales, Research & Development) exhibit higher turnover rates.

### Visualizations
1. **Attrition Distribution:**
   - A bar plot showing the distribution of employees who left (`Yes`) versus those who stayed (`No`).
2. **Feature Importance:**
   - A SHAP summary plot highlighting the top features influencing attrition.
3. **Dependence Plot:**
   - A scatter plot showing the relationship between `MonthlyIncome` and SHAP values, indicating how income impacts attrition risk.
4. **Categorical Feature Analysis:**
   - Bar plots comparing attrition rates across departments, job roles, and marital statuses.

---

## Actionable Recommendations
1. **Salary Adjustments:** Revise compensation packages to retain employees with lower incomes.
2. **Work-Life Balance:** Promote flexible work arrangements to reduce overtime-related stress.
3. **Career Growth Opportunities:** Provide mentorship programs and clear career progression paths.
4. **Department-Specific Strategies:** Investigate and address issues specific to high-attrition departments.
5. **Retention Programs:** Focus on improving onboarding processes and engagement activities for new hires.

---

## Challenges Faced and Solutions

### Challenges
1. **Class Imbalance:**
   - The dataset had an imbalance between employees who left (`Yes`) and those who stayed (`No`), affecting model performance.
   - **Solution:** Considered techniques like SMOTE to balance the dataset and improve minority class prediction.

2. **Feature Interpretability:**
   - Complex models like Random Forest can be difficult to interpret without tools like SHAP or LIME.
   - **Solution:** Used SHAP to explain model predictions and identify important features.

3. **Preprocessing Mismatches:**
   - Ensuring alignment between feature names and transformed data (e.g., after encoding or scaling) was challenging.
   - **Solution:** Verified feature names and shapes at each step to avoid mismatches.

4. **Computational Complexity:**
   - Training tree-based models on large datasets required significant computational resources.
   - **Solution:** Reduced dataset size during experimentation and optimized hyperparameters.

---

## Tools and Libraries
- **Programming Language:** Python
- **Libraries Used:**
  - `pandas`, `numpy`: Data manipulation and analysis.
  - `matplotlib`, `seaborn`: Data visualization.
  - `scikit-learn`: Model training and evaluation.
  - `shap`: Model interpretation and explainability.
- **Environment:** Jupyter Notebook or Python IDE.

---

## How to Run the Code
1. Clone the repository or download the notebook.
2. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn shap

3. Load the IBM HR Analytics Dataset (IBM_HR_Analytics.csv) into the project directory.
4. Run the notebook cells sequentially to reproduce the results.

## 🤝 Acknowledgment
This project is part of my internship tasks.
Big thanks to Developers Hub for their guidance and support!

## 📜 License
This project is open-source and available under the MIT License .

## 🔥 Let's fight fraud with AI! 🚀

## #MachineLearning #AI #FraudDetection #DataScience #Python #Internship


