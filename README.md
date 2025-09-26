# Health-Insurance-EDA-Prediction
🧠 Medical Expenses & Insurance Premium Prediction

📌 Project Overview

This project focuses on analyzing and predicting annual medical expenses and insurance premiums based on customers’ demographic and health-related features.

The dataset contains more than 1300 real records, and the aim is to explore how each personal and regional feature contributes to overall insurance costs. These insights help insurance companies optimize pricing models, reduce risks, and better serve customers.

⸻

🎯 Project Goals
 • ✅ Predict medical expenses and premium amounts using machine learning
 • ✅ Identify the most important cost-driving features
 • ✅ Evaluate and compare different regression models
 • ✅ Build an end-to-end pipeline from data exploration to final evaluation
 • ✅ Provide business insight and explainability.

 
 🧾 Dataset Features (Description)

The dataset includes the following features, each of which may play an important role in predicting insurance expenses and premiums:
 • Age:
Represents the age of the individual in years. It’s a key factor, as healthcare costs generally increase with age.
 • Sex:
Indicates the gender of the individual – either male or female. It may affect risk assessment and pricing in insurance models.
 • BMI (Body Mass Index):
A numerical measure of body fat based on height and weight. Higher BMI levels often correlate with increased medical expenses.
 • Children:
The number of dependents (children) covered under the insurance policy. It may affect both expenses and premium calculations.
 • Region:
The geographical region where the individual resides (e.g., southeast, northwest, etc.). Different regions might show different cost patterns due to healthcare access, pricing, and policies.
 • Expenses:
The actual medical costs incurred by the policyholder. This is one of the main target variables to predict.
 • Premium:
The annual insurance premium charged to the customer. This is the second target variable in the prediction task.

🔍 Workflow

1. Exploratory Data Analysis (EDA)
 • Visualize distributions of features
 • Study feature-target correlations
 • Identify outliers and data patterns

2. Data Preprocessing
 • Encode categorical features (sex, region)
 • Scale numeric values (age, bmi, etc.)
 • Handle missing values if any

3. Model Development

Regression models trained and compared:
 • Linear Regression
 • Ridge & Lasso
 • ElasticNet
 • KNN Regressor
 • Decision Tree
 • Random Forest
 • Gradient Boosting
 • XGBoost
 • CatBoost

4. Hyperparameter Tuning

Used GridSearchCV to tune parameters like:
 • alpha, max_depth, n_estimators, learning_rate, etc.

5. Evaluation Metrics

For each model:
 • RMSE (Root Mean Squared Error)
 • MAE (Mean Absolute Error)
 • MSE (Mean Squared Error)
 • R² (R-squared score)
 • Execution Time ⏱️

⸻

📊 Output Insights

After training and evaluation:
 • 🔻 5 worst predictions: where model had the highest error
 • 🔺 5 best predictions: most accurate model predictions
 • 📌 Feature Importance: shows which features most influenced predictions (in tree-based models)

⸻

💡 Business Value

This model helps insurance companies to:
 • Design fair and personalized premiums
 • Identify high-risk profiles (based on age, BMI, etc.)
 • Improve cost forecasting and financial planning
 • Make data-driven decisions.

 💼 Tools & Libraries
 • Python (Jupyter Notebook)
 • Pandas, NumPy, Seaborn, Matplotlib
 • Scikit-learn
 • XGBoost, CatBoost
📬 Contact

Have questions, feedback, or suggestions?
Feel free to open an issue or connect via GitHub.
