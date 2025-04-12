# 🏥 Insurance Premium Predictive Model

### Developed a machine learning solution to optimize premium pricing by forecasting health insurance costs based on customer health profile.

## 🔗 Project Demonstration
### Demo GIF

![Premium](https://github.com/user-attachments/assets/71366618-a702-4c6c-a131-a8ecdee3b521)

## 🪪 Credits

This capstone project is a part of the “_Master Machine Learning for Data Science & AI: Beginner to Advanced_” course offered by **Codebasics** - All rights reserved.

- **Course Instructor**: Mr. Dhaval Patel
- **Platform**: codebasics.io – All rights reserved.

All education content and dataset used as learning resources belong to Codebasics and are protected under their respective rights and terms of use.

## 📌 Table of Contents
- [Description](#description)
- [Tech Stack](#tech-stack)
- [Data Understanding](#data-understanding)
- [Data Preparation](#data-preparation)
- [Feature Engineering](#feature-engineering)
- [Model Development](#model-development)
- [Error Analysis](#error-analysis)
- [Data Segmentation](#data-segmentation)
- [Model Retraining](#model-retraining)
- [Streamlit Application](#streamlit-application)
- [Reflections](#reflections)
- [Conclusion](#conclusion)

## 📝 Description

**AtliQ AI** will develop a predictive model for **Shield Insurance** to estimate health insurance premiums based on key factors like age, smoking habits, BMI, and medical history. 

The project will be executed in two phases:
- **Phase 1 (MVP)**: Build and deploy a predictive model integrated into a Streamlit application. 
- **Phase 2**: Develop infrastructure to enable straight-through processing (STP) of insurance quotes.

_Note: This project scope is limited to Phase 1. A detailed Phase 1 walkthrough is provided in scope-of-work document._

## 🛠️ Tech Stack  
| Task                 | Libraries Used                      |
|----------------------|-------------------------------------|
| Data Preprocessing   | Pandas                              |
| Data Visualization   | Matplotlib, Seaborn                 |
| Feature Engineering  | Pandas, Statsmodels, Scikit-learn   |
| Model Training       | Scikit-learn, XGBoost               |
| Model Fine Tuning    | Scikit-learn                        |
| UI Frontend          | Streamlit                           |

## 📊 Data Understanding
### Raw Data Overview:
| Features               | Description                         | Data Type |
|------------------------|-------------------------------------|-----------|
| Age                    | Age of the individual               | Integer   |
| Gender                 | Gender of the individual            | String    |
| Region                 | Geographic region of residence      | String    |
| Marital_status         | Marital status of the individual    | String    |
| Number Of Dependants   | Number of dependents                | Integer   |
| BMI_Category           | Body Mass Index classification      | String    |
| Smoking_Status         | Smoking habits (No Smoking, Regular) | String    |
| Employment_Status      | Employment type (Salaried/Self-Employed) | String    |
| Income_Level           | Broad income range category         | String    |
| Income_Lakhs           | Annual income in lakhs (Indian Rs.)  | Float     |
| Medical History        | Pre-existing medical conditions     | String    |
| Insurance_Plan         | Type of insurance plan (Bronze/Silver/Gold) | String    |
| Annual_Premium_Amount  | Yearly insurance premium amount     | Integer   |

### Data Quality Assessment:
```
import pandas as pd

df = pd.read_excel("customer_profile_with_premium.xlsx")

print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}",end="\n\n")

print(f"Duplicate rows: {df.duplicated().sum()}",end="\n\n")

print("Missing Values:")
print(df.isna().sum())

print("Check for outliers:")

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
numeric_columns

for col in numeric_columns:
    sns.boxplot(x=df[col])
    plt.show()
```

## 🧼 Data Preparation
### Data Cleaning:
```
# Column header consistency
df.columns = df.columns.str.replace(" ","_").str.lower()

# Handle Duplicate Rows
df.drop_duplicates(inplace=True)

# Handle Missing Values
df.dropna(inplace=True)

# Outlier Treatment
# Age column
df1 = df[df.age<=100]

# Income column
quantile_thresold = df1.income_lakhs.quantile(0.999)
quantile_thresold

df2 = df1[df1.income_lakhs<=quantile_thresold].copy()

# Fixing Inconsistency
df2['smoking_status'].replace({
    'Not Smoking': 'No Smoking',
    'Does Not Smoke': 'No Smoking',
    'Smoking=0': 'No Smoking'
}, inplace=True)

df2['smoking_status'].unique()

# Column Split
df2[['disease1', 'disease2']] = df2['medical_history'].str.split(" & ", expand=True).apply(lambda x: x.str.lower())
df2['disease1'].fillna('none', inplace=True)
df2['disease2'].fillna('none', inplace=True)
```

## 🔧 Feature Engineering
### Feature Encoding
```
df2['insurance_plan'] = df2['insurance_plan'].map({'Bronze': 1, 'Silver': 2, 'Gold': 3})
df2['income_level'] = df2['income_level'].map({'<10L':1, '10L - 25L': 2, '25L - 40L':3, '> 40L':4})

nominal_cols = ['gender', 'region', 'marital_status', 'bmi_category', 'smoking_status', 'employment_status']
df3 = pd.get_dummies(df2, columns=nominal_cols, drop_first=True, dtype=int)
```

### Calculate Risk Score
```
# Define the risk scores for each condition
risk_scores = {
    "diabetes": 6,
    "heart disease": 8,
    "high blood pressure":6,
    "thyroid": 5,
    "no disease": 0,
    "none":0
}

# Map risk score
for disease in ['disease1', 'disease2']:
    df2['total_risk_score'] += df2[disease].map(risk_scores)

# Normalize the risk score to a range of 0 to 1
max_score = df2['total_risk_score'].max()
min_score = df2['total_risk_score'].min()
df2['normalized_risk_score'] = (df2['total_risk_score'] - min_score) / (max_score - min_score)
df2.head(2)

# Drop additional columns
df4 = df3.drop(['medical_history','disease1', 'disease2', 'total_risk_score'], axis=1)
```
### Feature Scaling
```
from sklearn.preprocessing import MinMaxScaler

cols_to_scale = ['age','number_of_dependants', 'income_level',  'income_lakhs', 'insurance_plan']
scaler = MinMaxScaler()
```

### Calculate VIF for Multicolinearity
```
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(data):
    vif_df = pd.DataFrame()
    vif_df['Column'] = data.columns
    vif_df['VIF'] = [variance_inflation_factor(data.values,i) for i in range(data.shape[1])]
    return vif_df

X = df4.drop('annual_premium_amount', axis='columns')
y = df4['annual_premium_amount']

calculate_vif(X)

# we will drop income_lakhs due to high VIF value
X_reduced = X.drop('income_level', axis="columns")
```

## 🤖 Model Development
### Train Test Split
```
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.30, random_state=10)

# shape of the X_train, X_test, y_train, y_test features
print("x train: ",X_train.shape)
print("x test: ",X_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)
```

### Linear Regression Model
```
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
test_score = model_lr.score(X_test, y_test)
train_score = model_lr.score(X_train, y_train)
train_score, test_score

y_pred = model_lr.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred)
rmse_lr = np.sqrt(mse_lr)
print("Linear Regression ==> MSE: ", mse_lr, "RMSE: ", rmse_lr)

feature_importance = model_lr.coef_

# Create a DataFrame for easier handling
coef_df = pd.DataFrame(feature_importance, index=X_train.columns, columns=['Coefficients'])

# Sort the coefficients for better visualization
coef_df = coef_df.sort_values(by='Coefficients', ascending=True)

# Plotting
plt.figure(figsize=(8, 4))
plt.barh(coef_df.index, coef_df['Coefficients'], color='steelblue')
plt.xlabel('Coefficient Value')
plt.title('Feature Importance in Linear Regression')
plt.show()
```

### Ridge Regression Model
```
model_rg = Ridge(alpha=1)
model_rg.fit(X_train, y_train)
test_score = model_rg.score(X_test, y_test)
train_score = model_rg.score(X_train, y_train)
print(f"Ridge Model Train Score: {train_score}")
print(f"Ridge Model Test Score: {test_score}")

y_pred = model_rg.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred)
rmse_lr = np.sqrt(mse_lr)
print("Ridge Regression ==> MSE: ", mse_lr, "RMSE: ", rmse_lr)
```

### XGBoost 
```
from xgboost import XGBRegressor

model_xgb = XGBRegressor(n_estimators=20, max_depth=3)
model_xgb.fit(X_train, y_train)
print("XGBoost Model Test Score: {model_xgb.score(X_test, y_test)}")

y_pred = model_xgb.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred)
rmse_lr = np.sqrt(mse_lr)
print(f"XGBoost Regression ==> MSE: ", mse_lr, "RMSE: ", rmse_lr)

# Using RandomizedSearchCV

model_xgb = XGBRegressor()
param_grid = {
    'n_estimators': [20, 40, 50],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
}
random_search = RandomizedSearchCV(model_xgb, param_grid, n_iter=10, cv=3, scoring='r2', random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)x
print("XGBoost Model Best Score: {random_search.best_score_}")
print("XGBoost Best Model Parameters: {random_search.best_params_}")
```
####  Feature Importance in XGBoost 
```
feature_importance = best_model.feature_importances_

# Create a DataFrame for easier handling
coef_df = pd.DataFrame(feature_importance, index=X_train.columns, columns=['Coefficients'])

# Sort the coefficients for better visualization
coef_df = coef_df.sort_values(by='Coefficients', ascending=True)

# Plotting
plt.figure(figsize=(8, 4))
plt.barh(coef_df.index, coef_df['Coefficients'], color='steelblue')
plt.xlabel('Coefficient Value')
plt.title('Feature Importance in XGBoost')
plt.show()
```

## Error Analysis
Our XGBoost Regressor model's best score was 0.9809474553641963. But need to find margin of error as well. 
```
y_pred = best_model.predict(X_test)

residuals = y_pred - y_test
residuals_pct = (residuals / y_test) * 100

results_df = pd.DataFrame({
    'actual': y_test, 
    'predicted': y_pred, 
    'diff': residuals, 
    'diff_pct': residuals_pct
})
print(results_df.head())

# Visualize the results
sns.histplot(results_df['diff_pct'], kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Diff PCT')
plt.ylabel('Frequency')
plt.show()

# Check for `diff_pct` more than 10%
extreme_error_threshold = 10  # You can adjust this threshold based on your domain knowledge or requirements
extreme_results_df = results_df[np.abs(results_df['diff_pct']) > extreme_error_threshold]
print(extreme_results_df.head())

extreme_errors_pct = extreme_results_df.shape[0]*100/X_test.shape[0]
print(extreme_errors_pct)
```
We have 30% extreme errors which means for 30% customers we will either overcharge or undercharge by 10% or more.

### Distirbution of Extreme errors vs Overall
```
extreme_errors_df = X_test.loc[extreme_results_df.index]

for feature in X_test.columns:
    plt.figure(figsize=(10, 4))
    sns.histplot(extreme_errors_df[feature], color='red', label='Extreme Errors', kde=True)
    sns.histplot(X_test[feature], color='blue', label='Overall', alpha=0.5, kde=True)
    plt.legend()
    plt.title(f'Distribution of {feature} for Extreme Errors vs Overall')
    plt.show()
```
`age` column was identified to have shown signs of extreme error. Now, we need to reverse feature scaling to find out the exact age group.

### Reverse Scaling
```
extreme_errors_df['income_level']=-1 # we have to add a dummy income column as we have dropped it earlier.

df_reversed = pd.DataFrame()
df_reversed[cols_to_scale] = scaler.inverse_transform(extreme_errors_df[cols_to_scale])
df_reversed.head()

sns.histplot(df_reversed.age) 
```
Histogram shows that majority of the extreme errors are coming from young age group (i.e. <25 years of age). We need to may be build a separate model for this segment.

## Data Segmentation
To tackle this error, we have to split the data into two segments and create two different models
1. Young customers (<25 years of age)
2. Rest (>25 years of age)
```
import pandas as pd
df = pd.read_excel("customer_profile_with_premium.xlsx")

df_young = df[df.Age<=25]
df_rest = df[df.Age>25]

df_young.to_excel("customer_profile_with_premium_young.xlsx", index=False)
df_rest.to_excel("customer_profile_with_premium_rest.xlsx", index=False)
```
Results from this approach:

| Particular        | Overall | Rest  | Young |
|-------------------|---------|-------|-------|
| Linear Regression | 0.928   | 0.953 | 0.602 |
| Ridge Regression  | 0.928   | 0.953 | 0.602 |
| XGBoost           | 0.98    | 0.99  | 0.599 |
| Error             | 0.299   | 0.003 | 0.73  |

Since, `Linear Regression` has maginally higher score than other two models for `Young customers` we'll use it as primary model for Younger customers to check distirbution of Extreme errors.

### Distirbution of Extreme errors vs Overall of Younger customers
```
for feature in X_test.columns:
    plt.figure(figsize=(10, 4))
    sns.histplot(extreme_errors_df[feature], color='red', label='Extreme Errors', kde=True)
    sns.histplot(X_test[feature], color='blue', label='Overall', alpha=0.5, kde=True)
    plt.legend()
    plt.title(f'Distribution of {feature} for Extreme Errors vs Overall')
    plt.show()
```
By comparing distributions of results with extreme errors dataframe for Younger customers we don't get much insights. We may need more features
in order to improve the performance. We will ask business to collect more features for our dataset to move ahead.
...
Post request made in weekly project status meeting, we received a new copy of `customer_profile_with_premium` dataset but this time with `genetic factor` only for Younger customers. Therefore, we have to move forward with it.

## Model Retraining
We simply follow the same procedure as we did before but since we don't have `genetic factor` for demograhy age > 25, we have to add dummy genetical_risk column to be consistent with young model
```
df = pd.read_excel("premiums_rest.xlsx")

df['Genetical_Risk'] = 0
```
Results after getting more data:

| Particular        | Overall | Rest  | Young |
|-------------------|---------|-------|-------|
| Linear Regression | 0.928   | 0.953 | 0.953 |
| Ridge Regression  | 0.928   | 0.953 | 0.953 |
| XGBoost           | 0.98    | 0.99  | 0.98  |
| Error             | 0.299   | 0.003 | 0.003 |

### Export the Model
```
from joblib import dump

dump(best_model, "artifacts/model_rest.joblib")
scaler_with_cols = {
    'scaler': scaler,
    'cols_to_scale': cols_to_scale
}
dump(scaler_with_cols, "artifacts/scaler_rest.joblib")
```

## Streamlit Application
### Prediction Helper:
```
import pandas as pd
import joblib

model_young = joblib.load("artifacts\model_young.joblib")
model_rest = joblib.load("artifacts\model_rest.joblib")
scaler_young = joblib.load("artifacts\scaler_young.joblib")
scaler_rest = joblib.load("artifacts\scaler_rest.joblib")

def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    # Split the medical history into potential two parts and convert to lowercase
    diseases = medical_history.lower().split(" & ")

    # Calculate the total risk score by summing the risk scores for each part
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)  # Default to 0 if disease not found

    max_score = 14 # risk score for heart disease (8) + second max risk score (6) for diabetes or high blood pressure
    min_score = 0  # Since the minimum score is always 0

    # Normalize the total risk score
    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)

    return normalized_risk_score

def preprocess_input(input_dict):
    # Define the expected columns and initialize the DataFrame with zeros
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

    df = pd.DataFrame(0, columns=expected_columns, index=[0])
    # df.fillna(0, inplace=True)

    # Manually assign values for each categorical input based on input_dict
    for key, value in input_dict.items():
        if key == 'Gender' and value == 'Male':
            df['gender_Male'] = 1
        elif key == 'Region':
            if value == 'Northwest':
                df['region_Northwest'] = 1
            elif value == 'Southeast':
                df['region_Southeast'] = 1
            elif value == 'Southwest':
                df['region_Southwest'] = 1
        elif key == 'Marital Status' and value == 'Unmarried':
            df['marital_status_Unmarried'] = 1
        elif key == 'BMI Category':
            if value == 'Obesity':
                df['bmi_category_Obesity'] = 1
            elif value == 'Overweight':
                df['bmi_category_Overweight'] = 1
            elif value == 'Underweight':
                df['bmi_category_Underweight'] = 1
        elif key == 'Smoking Status':
            if value == 'Occasional':
                df['smoking_status_Occasional'] = 1
            elif value == 'Regular':
                df['smoking_status_Regular'] = 1
        elif key == 'Employment Status':
            if value == 'Salaried':
                df['employment_status_Salaried'] = 1
            elif value == 'Self-Employed':
                df['employment_status_Self-Employed'] = 1
        elif key == 'Insurance Plan':  # Correct key usage with case sensitivity
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif key == 'Age':  # Correct key usage with case sensitivity
            df['age'] = value
        elif key == 'Number of Dependants':  # Correct key usage with case sensitivity
            df['number_of_dependants'] = value
        elif key == 'Income in Lakhs':  # Correct key usage with case sensitivity
            df['income_lakhs'] = value
        elif key == "Genetical Risk":
            df['genetical_risk'] = value

    # Assuming the 'normalized_risk_score' needs to be calculated based on the 'age'
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])
    df = handle_scaling(input_dict['Age'], df)

    return df

def handle_scaling(age, df):
    # scale age and income_lakhs column
    if age <= 25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    df['income_level'] = None # since scaler object expects income_level supply it. This will have no impact on anything
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    df.drop('income_level', axis='columns', inplace=True)

    return df

def predict(input_dict):
    input_df = preprocess_input(input_dict)

    if input_dict['Age'] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)

    return int(prediction[0])
```
### Main File
```
import streamlit as st
from prediction_helper import predict

# Define the page layout
st.title('Health Insurance Cost Predictor')

categorical_options = {
    'Gender': ['Male', 'Female'],
    'Marital Status': ['Unmarried', 'Married'],
    'BMI Category': ['Normal', 'Obesity', 'Overweight', 'Underweight'],
    'Smoking Status': ['No Smoking', 'Regular', 'Occasional'],
    'Employment Status': ['Salaried', 'Self-Employed', 'Freelancer', ''],
    'Region': ['Northwest', 'Southeast', 'Northeast', 'Southwest'],
    'Medical History': [
        'No Disease', 'Diabetes', 'High blood pressure', 'Diabetes & High blood pressure',
        'Thyroid', 'Heart disease', 'High blood pressure & Heart disease', 'Diabetes & Thyroid',
        'Diabetes & Heart disease'
    ],
    'Insurance Plan': ['Bronze', 'Silver', 'Gold']
}

# Create four rows of three columns each
row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)

# Assign inputs to the grid
with row1[0]:
    age = st.number_input('Age', min_value=18, step=1, max_value=100)
with row1[1]:
    number_of_dependants = st.number_input('Number of Dependants', min_value=0, step=1, max_value=20)
with row1[2]:
    income_lakhs = st.number_input('Income in Lakhs', step=1, min_value=0, max_value=200)

with row2[0]:
    genetical_risk = st.number_input('Genetical Risk', step=1, min_value=0, max_value=5)
with row2[1]:
    insurance_plan = st.selectbox('Insurance Plan', categorical_options['Insurance Plan'])
with row2[2]:
    employment_status = st.selectbox('Employment Status', categorical_options['Employment Status'])

with row3[0]:
    gender = st.selectbox('Gender', categorical_options['Gender'])
with row3[1]:
    marital_status = st.selectbox('Marital Status', categorical_options['Marital Status'])
with row3[2]:
    bmi_category = st.selectbox('BMI Category', categorical_options['BMI Category'])

with row4[0]:
    smoking_status = st.selectbox('Smoking Status', categorical_options['Smoking Status'])
with row4[1]:
    region = st.selectbox('Region', categorical_options['Region'])
with row4[2]:
    medical_history = st.selectbox('Medical History', categorical_options['Medical History'])

# Create a dictionary for input values
input_dict = {
    'Age': age,
    'Number of Dependants': number_of_dependants,
    'Income in Lakhs': income_lakhs,
    'Genetical Risk': genetical_risk,
    'Insurance Plan': insurance_plan,
    'Employment Status': employment_status,
    'Gender': gender,
    'Marital Status': marital_status,
    'BMI Category': bmi_category,
    'Smoking Status': smoking_status,
    'Region': region,
    'Medical History': medical_history
}

# Button to make prediction
if st.button('Predict'):
    prediction = predict(input_dict)
    st.success(f'Predicted Health Insurance Cost: {prediction}')
```

## Reflections
1. This project provided a hands-on experience the ML pipeline in the insurance domain.
2. We navigated through critical stages of data cleaning, feature engineering, and multicollinearity handling.
3. Crafting domain-specific features like risk scores based on medical history was a great challenge.
4. Exploring different algorithms — Linear Regression, Ridge Regression, and XGBoost gave us a comparative perspective on model performance and generalization.
5. Finally, deploying the model via Streamlit transformed it into an interactive tool for business users.


## Conclusion
In conclusion, the Healthcare Premium Predictive Model succeeded in its Phase 1 goal — delivering a scalable, data-driven solution that can estimate health insurance premiums based on individual health profiles
