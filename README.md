# 🏥 Insurance Premium Prediction | BFSI and Fintech

### Developed an annual insurance premium prediction model based on policy buyer's demographic & health profile using machine learning.

## 📌 Table of Contents
- <a href="#overview">Overview</a>
- <a href="#model-preview">Model Preview</a>
- <a href="#dataset">Dataset</a>
- <a href="#tools-technologies">Tools & Technologies</a>
- <a href="#project-structure">Project Structure</a>
- <a href="#data-cleaning-preparation">Data Cleaning & Preparation</a>
- <a href="#model-development">Model Development</a>
- <a href="#error-analysis">Error Analysis</a>
- <a href="#streamlit-app">Streamlit App</a>
- <a href="#author-contact">Author & Contact</a>

<h2><a class="anchor" id="overview"></a>📝 Overview</h2>

This project aims to develop a machine learning solution using historic data of existing policy holder's demographic & health profile using machine learning to help health insurance underwriters in the process of assessing the health risk associated with an individual and set the premium price accordingly.

- Develop a regression predictive model
- Ensure model accuracy is above 97%
- Ensure margin of error for residual must be below 10%
- Deploy a most viable product (MVP) using Streamlit application.

<h2><a class="anchor" id="model-preview"></a>🔗 Model Preview</h2>

![Model Preview](https://github.com/gaurav-patil-git/01_Insurance_Premium_Prediction_Model/blob/main/visuals/Premium.gif)

<h2><a class="anchor" id="credits"></a>🪪 Credits</h2>

This capstone project is a part of the “_Master Machine Learning for Data Science & AI: Beginner to Advanced_” course offered by **Codebasics** - All rights reserved.

- **Course Instructor**: Mr. Dhaval Patel
- **Platform**: codebasics.io – All rights reserved.

All education content and dataset used as learning resources belong to Codebasics and are protected under their respective rights and terms of use.

<h2><a class="anchor" id="dataset"></a>📊 Dataset</h2>

`.xlsx` files located in `/data/` folder


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

<h2><a class="anchor" id="tools-technologies"></a>🛠️ Tools & Technologies</h2>

| Task                 | Libraries Used                      |
|----------------------|-------------------------------------|
| Data Preprocessing   | Pandas                              |
| Data Visualization   | Matplotlib, Seaborn                 |
| Feature Engineering  | Pandas, Statsmodels, Scikit-learn   |
| Model Training       | Scikit-learn, XGBoost               |
| Model Fine Tuning    | Scikit-learn                        |
| UI Frontend          | Streamlit                           |

<h2><a class="anchor" id="project-structure"></a>📁 Project Structure</h2>

```
01_Insurance_Premium_Prediction_Model/
│
├── data/
│   ├── raw/               # Original, immutable data dumps
│   ├── processed/         # Cleaned & feature-engineered datasets
│
├── documents/             # Scope of work
│
├── models/                # Saved model and scaler objects 
│
├── mvp/                   # Minimum Viable Product (Streamlit app)
│
├── notebooks/             # Jupyter notebooks organized by purpose
│   ├── other/             # For other age groups
│   ├── overall/           # Full dataset
│   ├── young/             # For young population segment
│
├── visuals/               # Mockups and model preview
│
├── README.md              # High-level project overview
├── .gitignore             # Ignore data, models, logs if using Git

```

<h2><a class="anchor" id="data-cleaning-preparation"></a>🧼 Data Cleaning & Preparation</h2>

### **Data Cleaning**
- Identified and corrected inconsistent column header naming convention
- Detected and removed negligible (0.026 %) no. of rows with missing values
- Detected and handled anomalies :
  - Number Of Dependants : -3 (min value) -> used absolute values
  - Age : 356 (max value) -> Filtered the data
  - Income in lakhs: 930 (outlier max value) -> used 3 std as filter
- Split Medical History column into two for atomicity
- Corrected categories of smoking status : 'Smoking=0', 'Does Not Smoke', 'Not Smoking'

### **Feature Engineering**
- Applied One Hot Encoding on 4 columns
- Applied Label Encoding on 4 columns
- Derived total risk score using medical history
- Applied feature scaling
- Used VIF to identify multicolinearity and eliminated ones with high VIF
- Exported processed data and scaler object

<h2><a class="anchor" id="model-development"></a>🤖 Model Development</h2>

### **Model Training**
Performance table of different models with test rank:
| Regressor | Train | Test | Test Rank |
|---|---|---|---|
| XGBoost | 0.9861 | 0.9811 | 1 |
| Random Forest | 0.9965 | 0.9792 | 2 |
| Decision Tree | 0.9991 | 0.9633 | 3 |
| Linear | 0.9158 | 0.9143 | 4 |
| Lasso | 0.9158 | 0.9143 | 4 |
| Ridge | 0.9158 | 0.9143 | 4 |

- Since XGBoost performed better than others, we moved ahead with it for hyperparameter fine tuning using Grid Search CV.

<h2><a class="anchor" id="error-analysis"></a>⚠️ Error Analysis</h2>

- Through error analysis we realized that around 29% predictions from total predictions had margin of error above 10%.
- This would result in 29% policyholders to be significantly overcharged or undercharged.
- Further analysis, revealed that our was unable to predict premium amount for younger age (<= 25)
- We segmented our data into two segments and further developed two models
  - Model A (young) -> Younger age group (18-25)
  - Model B (other) -> Rest of the age groups
- Also, we request for more data from stakeholder to resolve the issue. They revert back with additional data of `genetical risk` only for young age group.

### **Model A (young)**
Performance table of different models with test rank:
| Regression | Train | Test | Test Rank |
|---|---|---|---|
| XGBoost | 0.9861 | 0.9811 | 1 |
| Random Forest | 0.9965 | 0.9792 | 2 |
| Decision Tree | 0.9991 | 0.9633 | 3 |
| Linear | 0.9158 | 0.9143 | 4 |
| Lasso | 0.9158 | 0.9143 | 4 |
| Ridge | 0.9158 | 0.9143 | 4 |

### **Model B (other)**
Performance table of different models with test rank:
| Regression | Train | Test | Test Rank |
|---|---|---|---|
| XGBoost | 0.9986 | 0.998 | 1 |
| Random Forest | 0.9997 | 0.9976 | 2 |
| Decision Tree | 1.0000 | 0.9958 | 3 |
| Linear | 0.9236 | 0.9224 | 4 |
| Lasso | 0.9236 | 0.9224 | 4 |
| Ridge | 0.9236 | 0.9224 | 4 |

<h2><a class="anchor" id="streamlit-app"></a>📱 Streamlit App</h2>

- Exported both respective models using `joblib`
- Developed and deployed a most viable product (MVP) using `streamlit`
- This MVP will be utilized by underwriters for 3 to 6 months for feedback and improvement before production

<h2><a class="anchor" id="author-contact"></a>📝 Author & Contact</h2>

**Gaurav Patil** (Data Analyst) 
- 🔗 [LinkedIn](https://www.linkedin.com/in/gaurav-patil-in/)


