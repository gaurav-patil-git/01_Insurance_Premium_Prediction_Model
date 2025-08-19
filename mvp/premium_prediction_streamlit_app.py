# Importing Libraries
import pandas as pd 
import streamlit as st
import joblib

# Loading Model & Scaler Obj.
model_young = joblib.load('01 Healthcare Premium Prediction/models/model_young.joblib')
model_other = joblib.load('01 Healthcare Premium Prediction/models/model_other.joblib')
scaler_obj = joblib.load('01 Healthcare Premium Prediction/models/scaler_obj_gr.joblib')

# Accessing scaler & list of columns
scaler = scaler_obj['scaler']
cols_to_scale = scaler_obj['cols_to_scale']

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
    gender = st.selectbox('Gender', categorical_options['Gender'])
with row1[2]:
    region = st.selectbox('Region', categorical_options['Region'])
    
with row2[0]:
    marital_status = st.selectbox('Marital Status', categorical_options['Marital Status'])
with row2[1]:
    employment_status = st.selectbox('Employment Status', categorical_options['Employment Status'])
with row2[2]:
    income_lakhs = st.number_input('Income in Lakhs', step=1, min_value=0, max_value=200)

with row3[0]:
    number_of_dependants = st.number_input('Number of Dependants', min_value=0, step=1, max_value=20)
with row3[1]:
    bmi_category = st.selectbox('BMI Category', categorical_options['BMI Category'])
with row3[2]:
    smoking_status = st.selectbox('Smoking Status', categorical_options['Smoking Status'])

with row4[0]:
    medical_history = st.selectbox('Medical History', categorical_options['Medical History'])
with row4[1]:
    genetical_risk = st.number_input('Genetical Risk', step=1, min_value=0, max_value=5)
with row4[2]:
    insurance_plan = st.selectbox('Insurance Plan', categorical_options['Insurance Plan'])

# Create a dictionary for input values
input_data = {
    'Age': age,
    'Gender': gender,
    'Region': region,
    'Marital Status': marital_status,
    'Employment Status': employment_status,
    'Income in Lakhs': income_lakhs,
    'Number of Dependants': number_of_dependants,
    'BMI Category': bmi_category,
    'Smoking Status': smoking_status,
    'Medical History': medical_history,
    'Genetical Risk': genetical_risk,
    'Insurance Plan': insurance_plan
}

feature_order = [
    'age', 
    'number_of_dependants', 
    'income_lakhs', 
    'bmi_category', 
    'smoking_status', 
    'insurance_plan', 
    'gender_Male', 
    'region_Northwest', 
    'region_Southeast', 
    'region_Southwest', 
    'marital_status_Unmarried', 
    'employment_status_Salaried', 
    'employment_status_Self-Employed', 
    'total_risk_score'
]

# Preprocessing applicant input data
def preprocessor(input_data):
    processed_cols = [
    # One hot encoded columns
    'gender_Male',
    'region_Northwest',
    'region_Southeast',
    'region_Southwest',
    'marital_status_Unmarried',
    'employment_status_Salaried',
    'employment_status_Self-Employed',

    # Label encoded columns
    'bmi_category',
    'smoking_status',
    'insurance_plan',

    # cols to scale
    'age',
    'number_of_dependants',
    'income_lakhs',
    'genetical_risk',
    'total_risk_score'
    ]

    # defining dataframe
    df = pd.DataFrame(0, columns= processed_cols, index= [0])

    # encoding map
    bmi_map = {'Underweight':0,'Normal':1,'Overweight':2,'Obesity':3}
    smoking_map = {'No Smoking':0,'Regular':1,'Occasional':2}
    insurance_map = {'Bronze':0,'Silver':1,'Gold':2}
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }

    for key, value in input_data.items():
        # One Hot Encoding
        if key == 'Gender' and value == 'Male':
            df['gender_Male'] = 1
        elif key == 'Region':
            if value == 'Northwest':
                df['region_Northwest'] = 1
            elif  value == 'Southeast':
                df['region_Southeast'] = 1
            elif value == 'Southwest':
                df['region_Southwest'] = 1
        elif key == 'Marital Status'and value == 'Unmarried':
            df['marital_status_Unmarried'] == 1
        elif key == 'Employment Status':
            if value == 'Salaried':
                df['employment_status_Salaried'] = 1
            elif value == 'Salaried':
                df['employment_status_Self-Employed'] = 1
        
        # Label Encoding
        elif key == 'BMI Category':
            df['bmi_category'] = bmi_map.get(value,1)
        elif key == 'Smoking Status':
            df['smoking_status'] = smoking_map.get(value,1)
        elif key == 'Insurance Plan':
            df['insurance_plan'] = insurance_map.get(value,1)

        # Cols to Scale
        elif key == 'Age':
            df['age'] = value
        elif key == 'Number of Dependants':
            df['number_of_dependants'] = value
        elif key == 'Income in Lakhs':
            df['income_lakhs'] = value
        elif key == 'Genetical Risk':
            df['genetical_risk'] = value
        
        # Calculate Risk Score
        elif key == 'Medical History':
            diseases = value.lower().split(" & ")
            total_risk_score = [risk_scores.get(disease,0) for disease in diseases]
            df['total_risk_score'] = sum(total_risk_score)
    
    # Inserting Proxy column 
    df[['income_level', 'disease1', 'disease2']] = None

    # Scaling
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    df.drop(columns= ['income_level', 'disease1', 'disease2'], axis= 1, inplace= True)
    
    return df

def predictor(input_data):
    df = preprocessor(input_data)

    if input_data['Age'] <= 25:
        prediction = model_young.predict(df[feature_order])
    else:
        prediction = model_other.predict(df.drop(columns= ['genetical_risk'])[feature_order])
    
    return int(prediction[0])

# Button to make prediction
if st.button('Predict'):
    prediction = predictor(input_data)
    st.success(f'Predicted Health Insurance Cost: {prediction:,.2f}')

