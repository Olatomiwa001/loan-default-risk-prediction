import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import time
from model import load_model, predict
from src.loan_default_prediction import (
    generate_loan_data, 
    preprocess_data, 
    perform_hyperparameter_tuning, 
    train_and_evaluate_models
)

# Ensure the src folder is recognized
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./src'))

st.set_page_config(page_title="Loan Default Predictor", page_icon="💰", layout="wide")

st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select a Page", ["🏠 Home", "📊 Data Exploration", "📈 Loan Default Predictor", "📉 Model Performance", "📖 About Project"])

@st.cache_resource
def prepare_model():
    loan_data = generate_loan_data(n_samples=5000)
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(loan_data)
    tuned_models, tuning_results = perform_hyperparameter_tuning(X_train, y_train)
    results = train_and_evaluate_models(tuned_models, X_train, X_test, y_train, y_test)
    return loan_data, tuned_models, results, feature_names, scaler

loan_data, tuned_models, results, feature_names, scaler = prepare_model()

# if 'credit_score' in loan_data.columns and 'default' in loan_data.columns:
    
# else:
#     st.error("Error: Required columns missing in the dataset.")

if app_mode == "🏠 Home":
    st.title("🏦 Loan Default Risk Predictor")
    st.write("### Predict the risk of loan default using machine learning models")
    
    st.subheader("Loan Default Distribution")
    # fig, ax = plt.subplots(figsize=(8, 4))
    fig, ax = plt.subplots(figsize=(12, 6))

    # sns.countplot(data=loan_data, x='default', palette='coolwarm')
    sns.histplot(data=loan_data, x='credit_score', kde=True, bins=30, hue=loan_data['default'].astype(str))
    plt.title("Credit Score Distribution")
    # plt.title("Distribution of Loan Defaults")
    st.pyplot(fig)
    # st.pyplot(fig)


    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Data Samples", len(loan_data))
        st.metric("Default Rate", f"{loan_data['default'].mean()*100:.2f}%")
    # with col2:
    #     st.image("https://www.investopedia.com/thmb/2lxOt72b8Xwujhn8CfT5o4aV_Eo=/1500x1000/filters:no_upscale()/default-risk-5090763-FINAL-227b2487dddf4cfbbf16e1e49f22d8d5.png", width=400)

elif app_mode == "📊 Data Exploration":
    st.title("📊 Data Exploration")
    st.write("### Explore the dataset used for training the models")
    st.dataframe(loan_data.head(10))
    
    if 'credit_score' in loan_data.columns and 'default' in loan_data.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(data=loan_data, x='credit_score', kde=True, bins=30, hue=loan_data['default'].astype(str))
        plt.title("Credit Score Distribution")
        st.pyplot(fig)
    else:
        st.error("Error: 'credit_score' column missing in the dataset.")

elif app_mode == "📈 Loan Default Predictor":
    st.title("📈 Loan Default Risk Assessment")
    
    col1, col2 = st.columns(2)
    with col1:
        loan_amount = st.slider("Loan Amount ($)", 1000, 50000, 25000)
        annual_income = st.slider("Annual Income ($)", 20000, 200000, 75000)
        credit_score = st.slider("Credit Score", 300, 850, 650)
    with col2:
        employment_years = st.slider("Years of Employment", 0, 20, 5)
        debt_to_income_ratio = st.slider("Debt to Income Ratio", 0.1, 0.6, 0.3)
        loan_purpose = st.selectbox("Loan Purpose", ['education', 'home', 'business', 'personal'])
    
    input_data = pd.DataFrame({
        'loan_amount': [loan_amount],
        'annual_income': [annual_income],
        'credit_score': [credit_score],
        'employment_years': [employment_years],
        'debt_to_income_ratio': [debt_to_income_ratio],
        'loan_purpose': [loan_purpose]
    })
    
    input_encoded = pd.get_dummies(input_data, columns=['loan_purpose'])
    missing_cols = set(feature_names) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0
    input_encoded = input_encoded[feature_names]
    
    rf_model = tuned_models['Random Forest']
    default_prob = rf_model.predict_proba(input_encoded)[0][1]
    
    risk_level, color = ("Low Risk", "green") if default_prob < 0.3 else ("Medium Risk", "yellow") if default_prob < 0.6 else ("High Risk", "red")
    
    st.write(f"### Predicted Default Probability: {default_prob*100:.2f}%")
    st.markdown(f"### Risk Level: <span style='color:{color}'>{risk_level}</span>", unsafe_allow_html=True)

elif app_mode == "📉 Model Performance":
    st.title("📉 Model Performance")
    for model_name, result in results.items():
        st.write(f"### {model_name}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{result['accuracy']:.2f}")
        with col2:
            st.metric("ROC AUC", f"{result['roc_auc']:.2f}")
        st.text(result['classification_report'])

elif app_mode == "📖 About Project":
    st.title("📖 About Loan Default Risk Prediction")
    st.write("""
    This project aims to help financial institutions predict loan defaults using machine learning models.
    
    **Project Features:**
    - Uses real-world financial data to make accurate predictions.
    - Employs machine learning models like Random Forest and Gradient Boosting.
    - Provides an interactive dashboard for risk assessment.
    - Helps lenders minimize losses by predicting high-risk loans.
    
    **Methodology:**
    - Data collection from financial institutions.
    - Data preprocessing, feature engineering, and model training.
    - Hyperparameter tuning and evaluation using accuracy and AUC metrics.
    """)

st.sidebar.write("📌 Created for Data Science Portfolio")
