import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    roc_curve, roc_auc_score
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def generate_loan_data(n_samples=2000):
    np.random.seed(42)
    data = {
        'loan_amount': np.random.uniform(1000, 50000, n_samples),
        'annual_income': np.random.uniform(20000, 200000, n_samples),
        'credit_score': np.random.normal(650, 100, n_samples).clip(300, 850),
        'employment_years': np.random.uniform(0, 20, n_samples),
        'debt_to_income_ratio': np.random.uniform(0.1, 0.6, n_samples),
        'age': np.random.uniform(18, 65, n_samples),
        'loan_purpose': np.random.choice(['education', 'home', 'business', 'personal'], n_samples),
        'default': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    }
    
    df = pd.DataFrame(data)
    df.loc[(df['credit_score'] < 500) | 
           (df['debt_to_income_ratio'] > 0.5) | 
           (df['employment_years'] < 1) | 
           (df['age'] < 25), 'default'] = 1
    return df

def preprocess_data(df):
    df_encoded = pd.get_dummies(df, columns=['loan_purpose'])
    X = df_encoded.drop('default', axis=1)
    y = df_encoded['default']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

def perform_hyperparameter_tuning(X_train, y_train):
    param_grids = {
        'Logistic Regression': {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l2']},
        'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]},
        'XGBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.3], 'max_depth': [3, 5, 7]}
    }
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
    }
    
    tuned_models = {}
    tuning_results = {}
    
    for name, model in models.items():
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        tuned_models[name] = grid_search.best_estimator_
        tuning_results[name] = {'Best Parameters': grid_search.best_params_, 'Best Score': grid_search.best_score_}
    
    return tuned_models, tuning_results

def cross_validation_performance(models, X_train, y_train):
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        cv_results[name] = {'Mean ROC AUC': scores.mean(), 'Standard Deviation': scores.std()}
    return cv_results

def train_and_evaluate_models(tuned_models, X_train, X_test, y_train, y_test):
    results = {}
    
    for name, model in tuned_models.items():
        if name == 'XGBoost':
            model.fit(X_train, y_train, eval_metric='logloss')
        else:
            model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
    
    return results
def train_and_evaluate_models(tuned_models, X_train, X_test, y_train, y_test):
    results = {}
    
    for name, model in tuned_models.items():
        model.fit(X_train, y_train)  # <-- REMOVE eval_metric from here
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
    
    return results

def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 6))
    
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
    
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    plt.title('ROC Curves for Different Models')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/roc_curves.png')
    plt.close()

def plot_feature_importance(models, X):
    rf_model = models['Random Forest']
    feature_importance = pd.DataFrame({'feature': X, 'importance': rf_model.feature_importances_}).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance in Random Forest Model')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png')
    plt.close()

def main():
    os.makedirs('results', exist_ok=True)  
    
    loan_data = generate_loan_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(loan_data)
    
    tuned_models, tuning_results = perform_hyperparameter_tuning(X_train, y_train)
    cv_performance = cross_validation_performance(tuned_models, X_train, y_train)
    results = train_and_evaluate_models(tuned_models, X_train, X_test, y_train, y_test)
    
    print("\n--- Model Performance ---")
    for model_name, metrics in results.items():
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {metrics['accuracy']:.2f}")
        print(f"ROC AUC: {metrics['roc_auc']:.2f}")
        print("Classification Report:\n", metrics['classification_report'])
    
    plot_roc_curves(tuned_models, X_test, y_test)
    plot_feature_importance(tuned_models, feature_names)
    
    return tuned_models, results

if __name__ == "__main__":
    models, results = main()
