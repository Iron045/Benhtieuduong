import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

def load_data():
    data = pd.read_csv('diabetes.csv')  # Load dataset
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    return X, y

def preprocess_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% train
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 15% val, 15% test
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler

def train_models(X_train, y_train):
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    
    logistic_model = LogisticRegression(random_state=42)
    logistic_model.fit(X_train, y_train)
    
    mlp_model = MLPClassifier(random_state=42, max_iter=500)
    mlp_model.fit(X_train, y_train)
    
    estimators = [('svm', svm_model), ('logistic', logistic_model)]
    stacking_model = StackingClassifier(estimators=estimators, final_estimator=mlp_model)
    stacking_model.fit(X_train, y_train)
    
    return svm_model, logistic_model, mlp_model, stacking_model

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC-AUC': roc_auc
        }

    return results
