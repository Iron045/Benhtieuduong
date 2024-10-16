import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_data():
    # Tải dữ liệu
    data = pd.read_csv('diabetes.csv')
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    return X, y

def preprocess_data(X, y):
    # Chia dữ liệu thành tập huấn luyện, validation và kiểm tra
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% huấn luyện
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 15% validation, 15% kiểm tra
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler

def train_models(X_train, y_train):
    # Huấn luyện mô hình SVM
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    
    # Huấn luyện mô hình Hồi quy Logistic
    logistic_model = LogisticRegression(random_state=42)
    logistic_model.fit(X_train, y_train)
    
    # Huấn luyện mô hình Mạng nơ-ron
    mlp_model = MLPClassifier(random_state=42, max_iter=500)
    mlp_model.fit(X_train, y_train)
    
    # Sử dụng Stacking cho các mô hình
    estimators = [('svm', svm_model), ('logistic', logistic_model)]
    stacking_model = StackingClassifier(estimators=estimators, final_estimator=mlp_model)
    stacking_model.fit(X_train, y_train)
    
    return svm_model, logistic_model, mlp_model, stacking_model

def evaluate_models(models, X_test, y_test, X_train, y_train, X_val, y_val):
    results = {}
    for name, model in models.items():
        # Dự đoán trên tập kiểm tra
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        # Độ chính xác tập huấn luyện và validation
        train_accuracy = accuracy_score(y_train, y_pred_train)
        val_accuracy = accuracy_score(y_val, y_pred_val)
        
        # Độ chính xác trên tập kiểm tra
        accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        
        # Tính ROC-AUC nếu mô hình có thuộc tính predict_proba
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = None
        
        results[name] = {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC-AUC': roc_auc
        }
    return results
