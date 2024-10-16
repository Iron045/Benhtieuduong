import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Tải và xử lý dữ liệu
def load_data():
    data = pd.read_csv('diabetes.csv')
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    return X, y

def preprocess_data(X, y):
    # Chia dữ liệu thành 3 tập: training, validation, testing
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler

# Huấn luyện các mô hình
def train_models(X_train, y_train):
    # Tuning các mô hình cơ sở
    param_svm = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 1]
    }
    param_logistic = {
        'C': [0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear', 'newton-cg', 'saga'],
        'max_iter': [100, 200, 300]
    }
    param_mlp = {
        'hidden_layer_sizes': [(50,), (100,), (150,), (200,)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'learning_rate_init': [0.001, 0.01, 0.1]
    }
    
    # SVM
    svm_model = GridSearchCV(SVC(probability=True, random_state=42), param_svm, cv=5, scoring='accuracy')
    svm_model.fit(X_train, y_train)
    
    # Logistic Regression
    logistic_model = GridSearchCV(LogisticRegression(random_state=42), param_logistic, cv=5, scoring='accuracy')
    logistic_model.fit(X_train, y_train)
    
    # MLPClassifier
    mlp_model = GridSearchCV(MLPClassifier(max_iter=500, random_state=42), param_mlp, cv=5, scoring='accuracy')
    mlp_model.fit(X_train, y_train)
    
    # StackingClassifier
    base_learners = [
        ('svm', svm_model.best_estimator_),
        ('logistic', logistic_model.best_estimator_)
    ]
    stacking_model = StackingClassifier(estimators=base_learners, final_estimator=mlp_model.best_estimator_)
    stacking_model.fit(X_train, y_train)
    
    return svm_model, logistic_model, mlp_model, stacking_model

# Đánh giá mô hình
def evaluate_models(models, X_train, y_train, X_val, y_val, X_test, y_test):
    results = {}
    for model_name, model in models.items():
        # Dự đoán trên các tập
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        # Tính toán các chỉ số
        train_accuracy = accuracy_score(y_train, y_pred_train)
        val_accuracy = accuracy_score(y_val, y_pred_val)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        
        # Tính ROC-AUC
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = None
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
        plt.title(f'Confusion Matrix for {model_name}')
        st.pyplot()
        
        # Lưu kết quả
        results[model_name] = {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC-AUC': roc_auc
        }
    
    return results

# Giao diện Streamlit
st.title("Đánh Giá Mô Hình Tiểu Đường")
st.write("Chọn các mô hình để huấn luyện và đánh giá:")

# Tải dữ liệu
X, y = load_data()

# Tiền xử lý dữ liệu
X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_data(X, y)

# Huấn luyện mô hình
models = {
    'SVM': train_models(X_train, y_train)[0],
    'Logistic Regression': train_models(X_train, y_train)[1],
    'MLP Classifier': train_models(X_train, y_train)[2],
    'Stacking': train_models(X_train, y_train)[3]
}

# Đánh giá mô hình
results = evaluate_models(models, X_train, y_train, X_val, y_val, X_test, y_test)

# Hiển thị kết quả
for model_name, metrics in results.items():
    st.subheader(f"**{model_name}**:")
    for metric, value in metrics.items():
        st.write(f"{metric}: {value:.2f}")
