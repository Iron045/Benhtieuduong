import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import joblib
import os

# Load the dataset
data = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

# Chia dữ liệu thành features và labels
X = data.drop(columns=['Diabetes_binary'])  # 'Diabetes_binary' là cột mục tiêu
y = data['Diabetes_binary']

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Kiểm tra xem có mô hình checkpoint nào không
checkpoint_path = 'mlp_checkpoint.pkl'
if os.path.exists(checkpoint_path):
    model_mlp = joblib.load(checkpoint_path)
    st.write("Đã nạp mô hình từ checkpoint")
else:
    # Nếu không có checkpoint, khởi tạo mô hình mới
    model_mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1, warm_start=True, random_state=42)

# Huấn luyện mô hình trong các epoch, lưu checkpoint sau mỗi epoch
for epoch in range(10):  # Huấn luyện trong 10 lần lặp
    model_mlp.fit(X_train, y_train)
    
    # Lưu mô hình sau mỗi epoch
    joblib.dump(model_mlp, checkpoint_path)
    st.write(f"Mô hình đã được lưu sau epoch {epoch + 1}")

# Sau khi huấn luyện xong, lưu mô hình cuối cùng vào file
joblib.dump(model_mlp, 'mlp_model_final.pkl')
