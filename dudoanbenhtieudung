import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
# Tải dữ liệu từ file CSV (sử dụng tập dữ liệu Pima Indians Diabetes)
data = pd.read_csv('diabetes.csv')

# Chia dữ liệu thành features và labels
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình Logistic Regression (hoặc có thể thay bằng mô hình khác như SVM, MLPClassifier)
model = LogisticRegression(C=1.0, penalty='l2', random_state=42)
model.fit(X_train, y_train)

# Lưu mô hình và scaler (nếu cần dùng lại)
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Giao diện Streamlit
st.title("Dự đoán bệnh tiểu đường")

# Các input từ người dùng
pregnancies = st.number_input('Số lần mang thai', min_value=0, max_value=20, value=1)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=100)
blood_pressure = st.number_input('Huyết áp', min_value=0, max_value=122, value=70)
skin_thickness = st.number_input('Độ dày da', min_value=0, max_value=99, value=20)
insulin = st.number_input('Insulin', min_value=0, max_value=900, value=85)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input('Tuổi', min_value=21, max_value=100, value=30)

# Khi người dùng bấm nút "Dự đoán"
if st.button("Dự đoán"):
    # Tạo dữ liệu đầu vào
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Chuẩn hóa dữ liệu đầu vào
    scaler = joblib.load('scaler.pkl')
    input_data_scaled = scaler.transform(input_data)
    
    # Dự đoán và hiển thị kết quả
    model = joblib.load('logistic_regression_model.pkl')
    prediction = model.predict(input_data_scaled)
    confidence = model.predict_proba(input_data_scaled)

    # Hiển thị kết quả
    result = 'Có nguy cơ mắc bệnh' if prediction[0] == 1 else 'Không có nguy cơ mắc bệnh'
    st.write(f"Kết quả dự đoán: {result}")
    st.write(f"Độ tin cậy của dự đoán: {confidence.max() * 100:.2f}%")


