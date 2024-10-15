import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import joblib

# Load the dataset
data = pd.read_csv('diabetes_health_indicators.csv')

# Kiểm tra thông tin về dữ liệu
# st.write(data.head())

# Chia dữ liệu thành features và labels
X = data.drop(columns=['Diabetes_binary'])  # 'Diabetes_binary' là cột mục tiêu
y = data['Diabetes_binary']

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Huấn luyện các mô hình
model_lr = LogisticRegression(random_state=42)
model_lr.fit(X_train, y_train)

model_svm = SVC(kernel='rbf', probability=True, random_state=42)
model_svm.fit(X_train, y_train)

model_mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
model_mlp.fit(X_train, y_train)

# Lưu mô hình và scaler để dùng cho dự đoán sau này
joblib.dump(model_lr, 'logistic_regression_model.pkl')
joblib.dump(model_svm, 'svm_model.pkl')
joblib.dump(model_mlp, 'mlp_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Giao diện Streamlit
st.title("Dự đoán bệnh tiểu đường dựa trên các chỉ số sức khỏe")

# Nhập liệu từ người dùng
high_bp = st.selectbox('Cao huyết áp', (0, 1))  # 0: Không, 1: Có
high_chol = st.selectbox('Cholesterol cao', (0, 1))
bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=25.0)
smoker = st.selectbox('Hút thuốc', (0, 1))  # 0: Không, 1: Có
stroke = st.selectbox('Đột quỵ', (0, 1))  # 0: Không, 1: Có
heart_disease = st.selectbox('Bệnh tim', (0, 1))  # 0: Không, 1: Có
phys_activity = st.selectbox('Hoạt động thể chất', (0, 1))  # 0: Không, 1: Có
age = st.number_input('Tuổi', min_value=18, max_value=120, value=50)

# Lựa chọn mô hình
model_choice = st.selectbox('Chọn phương pháp dự đoán', ('Logistic Regression', 'Support Vector Machine', 'Neural Network'))

# Khi người dùng bấm nút "Dự đoán"
if st.button("Dự đoán"):
    # Tạo dữ liệu đầu vào từ các input
    input_data = np.array([[high_bp, high_chol, bmi, smoker, stroke, heart_disease, phys_activity, age]])
    
    # Chuẩn hóa dữ liệu đầu vào
    scaler = joblib.load('scaler.pkl')
    input_data_scaled = scaler.transform(input_data)
    
    # Lựa chọn mô hình dự đoán
    if model_choice == 'Logistic Regression':
        model = joblib.load('logistic_regression_model.pkl')
    elif model_choice == 'Support Vector Machine':
        model = joblib.load('svm_model.pkl')
    elif model_choice == 'Neural Network':
        model = joblib.load('mlp_model.pkl')
    
    # Dự đoán và hiển thị kết quả
    prediction = model.predict(input_data_scaled)
    confidence = model.predict_proba(input_data_scaled)

    # Hiển thị kết quả dự đoán
    result = 'Có nguy cơ mắc bệnh tiểu đường' if prediction[0] == 1 else 'Không có nguy cơ mắc bệnh tiểu đường'
    st.write(f"Kết quả dự đoán: {result}")
    st.write(f"Độ tin cậy của dự đoán: {confidence.max() * 100:.2f}%")
