import streamlit as st
import numpy as np
import train_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# Tải và tiền xử lý dữ liệu
X, y = train_model.load_data()
X_train, X_val, X_test, y_train, y_val, y_test, scaler = train_model.preprocess_data(X, y)

# Huấn luyện các mô hình
svm_model, logistic_model, mlp_model, stacking_model = train_model.train_models(X_train, y_train)

# Bản đồ các mô hình với tên gọi
models = {
    "SVM": svm_model,
    "Hồi quy Logistic": logistic_model,
    "Mạng nơ-ron": mlp_model,
    "Mô hình Stacking": stacking_model
}

# Giao diện Streamlit
st.title("Ứng dụng dự đoán bệnh tiểu đường")

# Thanh bên để chọn mô hình
st.sidebar.title("Chọn mô hình")
model_choice = st.sidebar.selectbox("Mô hình", list(models.keys()))

# Các trường nhập liệu cho dữ liệu bệnh nhân
st.write("## Nhập dữ liệu bệnh nhân:")
pregnancies = st.number_input('Số lần mang thai', min_value=0, max_value=20, value=6)
glucose = st.number_input('Mức glucose', min_value=0, max_value=200, value=148)
blood_pressure = st.number_input('Huyết áp', min_value=0, max_value=150, value=72)
skin_thickness = st.number_input('Độ dày da', min_value=0, max_value=100, value=35)
insulin = st.number_input('Mức insulin', min_value=0, max_value=900, value=0)
bmi = st.number_input('Chỉ số BMI', min_value=0.0, max_value=60.0, value=33.6)
dpf = st.number_input('Chức năng phả hệ tiểu đường', min_value=0.0, max_value=2.5, value=0.627)
age = st.number_input('Tuổi', min_value=1, max_value=100, value=50)

# Dữ liệu đầu vào để dự đoán
input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]).reshape(1, -1)
input_data_scaled = scaler.transform(input_data)

# Nút bấm để dự đoán
if st.button('Dự đoán'):
    model = models[model_choice]
    prediction = model.predict(input_data_scaled)
    result = 'Bệnh tiểu đường' if prediction == 1 else 'Không có bệnh tiểu đường'
    st.write(f"Kết quả dự đoán: {result}")

# Hiển thị độ chính xác và biểu đồ
if st.checkbox('Hiển thị độ chính xác và biểu đồ mô hình'):
    st.write("### Độ chính xác và biểu đồ:")
    accuracies = train_model.evaluate_models(models, X_test, y_test, X_train, y_train, X_val, y_val)
    
    for model_name, metrics in accuracies.items():
        st.write(f"**{model_name}:**")
        st.write(f"- Độ chính xác (tập huấn luyện): {metrics['train_accuracy']:.2f}")
        st.write(f"- Độ chính xác (tập validation): {metrics['val_accuracy']:.2f}")
        st.write(f"- Độ chính xác (tập kiểm tra): {metrics['Accuracy']:.2f}")
        st.write(f"- Độ chính xác ROC-AUC: {metrics['ROC-AUC']:.2f}" if metrics['ROC-AUC'] is not None else "")
        
        # Ma trận nhầm lẫn
        if model_name in models:
            model = models[model_name]
            try:
                conf_matrix = confusion_matrix(y_test, model.predict(X_test))
                plt.figure(figsize=(6, 4))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Ma trận nhầm lẫn cho {model_name}')
                plt.ylabel('Thực tế')
                plt.xlabel('Dự đoán')
                st.pyplot(plt.gcf())  # Hiển thị biểu đồ
                plt.clf()  # Xóa biểu đồ để vẽ biểu đồ khác

            except Exception as e:
                st.error(f"Lỗi khi tạo ma trận nhầm lẫn: {e}")
        
        # Đường cong ROC
        try:
            if hasattr(model, 'predict_proba'):
                fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
                plt.figure(figsize=(6, 4))
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {metrics["ROC-AUC"]:.2f})')
                plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
                plt.title(f'Đường cong ROC cho {model_name}')
                plt.xlabel('Tỷ lệ dương tính giả')
                plt.ylabel('Tỷ lệ dương tính thật')
                plt.legend()
                st.pyplot(plt.gcf())  # Hiển thị biểu đồ
                plt.clf()  # Xóa biểu đồ để vẽ biểu đồ khác
            else:
                st.warning(f"{model_name} không hỗ trợ dự đoán xác suất cho Đường cong ROC.")
        except Exception as e:
            st.error(f"Lỗi khi tạo đường cong ROC: {e}")
