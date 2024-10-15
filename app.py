# app.py
import streamlit as st
import numpy as np
import train_model

# Load and preprocess data
X, y = train_model.load_data()
X_train, X_test, y_train, y_test, scaler = train_model.preprocess_data(X, y)

# Train models
svm_model, logistic_model, mlp_model, stacking_model = train_model.train_models(X_train, y_train)

# Map models to names
models = {
    "SVM": svm_model,
    "Logistic Regression": logistic_model,
    "Neural Network": mlp_model,
    "Stacking": stacking_model
}

# Streamlit UI
st.title("Diabetes Prediction App")

# Sidebar for model selection
st.sidebar.title("Choose Model")
model_choice = st.sidebar.selectbox("Model", list(models.keys()))

# Input fields for user data
st.write("## Input Patient Data:")
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=85)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150, value=70)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=30)
insulin = st.number_input('Insulin', min_value=0, max_value=900, value=100)
bmi = st.number_input('BMI', min_value=0.0, max_value=60.0, value=25.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input('Age', min_value=1, max_value=100, value=30)

# Input data to be used for prediction
input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]).reshape(1, -1)
input_data_scaled = scaler.transform(input_data)

# Button to predict
if st.button('Predict'):
    model = models[model_choice]
    prediction = model.predict(input_data_scaled)
    result = 'Diabetes' if prediction == 1 else 'No Diabetes'
    st.write(f"Prediction: {result}")

# Show model accuracy
if st.checkbox('Show Model Accuracy'):
    accuracies = train_model.evaluate_models(models, X_test, y_test)
    st.write("### Model Accuracies:")
    for model_name, accuracy in accuracies.items():
        st.write(f"{model_name}: {accuracy:.2f}")
