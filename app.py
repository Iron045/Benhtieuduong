import streamlit as st
import numpy as np
import train_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# Load and preprocess data
X, y = train_model.load_data()
X_train, X_val, X_test, y_train, y_val, y_test, scaler = train_model.preprocess_data(X, y)

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
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=6)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=148)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150, value=72)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=35)
insulin = st.number_input('Insulin', min_value=0, max_value=900, value=0)
bmi = st.number_input('BMI', min_value=0.0, max_value=60.0, value=33.6)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.627)
age = st.number_input('Age', min_value=1, max_value=100, value=50)

# Input data to be used for prediction
input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]).reshape(1, -1)
input_data_scaled = scaler.transform(input_data)

# Button to predict
if st.button('Predict'):
    model = models[model_choice]
    prediction = model.predict(input_data_scaled)
    result = 'Diabetes' if prediction == 1 else 'No Diabetes'
    st.write(f"Prediction: {result}")

# Show model metrics and confusion matrix/ROC-AUC
if st.checkbox('Show Model Metrics and Plots'):
    st.write("### Model Metrics and Plots:")
    accuracies = train_model.evaluate_models(models, X_test, y_test)
    
    for model_name, metrics in accuracies.items():
        st.write(f"**{model_name}:**")
        st.write(f"- Accuracy: {metrics['Accuracy']:.2f}")
        st.write(f"- Precision: {metrics['Precision']:.2f}")
        st.write(f"- Recall: {metrics['Recall']:.2f}")
        st.write(f"- F1 Score: {metrics['F1 Score']:.2f}")
        if metrics['ROC-AUC'] is not None:
            st.write(f"- ROC-AUC: {metrics['ROC-AUC']:.2f}")
        
        # Confusion Matrix
        if model_name in models:
            model = models[model_name]
            try:
                conf_matrix = confusion_matrix(y_test, model.predict(X_test))
                plt.figure(figsize=(6, 4))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix for {model_name}')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(plt.gcf())  # Use Streamlit to display the current figure
                plt.clf()  # Clear the figure for the next plot

            except Exception as e:
                st.error(f"Error encountered while generating the confusion matrix: {e}")
        
        # ROC Curve
        try:
            if hasattr(model, 'predict_proba'):
                fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
                plt.figure(figsize=(6, 4))
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {metrics["ROC-AUC"]:.2f})')
                plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
                plt.title(f'ROC Curve for {model_name}')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend()
                st.pyplot(plt.gcf())  # Use Streamlit to display the current figure
                plt.clf()  # Clear the figure for the next plot
            else:
                st.warning(f"{model_name} does not support probability predictions for ROC Curve.")
        except Exception as e:
            st.error(f"Error encountered while generating the ROC curve: {e}")
