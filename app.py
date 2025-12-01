import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
@st.cache_resource
def load_model():
    return joblib.load("rf_pipeline_model.pkl")

model = load_model()

# App title and sidebar
st.set_page_config(page_title="Random Forest Predictor", layout="wide")
st.title("🎯 Employee Exit Forecast")
st.markdown("Upload data or enter values manually to get predictions.")

st.sidebar.title("📊 Project Info")
st.sidebar.markdown("""
 🔍 *Objective*:
Predict the likelihood of employee attrition using historical HR data and machine learning techniques. 

                  
**🧠Model:** Random Forest Classifier  

**Pipeline:** Preprocessing + Classification 

🎯 Output:
Classifies employees into risk categories (Low, Medium, High) with confidence scores and contributing factors. 

💼 Business Impact:
Helps HR teams proactively identify at-risk employees, reduce turnover costs, and improve retention strategies.

**👨‍💻 Built By:** Akash Kumar Nayak  

**Use Case:** Predict outcome based on user input  
""")

# Input form
st.header("🔢 Manual Input")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        satisfaction_level= st.number_input(" satisfaction_level (e.g., 0.77)", value=0.77)
        number_project= st.number_input(" number_project (e.g., 3)", value=3)
        time_spend_company=st.number_input("time_spend_company (e.g., 3)", value=3)
        Department	= st.selectbox(" Department	", ['technical', 'support', 'IT', 'management', 'marketing', 'sales',
       'product_mng', 'accounting', 'RandD', 'hr'])

    with col2:
        last_evaluation= st.number_input("last_evaluation (e.g., 0.98)", value=0.98)
        average_montly_hours	= st.number_input("average_montly_hours (e.g., 286)", value=286)
        salary	= st.selectbox("salary", ['high', 'medium', 'low'])

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([{
            'satisfaction_level': satisfaction_level,
            'last_evaluation': last_evaluation,
            'number_project': number_project,
            'average_montly_hours':  average_montly_hours,
            'time_spend_company': time_spend_company,  # fixed or add as input
            'Department':  Department,
            'salary':  salary	
        }])
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        st.success(f"🎯 Prediction: {prediction}")
        st.info(f"Confidence: {max(proba)*100:.2f}%")

# Batch prediction
st.header("📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with the same columns as training data", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    preds = model.predict(df)
    df['Prediction'] = preds
    st.write(df)
    st.download_button("Download Predictions", df.to_csv(index=False), file_name="predictions.csv")

# Optional: Feature importance
if st.checkbox("📊 Show Feature Importance (if available)"):
    try:
        importances = model.named_steps['classifier'].feature_importances_
        feature_names = model.named_steps['preprocessing'].get_feature_names_out()
        fig, ax = plt.subplots()
        ax.barh(feature_names, importances)
        ax.set_title("Feature Importance")
        st.pyplot(fig)
    except Exception as e:
        st.warning("Feature importance not available for this pipeline.")

# Footer
st.markdown("---")


st.caption("Built with ❤️ by Akash Kumar Nayak")