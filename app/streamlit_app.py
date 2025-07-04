# 📦 Import necessary libraries
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# --- App Configuration ---
st.set_page_config(page_title="University Dropout Prediction", page_icon="🎓", layout="wide")

# --- Sidebar: Input Fields ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135755.png", width=100)
st.sidebar.title("Student Features")
st.sidebar.markdown("""
Enter the student's academic and demographic details below. Each field includes a short explanation to help you provide accurate information.
""")

age = st.sidebar.slider(
    "🎂 Age at Enrollment",
    min_value=15, max_value=100, value=18,
    help="Student's age at the time of university enrollment."
)
admission_grade = st.sidebar.slider(
    "📝 Admission Grade",
    min_value=0.0, max_value=200.0, value=120.0,
    help="Grade used during university admission (0-200)."
)
tuition_up_to_date = st.sidebar.selectbox(
    "💰 Tuition Paid Up To Date?",
    ("No", "Yes"),
    help="Has the student paid tuition on time? Select 'Yes' if tuition is up to date."
)
tuition_up_to_date_val = 1 if tuition_up_to_date == "Yes" else 0
course_code = st.sidebar.number_input(
    "🏷️ Course Code (numeric)",
    min_value=1, value=1,
    help="Internal numeric course ID."
)
prev_qualification_grade = st.sidebar.slider(
    "📚 Previous Qualification Grade",
    min_value=0.0, max_value=200.0, value=100.0,
    help="Grade from previous qualification (0-200)."
)
curricular_approved_1st = st.sidebar.slider(
    "✅ Approved Units - 1st Semester",
    min_value=0, max_value=10, value=5,
    help="Number of curricular units approved in the 1st semester."
)
grade_1st = st.sidebar.slider(
    "📊 Average Grade - 1st Semester",
    min_value=0.0, max_value=20.0, value=10.0,
    help="Average grade in the 1st semester (0-20)."
)
curricular_approved_2nd = st.sidebar.slider(
    "✅ Approved Units - 2nd Semester",
    min_value=0, max_value=10, value=5,
    help="Number of curricular units approved in the 2nd semester."
)
grade_2nd = st.sidebar.slider(
    "📊 Average Grade - 2nd Semester",
    min_value=0.0, max_value=20.0, value=10.0,
    help="Average grade in the 2nd semester (0-20)."
)
evaluations_2nd = st.sidebar.slider(
    "🧪 Evaluations - 2nd Semester",
    min_value=0, max_value=20, value=5,
    help="Number of evaluations in the 2nd semester."
)

# --- Prepare Input Data ---
user_data = pd.DataFrame({
    "Age at enrollment": [age],
    "Admission grade": [admission_grade],
    "Tuition fees up to date": [tuition_up_to_date_val],
    "Curricular units 1st sem (approved)": [curricular_approved_1st],
    "Curricular units 2nd sem (approved)": [curricular_approved_2nd],
    "Curricular units 1st sem (grade)": [grade_1st],
    "Curricular units 2nd sem (grade)": [grade_2nd],
    "Curricular units 2nd sem (evaluations)": [evaluations_2nd],
    "Previous qualification (grade)": [prev_qualification_grade],
    "Course": [course_code]
})

# --- Main Page Layout ---
col_main, col_side = st.columns([2, 1])

with col_main:
    st.image("https://images.unsplash.com/photo-1513258496099-48168024aec0?auto=format&fit=crop&w=800&q=80", use_container_width=True)
    st.title("🎓 University Dropout Prediction App")
    st.markdown("""
    This app predicts whether a university student is likely to **graduate** or **drop out**, based on academic performance and background data.
    """)
    st.markdown("---")
    st.subheader("Prediction Results")
    # --- Model Selection (top of button) ---
    model_choice = st.radio(
        "Choose a model:",
        ("Logistic Regression", "Random Forest", "XGBoost"),
        horizontal=True
    )
    if st.button("🔮 Predict Dropout Risk"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        model_path = {
            "Logistic Regression": os.path.join(project_root, "models/lr_model.pkl"),
            "Random Forest": os.path.join(project_root, "models/rf_model.pkl"),
            "XGBoost": os.path.join(project_root, "models/xgb_model.pkl")
        }
        model = joblib.load(model_path[model_choice])
        prediction = model.predict(user_data)
        pred_label = "Dropout" if prediction[0] == 1 else "Graduate"
        if prediction[0] == 1:
            st.error(f"⚠️ **High Risk**: The student is likely to **drop out**.")
        else:
            st.success(f"✅ **Low Risk**: The student is likely to **graduate**.")
        st.markdown("---")
        # --- Feature Importance ---
        st.subheader("Feature Importance")
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_[0])
            else:
                importances = np.zeros(user_data.shape[1])
            feature_names = user_data.columns
            imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            imp_df = imp_df.sort_values("Importance", ascending=True)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(imp_df["Feature"], imp_df["Importance"], color="#4F8DFD")
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importance")
            st.pyplot(fig)
        except Exception as e:
            st.info("Feature importance not available for this model.")
        st.markdown("---")
        # --- Show User Input Table ---
        st.subheader("Entered Student Features")
        st.dataframe(user_data.T, use_container_width=True)

with col_side:
    st.markdown("### About")
    st.info(
        """
        This app uses machine learning models to predict student dropout risk. 
        - All input fields are explained in the sidebar.
        - Feature importance is visualized after prediction.
        - Your input is shown as a table for review.
        """
    )
    st.markdown("---")
    st.markdown("**How to use:**\n1. Enter student details in the sidebar.\n2. Select a model.\n3. Click 'Predict Dropout Risk'.\n4. View the prediction, feature importance, and your input.")
