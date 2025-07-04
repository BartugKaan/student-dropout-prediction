# 📦 Import necessary libraries
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from input_mapper import create_input_mapper

# --- App Configuration ---
st.set_page_config(page_title="University Dropout Prediction", page_icon="🎓", layout="wide")

# Initialize the input mapper
@st.cache_resource
def load_input_mapper():
    """Load the input mapper with scaler (cached for performance)"""
    return create_input_mapper()

mapper = load_input_mapper()

# --- Sidebar: Region Selection and Input Fields ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135755.png", width=100)
st.sidebar.title("Student Profile Setup")

# Region selection
st.sidebar.markdown("### 🌍 Select Your Region")
region = st.sidebar.selectbox(
    "Choose your academic system:",
    ["USA", "UK", "Türkiye"],
    help="Select the region that matches your academic background"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📝 Enter Details ({region} System)")

# Get the input schema for the selected region
schema = mapper.get_region_input_schema(region)

# Collect user inputs based on the selected region
user_inputs = {}

for field_key, field_config in schema.items():
    if field_config["type"] == "slider":
        if "step" in field_config:
            user_inputs[field_key] = st.sidebar.slider(
                field_config["label"],
                min_value=field_config["min"],
                max_value=field_config["max"],
                value=field_config["default"],
                step=field_config["step"],
                help=field_config["help"]
            )
        else:
            user_inputs[field_key] = st.sidebar.slider(
                field_config["label"],
                min_value=field_config["min"],
                max_value=field_config["max"],
                value=field_config["default"],
                help=field_config["help"]
            )
    elif field_config["type"] == "selectbox":
        user_inputs[field_key] = st.sidebar.selectbox(
            field_config["label"],
            field_config["options"],
            index=field_config["options"].index(field_config["default"]),
            help=field_config["help"]
        )

# Convert inputs to model format
model_data = mapper.map_to_model_format(region, user_inputs)

# --- Main Page Layout ---
col_main, col_side = st.columns([2, 1])

with col_main:
    st.image("https://images.unsplash.com/photo-1513258496099-48168024aec0?auto=format&fit=crop&w=800&q=80", use_container_width=True)
    st.title("🎓 University Dropout Prediction App")
    st.markdown(f"""
    This app predicts whether a university student is likely to **graduate** or **drop out**, 
    based on academic performance data adapted for **{region}** academic system.
    """)
    
    # Show region-specific explanation
    region_info = {
        "USA": "🇺🇸 Using US academic system with GPA (0-4.0), SAT scores, and credit hours",
        "UK": "🇬🇧 Using UK academic system with UCAS points, percentage grades, and modules",
        "Türkiye": "🇹🇷 Türkiye akademik sistemi kullanılıyor - YKS puanı, AKTS kredisi ve 0-100 notları"
    }
    
    st.info(region_info[region])
    
    st.markdown("---")
    st.subheader("Prediction Results")
    
    # --- Model Selection ---
    model_choice = st.radio(
        "Choose a prediction model:",
        ("Logistic Regression", "Random Forest", "XGBoost"),
        horizontal=True
    )
    
    if st.button("🔮 Predict Dropout Risk", type="primary"):
        try:
            # Load the selected model
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            model_path = {
                "Logistic Regression": os.path.join(project_root, "models/lr_model.pkl"),
                "Random Forest": os.path.join(project_root, "models/rf_model.pkl"),
                "XGBoost": os.path.join(project_root, "models/xgb_model.pkl")
            }
            
            model = joblib.load(model_path[model_choice])
            
            # Make prediction
            prediction = model.predict(model_data)
            prediction_proba = model.predict_proba(model_data)
            
            # Display results
            pred_label = "Dropout" if prediction[0] == 1 else "Graduate"
            dropout_probability = prediction_proba[0][1] * 100
            graduate_probability = prediction_proba[0][0] * 100
            
            # Create columns for results
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction[0] == 1:
                    st.error(f"⚠️ **High Risk**: Student likely to **drop out**")
                    st.metric("Dropout Probability", f"{dropout_probability:.1f}%")
                else:
                    st.success(f"✅ **Low Risk**: Student likely to **graduate**")
                    st.metric("Graduation Probability", f"{graduate_probability:.1f}%")
            
            with result_col2:
                # Create a simple probability chart
                fig, ax = plt.subplots(figsize=(6, 4))
                categories = ['Graduate', 'Dropout']
                probabilities = [graduate_probability, dropout_probability]
                colors = ['#28a745', '#dc3545']
                
                bars = ax.bar(categories, probabilities, color=colors, alpha=0.7)
                ax.set_ylabel('Probability (%)')
                ax.set_title('Prediction Probabilities')
                ax.set_ylim(0, 100)
                
                # Add probability labels on bars
                for bar, prob in zip(bars, probabilities):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{prob:.1f}%', ha='center', va='bottom')
                
                st.pyplot(fig)
                plt.close()
            
            st.markdown("---")
            
            # --- Feature Importance ---
            st.subheader("Feature Importance")
            try:
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                elif hasattr(model, "coef_"):
                    importances = np.abs(model.coef_[0])
                else:
                    importances = np.zeros(model_data.shape[1])
                
                feature_names = model_data.columns
                imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
                imp_df = imp_df.sort_values("Importance", ascending=True)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh(imp_df["Feature"], imp_df["Importance"], color="#4F8DFD")
                ax.set_xlabel("Importance")
                ax.set_title("Feature Importance in Prediction")
                ax.tick_params(axis='y', labelsize=10)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.info("Feature importance not available for this model.")
            
            st.markdown("---")
            
            # --- Input Summary ---
            st.subheader("Your Input Summary")
            
            # Create two columns for input summary
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.write("**Region-Specific Inputs:**")
                input_summary = mapper.get_input_summary(region, user_inputs)
                for label, value in input_summary.items():
                    st.write(f"• {label}: {value}")
            
            with summary_col2:
                st.write("**Model-Ready Format:**")
                st.dataframe(model_data.T, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.error("Please check that all model files are available and try again.")

with col_side:
    st.markdown("### About This App")
    st.info(
        """
        This app uses machine learning to predict student dropout risk using 
        region-specific academic data that's familiar to users.
        
        **Key Features:**
        - 🌍 Regional academic systems (USA, UK, Türkiye)
        - 📊 User-friendly input formats
        - 🎯 Multiple ML models for prediction
        - 📈 Feature importance analysis
        """
    )
    
    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown("""
    1. **Select your region** from the dropdown
    2. **Enter your academic details** using familiar terms
    3. **Choose a prediction model**
    4. **Click 'Predict'** to see results
    5. **Review** probability and feature importance
    """)
    
    st.markdown("---")
    st.markdown("### Regional Systems")
    
    with st.expander("🇺🇸 USA System"):
        st.markdown("""
        - **GPA**: 0-4.0 scale
        - **SAT**: 400-1600 range
        - **Credits**: Semester credit hours
        - **Grades**: Letter grade to GPA conversion
        """)
    
    with st.expander("🇬🇧 UK System"):
        st.markdown("""
        - **UCAS Points**: A-level equivalent
        - **Grades**: Percentage scale (0-100)
        - **Modules**: University course units
        - **Assessments**: Exams and coursework
        """)
    
    with st.expander("🇹🇷 Türkiye Sistemi"):
        st.markdown("""
        - **YKS Puanı**: 150-560 aralığı
        - **Notlar**: 0-100 ölçeği
        - **AKTS**: Avrupa kredi sistemi
        - **Sınavlar**: Dönem sonu sınavları
        """)

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    🎓 University Dropout Prediction System | Built with Streamlit & Machine Learning
    </div>
    """, 
    unsafe_allow_html=True
)
