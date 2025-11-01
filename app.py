# -------------------------------------------------------
# 💼 Expected CTC Prediction App using Streamlit
# -------------------------------------------------------

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -------------------------------------------------------
# Load trained model, scaler, and feature names
# -------------------------------------------------------
model = pickle.load(open("expected_ctc_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))

# -------------------------------------------------------
# Streamlit Page Configuration
# -------------------------------------------------------
st.set_page_config(page_title="Expected CTC Prediction", page_icon="💰", layout="wide")
st.title("💼 Expected CTC Prediction App")
st.write(
    "This app predicts a candidate’s **Expected CTC (₹)** using their professional and profile details. "
    "It’s powered by a tuned **XGBoost Regression Model** trained on real HR data."
)

# -------------------------------------------------------
# 📊 Part 1: Candidate Professional Information
# -------------------------------------------------------
st.subheader("📈 Candidate Professional Information")

col1, col2 = st.columns(2)

with col1:
    total_exp = st.number_input("Total Experience (Years)", min_value=0, max_value=50, step=1)
    field_exp = st.number_input("Experience in Applied Field (Years)", min_value=0, max_value=50, step=1)
    current_ctc = st.number_input("Current CTC (₹)", min_value=0, step=10000)
    no_of_companies = st.number_input("Number of Companies Worked", min_value=0, step=1)

with col2:
    publications = st.number_input("Number of Publications", min_value=0, step=1)
    certifications = st.number_input("Number of Certifications", min_value=0, step=1)
    intl_degree = st.selectbox("International Degree (Yes=1 / No=0)", [0, 1])
    appraisal = st.selectbox("Last Appraisal Rating", ['Excellent', 'Good', 'Average', 'Below Average'])

# -------------------------------------------------------
# 🎓 Part 2: Candidate Profile Details
# -------------------------------------------------------
st.subheader("🎓 Candidate Profile Details")

col3, col4 = st.columns(2)

with col3:
    department = st.selectbox("Department", ['Finance', 'HR', 'IT', 'Marketing', 'Operations', 'Sales'])
    role = st.selectbox("Role", ['Analyst', 'Consultant', 'Developer', 'Intern', 'Manager', 'Team Lead'])
    industry = st.selectbox("Industry", ['Banking', 'Education', 'Healthcare', 'Manufacturing', 'Software', 'Others'])

with col4:
    education = st.selectbox("Highest Education", ['Bachelors', 'Masters', 'PhD', 'Diploma'])
    location = st.selectbox("Current Location", ['Bangalore', 'Mumbai', 'Delhi', 'Pune', 'Hyderabad', 'Chennai'])
    pref_location = st.selectbox("Preferred Location", ['Bangalore', 'Mumbai', 'Delhi', 'Pune', 'Hyderabad', 'Chennai'])

# -------------------------------------------------------
# 🔍 Prediction Logic
# -------------------------------------------------------
if st.button("Predict Expected CTC 💰"):
    try:
        # Create empty DataFrame with all features
        input_df = pd.DataFrame(columns=feature_names)
        input_df.loc[0] = 0  # initialize all columns with 0

        # Fill numerical inputs
        input_df.loc[0, "Total_Experience"] = total_exp
        input_df.loc[0, "Total_Experience_in_field_applied"] = field_exp
        input_df.loc[0, "Current_CTC"] = current_ctc
        input_df.loc[0, "No_Of_Companies_worked"] = no_of_companies
        input_df.loc[0, "Number_of_Publications"] = publications
        input_df.loc[0, "Certifications"] = certifications
        input_df.loc[0, "International_degree_any"] = intl_degree

        # Encode categorical inputs (set 1 for selected category)
        def safe_set(col):
            if col in input_df.columns:
                input_df.loc[0, col] = 1

        safe_set(f"Department_{department}")
        safe_set(f"Role_{role}")
        safe_set(f"Industry_{industry}")
        safe_set(f"Education_{education}")
        safe_set(f"Curent_Location_{location}")
        safe_set(f"Preferred_location_{pref_location}")
        safe_set(f"Last_Appraisal_Rating_{appraisal}")

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict Expected CTC
        prediction = model.predict(input_scaled)[0]

        # Display result
        st.success(f"💰 Predicted Expected CTC: ₹ {prediction:,.2f}")

        # Optional: allow download of prediction
        output = pd.DataFrame({
            "Total_Experience": [total_exp],
            "Field_Experience": [field_exp],
            "Current_CTC": [current_ctc],
            "Predicted_Expected_CTC": [prediction]
        })
        csv = output.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Prediction as CSV",
            data=csv,
            file_name="predicted_ctc.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.markdown("---")
st.caption("Developed by Safwaan Siddiqui | Powered by XGBoost 💡")
st.caption("Data Source: LaunchED | Model Trained in Google Colab 🚀")
