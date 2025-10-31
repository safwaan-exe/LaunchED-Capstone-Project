# -------------------------------------------------------
# 💼 Expected CTC Prediction App using Streamlit
# -------------------------------------------------------

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -------------------------------------------------------
# Load model, scaler, and feature names
# -------------------------------------------------------
model = pickle.load(open("expected_ctc_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))  # saved from Colab

# -------------------------------------------------------
# Streamlit Page Configuration
# -------------------------------------------------------
st.set_page_config(page_title="Expected CTC Prediction", page_icon="💰", layout="wide")
st.title("💼 Expected CTC Prediction App")
st.write(
    "This app predicts a candidate’s **Expected CTC (₹)** based on their professional and academic details, "
    "using a tuned **XGBoost Regression Model**."
)

# -------------------------------------------------------
# Input Section
# -------------------------------------------------------
st.subheader("🧾 Enter Candidate Details")

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

# -------------------------------------------------------
# Prediction Logic
# -------------------------------------------------------
if st.button("🔍 Predict Expected CTC"):
    try:
        # Create an empty dataframe with all feature columns
        input_df = pd.DataFrame(columns=feature_names)
        input_df.loc[0] = 0  # initialize all columns to 0

        # Fill in numeric input values
        input_df.loc[0, "Total_Experience"] = total_exp
        input_df.loc[0, "Total_Experience_in_field_applied"] = field_exp
        input_df.loc[0, "Current_CTC"] = current_ctc
        input_df.loc[0, "No_Of_Companies_worked"] = no_of_companies
        input_df.loc[0, "Number_of_Publications"] = publications
        input_df.loc[0, "Certifications"] = certifications
        input_df.loc[0, "International_degree_any"] = intl_degree

        # Scale input features
        input_scaled = scaler.transform(input_df)

        # Predict Expected CTC
        prediction = model.predict(input_scaled)[0]

        # Display output
        st.success(f"💰 Predicted Expected CTC: ₹ {prediction:,.2f}")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.markdown("---")
st.caption("Developed by Safwaan Siddiqui | Powered by XGBoost 💡")
st.caption("Data Source: LaunchED | Model Trained in Google Colab 🚀")
