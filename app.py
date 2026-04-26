import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. LOAD ASSETS ---
model = pickle.load(open('house_model.pkl', 'rb'))
scaler = pickle.load(open('house_scaler.pkl', 'rb'))

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("Real Estate Price Estimator 🏠")
st.markdown("Enter property details below to estimate the market value.")

# --- 2. USER INPUTS (Based on your CSV columns) ---
col1, col2 = st.columns(2)

with col1:
    sq_ft = st.number_input('Square Footage', min_value=500, max_value=10000, value=2000)
    bedrooms = st.number_input('Number of Bedrooms', 1, 10, 3)
    bathrooms = st.number_input('Number of Bathrooms', 1, 10, 2)
    year_built = st.number_input('Year Built', 1800, 2024, 2000)

with col2:
    lot_size = st.number_input('Lot Size (Acres)', 0.1, 20.0, 0.5)
    garage_size = st.number_input('Garage Size (Cars)', 0, 5, 2)
    neighborhood = st.slider('Neighborhood Quality (1-10)', 1, 10, 5)

# --- 3. DATA PREPARATION ---
# Create DataFrame with exact column names from training
input_data = pd.DataFrame({
    'Square_Footage': [sq_ft],
    'Num_Bedrooms': [bedrooms],
    'Num_Bathrooms': [bathrooms],
    'Year_Built': [year_built],
    'Lot_Size': [lot_size],
    'Garage_Size': [garage_size],
    'Neighborhood_Quality': [neighborhood]
})

# --- 4. SCALING & PREDICTION ---
if st.button('Estimate House Price'):
    # Scale inputs using the saved scaler
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    
    st.divider()
    st.subheader(f"Estimated Market Value:")
    st.success(f"### ${prediction:,.2f}")
    
    # Visual feedback based on quality
    if neighborhood >= 8:
        st.info("💎 This property is in a high-tier neighborhood.")