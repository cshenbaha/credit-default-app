import streamlit as st
import pickle
import numpy as np

# Load model & columns
model = pickle.load(open('model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

st.title("💳 Credit Card Default Prediction")

st.write("Enter customer details:")

# Example inputs (you can expand later)
LIMIT_BAL = st.number_input("Credit Limit", min_value=0)
AGE = st.number_input("Age", min_value=18, max_value=100)
PAY_0 = st.number_input("Last Month Payment Delay", min_value=-2, max_value=8)

# Create input array (IMPORTANT: order must match training data)
input_data = np.zeros(len(columns))

# Map inputs to correct positions
for i, col in enumerate(columns):
    if col == 'LIMIT_BAL':
        input_data[i] = LIMIT_BAL
    elif col == 'AGE':
        input_data[i] = AGE
    elif col == 'PAY_0':
        input_data[i] = PAY_0

# Prediction
if st.button("Predict"):
    prediction = model.predict([input_data])[0]
    prob = model.predict_proba([input_data])[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Default ({prob:.2f})")
    else:
        st.success(f"✅ Low Risk ({prob:.2f})")