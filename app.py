import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import time

# ---------- LOAD ----------
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #eef2f3, #dfe9f3);
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.title("💳 Credit Risk Prediction Dashboard")
st.caption("Smart AI-based Credit Risk Analysis System")

# ---------- SIDEBAR ----------
st.sidebar.header("📊 Customer Details")

limit_bal = st.sidebar.slider("💰 Credit Limit (₹)", 10000, 1000000, 100000, step=10000)
age = st.sidebar.slider("🎂 Age", 18, 80, 25)

sex = st.sidebar.selectbox("👤 Gender", ["Male", "Female"])
sex = 1 if sex == "Male" else 2

education = st.sidebar.selectbox("🎓 Education", ["Graduate", "University", "High School", "Others"])
education = {"Graduate":1,"University":2,"High School":3,"Others":4}[education]

marriage = st.sidebar.selectbox("💍 Marital Status", ["Married", "Single", "Others"])
marriage = {"Married":1,"Single":2,"Others":3}[marriage]

payment_status = st.sidebar.selectbox(
    "💳 Payment Behaviour",
    ["On Time", "1 Month Delay", "2+ Months Delay"]
)

pay_map = {"On Time": -1, "1 Month Delay": 1, "2+ Months Delay": 3}
pay_0 = pay_map[payment_status]
pay_2 = pay_map[payment_status]
pay_3 = pay_map[payment_status]

bill_amt1 = st.sidebar.slider("📄 Last Month Bill (₹)", 0, 200000, 50000, step=5000)
bill_amt2 = st.sidebar.slider("📄 2 Months Bill (₹)", 0, 200000, 40000, step=5000)

pay_amt1 = st.sidebar.slider("💸 Last Month Payment (₹)", 0, 200000, 30000, step=5000)
pay_amt2 = st.sidebar.slider("💸 2 Months Payment (₹)", 0, 200000, 25000, step=5000)

st.info("👉 Click Predict to see results")

# ---------- BUTTON ----------
if st.button("🔍 Predict Risk"):

    with st.spinner("Analyzing customer data..."):
        time.sleep(1.5)

    # Prepare input
    input_dict = {
        'LIMIT_BAL': limit_bal,
        'SEX': sex,
        'EDUCATION': education,
        'MARRIAGE': marriage,
        'AGE': age,
        'PAY_0': pay_0,
        'PAY_2': pay_2,
        'PAY_3': pay_3,
        'BILL_AMT1': bill_amt1,
        'BILL_AMT2': bill_amt2,
        'PAY_AMT1': pay_amt1,
        'PAY_AMT2': pay_amt2
    }

    features = pd.DataFrame([input_dict])[columns]

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    # ---------- RESULT ----------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Risk Result")

        if probability < 0.3:
            st.success(f"✅ LOW RISK ({probability:.2f})")
            risk = "Low"
            suggestion = "✔ Safe customer. Approve credit."
        elif probability < 0.6:
            st.warning(f"⚠ MEDIUM RISK ({probability:.2f})")
            risk = "Medium"
            suggestion = "⚠ Monitor customer. Limit credit."
        else:
            st.error(f"🚨 HIGH RISK ({probability:.2f})")
            risk = "High"
            suggestion = "❌ Risky customer. Avoid high credit."

    with col2:
        st.subheader("📈 Risk Meter")
        st.progress(int(probability * 100))
        st.metric("Default Probability", f"{probability:.2f}")

    # ---------- BAR CHART ----------
    st.subheader("📉 Bills vs Payments")

    chart_df = pd.DataFrame({
        "Category": ["Bill1", "Bill2", "Payment1", "Payment2"],
        "Amount": [bill_amt1, bill_amt2, pay_amt1, pay_amt2]
    })

    st.bar_chart(chart_df.set_index("Category"))

    # ---------- PIE CHART ----------
    st.subheader("📊 Risk Distribution")

    fig, ax = plt.subplots()
    ax.pie([probability, 1-probability],
           labels=["Default Risk", "Safe"],
           autopct='%1.1f%%',
           colors=["red", "green"])
    st.pyplot(fig)

    # ---------- SUMMARY ----------
    st.subheader("📋 Customer Summary")

    summary_df = pd.DataFrame({
        "Feature": ["Credit Limit", "Age", "Risk Level", "Prediction"],
        "Value": [
            f"₹ {limit_bal}",
            age,
            risk,
            "Default" if prediction == 1 else "No Default"
        ]
    })

    st.dataframe(summary_df, use_container_width=True)

    # ---------- DECISION ----------
    st.subheader("💡 Final Decision")

    st.info(suggestion)

    # ---------- SUCCESS ANIMATION ----------
    st.balloons()
