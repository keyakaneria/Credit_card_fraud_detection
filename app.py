import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

model = joblib.load("models/fraud_model.pkl")

raw_df = pd.read_csv("data/creditcard.csv")

df = raw_df.copy()

scaler = StandardScaler()

df["scaled_amount"] = scaler.fit_transform(df[["Amount"]])
df["scaled_time"] = scaler.fit_transform(df[["Time"]])

df.drop(["Time", "Amount"], axis=1, inplace=True)

st.title("💳 Credit Card Fraud Detection")

st.write(
    """
    This application uses a Random Forest model trained on
    credit card transaction data to identify potentially
    fraudulent transactions.
    """
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Transactions",
    f"{len(raw_df):,}"
)

col2.metric(
    "Fraud Transactions",
    int(raw_df["Class"].sum())
)

col3.metric(
    "Fraud %",
    f"{(raw_df['Class'].mean() * 100):.2f}%"
)

st.divider()

amount = st.number_input(
    "Enter Transaction Amount ($)",
    min_value=0.0,
    value=100.0,
    step=10.0
)

if st.button("Analyze Transaction"):

    idx = (raw_df["Amount"] - amount).abs().idxmin()

    actual_amount = raw_df.loc[idx, "Amount"]

    sample = df.iloc[idx].copy()

    X = sample.drop("Class").to_frame().T

    prediction = model.predict(X)[0]

    probability = model.predict_proba(X)[0][1]

    st.subheader("Analysis Result")

    st.write(
        f"Closest matching transaction amount in dataset: **${actual_amount:.2f}**"
    )

    if prediction == 1:
        st.error("⚠ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")

    st.metric(
        "Fraud Probability",
        f"{probability * 100:.2f}%"
    )