import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# =========================================
# 1️⃣ Helper: Handle Missing Columns
# =========================================
def ensure_all_columns(df, expected_columns):
    """Ensure all columns exist; fill missing ones with safe defaults."""
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        st.warning(f"Adding missing columns automatically: {missing_cols}")
        for col in missing_cols:
            if col in ['gender', 'job', 'merchant', 'category', 'city', 'state']:
                df[col] = 'Unknown'
            elif col in ['age', 'amt', 'zip', 'city_pop', 'time_diff_sec']:
                df[col] = 0
            elif col in ['lat', 'long', 'merch_lat', 'merch_long']:
                df[col] = 0.0
            else:
                df[col] = np.nan
    return df[expected_columns]

# =========================================
# 2️⃣ Load Model
# =========================================
@st.cache_resource
def load_model():
    return joblib.load("fraud_detection.pkl")

model = load_model()

# =========================================
# 3️⃣ Expected Columns
# =========================================
expected_columns = [
    'trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt',
    'first', 'last', 'gender', 'street', 'city', 'state', 'zip',
    'lat', 'long', 'city_pop', 'job', 'dob', 'unix_time',
    'merch_lat', 'merch_long', 'age', 'time_diff_sec'
]

# =========================================
# 4️⃣ Streamlit UI
# =========================================
st.set_page_config(page_title="Fraud Detection System", page_icon="💳", layout="wide")
st.title("💳 Fraud Detection App")

mode = st.radio("Select Input Mode:", ["📂 Upload CSV", "✍️ Manual Entry"])

# =========================================
# 📂 MODE 1: CSV Upload
# =========================================
if mode == "📂 Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"📄 Uploaded file: {df.shape[0]} rows, {df.shape[1]} columns")
        df_fixed = ensure_all_columns(df, expected_columns)
        preds = model.predict(df_fixed)
        proba = model.predict_proba(df_fixed)[:, 1]

        df["Predicted_Class"] = preds
        df["Fraud_Probability"] = proba
        st.success("✅ Predictions completed successfully!")
        st.dataframe(df.head(10))

        # Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", csv, "fraud_predictions.csv", "text/csv")

# =========================================
# ✍️ MODE 2: Manual Entry
# =========================================
else:
    st.subheader("Enter Transaction Details")

    col1, col2 = st.columns(2)
    with col1:
        cc_num = st.text_input("Credit Card Number", "60416207185")
        amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=120.50)
        merchant = st.text_input("Merchant Name", "fraud_Leffler-Goldner")
        category = st.selectbox("Category", ["shopping_net", "gas_transport", "personal_care", "misc_pos", "home", "travel"])
        city = st.text_input("City", "Fort Washakie")
        state = st.text_input("State", "WY")
        zip_code = st.number_input("ZIP Code", value=82514)
        city_pop = st.number_input("City Population", value=1763)
        job = st.text_input("Job", "Engineer")
        gender = st.selectbox("Gender", ["M", "F", "Unknown"])

    with col2:
        dob = st.date_input("Date of Birth", datetime(1990, 1, 1))
        trans_date = st.date_input("Transaction Date", datetime(2020, 6, 21))
        trans_time = st.time_input("Transaction Time", datetime.now().time())
        lat = st.number_input("Latitude", value=43.0)
        long = st.number_input("Longitude", value=-108.9)
        merch_lat = st.number_input("Merchant Latitude", value=43.01)
        merch_long = st.number_input("Merchant Longitude", value=-108.98)
        time_diff_sec = st.number_input("Time Difference (sec)", value=0.0)
        first = st.text_input("First Name", "John")
        last = st.text_input("Last Name", "Doe")
        street = st.text_input("Street", "123 Main St")

    # Combine into DataFrame
    trans_datetime = f"{trans_date} {trans_time}"
    unix_time = int(datetime.timestamp(datetime.combine(trans_date, trans_time)))
    age = int((datetime.now().year - dob.year))

    sample = pd.DataFrame([{
        'trans_date_trans_time': trans_datetime,
        'cc_num': cc_num,
        'merchant': merchant,
        'category': category,
        'amt': amt,
        'first': first,
        'last': last,
        'gender': gender,
        'street': street,
        'city': city,
        'state': state,
        'zip': zip_code,
        'lat': lat,
        'long': long,
        'city_pop': city_pop,
        'job': job,
        'dob': dob,
        'unix_time': unix_time,
        'merch_lat': merch_lat,
        'merch_long': merch_long,
        'age': age,
        'time_diff_sec': time_diff_sec
    }])

    # Ensure all columns and predict
    if st.button("🔍 Predict Fraud"):
        sample_fixed = ensure_all_columns(sample, expected_columns)
        pred = model.predict(sample_fixed)[0]
        prob = model.predict_proba(sample_fixed)[:, 1][0]

        st.markdown("### 🧾 Prediction Result:")
        st.write(f"**Fraud Probability:** {prob:.4f}")
        st.write("**Predicted Class:** 🚨 FRAUD" if pred == 1 else "✅ NOT FRAUD")
