import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="E-Commerce Fraud Detection", page_icon="🛒", layout="wide")

st.title("🛍️ E-Commerce Fraud Detection App")
st.markdown("Predict whether a transaction is **Fraudulent or Genuine** using Machine Learning.")

# Sidebar for navigation
st.sidebar.header("🔍 Upload or Use Sample Data")
option = st.sidebar.radio("Choose option:", ("Use Sample Dataset", "Upload CSV"))

# Load data
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

if option == "Use Sample Dataset":
    df = load_data("dataset.csv")
else:
    uploaded = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

# Display data
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# EDA Section
with st.expander("🔎 Explore Data"):
    st.write("### Summary Statistics")
    st.write(df.describe())
    
    st.write("### Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='fraudulent', data=df, palette="viridis", ax=ax)
    st.pyplot(fig)

# Prepare data for model
X = df.drop('fraudulent', axis=1)
y = df['fraudulent']

# Handle categorical data
for col in X.select_dtypes(include='object'):
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Split and train model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X, y)

# Save model for reuse
joblib.dump(model, "model.pkl")

st.success("✅ Model trained successfully!")

# User input for prediction
st.sidebar.header("💡 Predict New Transaction")
input_data = {}
for col in X.columns:
    val = st.sidebar.text_input(f"{col}:", "0")
    input_data[col] = [float(val)]

input_df = pd.DataFrame(input_data)

if st.sidebar.button("🔮 Predict"):
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Genuine Transaction.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Author: Bharati Renge")
