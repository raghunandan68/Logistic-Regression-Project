import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)


st.set_page_config(page_title="Heart Disease Prediction",layout="wide")

st.title("End-to-End Logistic Regression Platform")

st.header("Step 1 : Data Ingestion")

uploaded_file = st.file_uploader(
    "Upload Heart Dataset CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    raw_path = os.path.join(RAW_DIR, "heart_raw.csv")
    df.to_csv(raw_path, index=False)
    st.success("Dataset Uploaded Successfully")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())


    st.header("Step 2 : Exploratory Data Analysis")
    st.write("Shape of Dataset :", df.shape)
    st.subheader("Missing Values")
    st.write(df.isnull().sum())
    st.subheader("Data Types")
    st.write(df.dtypes)
    numeric_df = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)


    st.header("Step 3 : Data Cleaning")
    strategy = st.selectbox(
        "Select Missing Value Strategy",
        ["Mean", "Median", "Drop Rows"]
    )
    df_clean = df.copy()
    if strategy == "Drop Rows":
        df_clean = df_clean.dropna()
    elif strategy == "Mean":
        numeric_cols = df_clean.select_dtypes(include=np.number).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    elif strategy == "Median":
        numeric_cols = df_clean.select_dtypes(include=np.number).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

    st.subheader("Cleaned Dataset")
    st.dataframe(df_clean.head())
    if st.button("Save Cleaned Dataset"):
        clean_path = os.path.join(CLEAN_DIR,"cleaned_heart.csv")
        df_clean.to_csv(clean_path, index=False)
        st.success("Cleaned Dataset Saved")
        st.info(f"Saved at : {clean_path}")

    st.header("Step 4 : Load Cleaned Dataset")
    clean_files = [
        f for f in os.listdir(CLEAN_DIR)
        if "heart" in f.lower()
    ]
    if not clean_files:
        st.error("No cleaned dataset found")
        st.stop()
    selected = st.selectbox(
        "Select Dataset",
        clean_files
    )
    df_model = pd.read_csv(
        os.path.join(CLEAN_DIR, selected)
    )
    st.dataframe(df_model.head())

    st.header("Step 5 : Feature Encoding")
    categorical_cols = [
        "Sex",
        "ChestPainType",
        "RestingECG",
        "ExerciseAngina",
        "ST_Slope"
    ]
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le
    st.success("Categorical Features Encoded")
    st.dataframe(df_model.head())


    st.header("Step 6 : Train Logistic Regression Model")
    target = "HeartDisease"
    X = df_model.drop(columns=[target])
    y = df_model[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.25,
        random_state=42
    )

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.success("✅ Model Trained Successfully...")

    st.header("Step 7 : Model Evaluation")
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"Accuracy Score : {accuracy:.2f}")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    st.subheader("Classification Report")
    report = classification_report(
        y_test,
        y_pred
    )
    st.text(report)
else:
    st.warning("Please upload the heart dataset CSV file.")