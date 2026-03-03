import streamlit as st
import pandas as pd
import requests
import os
import streamlit.components.v1 as components

st.set_page_config(page_title="Adult Income Prediction", layout="wide")

# --- Sidebar for navigation ---
page = st.sidebar.selectbox("Choose page", ["Single Prediction", "Batch Prediction & Profiling"])

# --- Shared features ---
features = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country"
]

# ------------------------
# Page 1: Single Prediction
# ------------------------
if page == "Single Prediction":
    st.header("Predict Income for a Single Example")

    # --- Inputs matching training features ---
    age = st.number_input("Age", min_value=0, max_value=120, value=37)
    workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Local-gov", "State-gov",
                                           "Federal-gov", "Self-emp-inc", "Without-pay", "Never-worked"])
    fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=1, value=284582)
    education = st.selectbox("Education", ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college",
                                           "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school",
                                           "5th-6th", "10th", "1st-4th", "Preschool", "12th"])
    marital_status = st.selectbox("Marital Status", ["Married-civ-spouse", "Divorced", "Never-married",
                                                     "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
    occupation = st.selectbox("Occupation", ["Exec-managerial", "Adm-clerical", "Handlers-cleaners",
                                             "Prof-specialty", "Other-service", "Sales", "Craft-repair",
                                             "Transport-moving", "Farming-fishing", "Machine-op-inspct",
                                             "Tech-support", "Protective-serv", "Priv-house-serv", "Armed-Forces"])
    relationship = st.selectbox("Relationship", ["Husband", "Not-in-family", "Wife", "Own-child", "Unmarried", "Other-relative"])
    race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
    sex = st.selectbox("Sex", ["Male", "Female"])
    education_num = st.number_input("Education Num", min_value=1, max_value=16, value=13)
    capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
    hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=168, value=40)
    native_country = st.selectbox("Native Country", ["United-States", "Canada", "Mexico", "Other"])

    input_data = {
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "education_num": education_num,
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": hours_per_week,
        "native_country": native_country
    }

    if st.button("Predict"):
        response = requests.post("http://127.0.0.1:8000/predict", json={"features": input_data})
        if response.status_code == 200:
            data = response.json()
            if "results" in data:
                result = data["results"][0]
                st.success(f"Predicted income: {result['prediction']}")
                if result["proba"]:
                    st.write("Prediction probabilities:", result["proba"])
            else:
                st.error(data.get("error", "Unknown error"))
        else:
            st.error(f"API returned status code {response.status_code}")
# ------------------------
# Page 2: Batch Prediction & Profiling
# ------------------------
elif page == "Batch Prediction & Profiling":
    st.header("Batch Prediction from CSV")

    uploaded_file = st.file_uploader("Upload CSV with features (optional 'income' column)", type="csv")
    if uploaded_file is not None:
        df_new = pd.read_csv(uploaded_file)

        # Extract y_true if present
        if "income" in df_new.columns:
            y_true = df_new["income"].str.strip().str.replace(".", "", regex=False)
            X_new = df_new.drop(columns=["income"])
        else:
            y_true = None
            X_new = df_new

        if st.button("Predict Batch"):
            # Spinner while calling API
            with st.spinner("Predicting batch, please wait..."):
                # Convert DataFrame to list of dicts for batch prediction
                features_list = X_new.to_dict(orient="records")
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json={"features": features_list}
                )

                if response.status_code == 200:
                    data = response.json()
                    if "results" in data:
                        predictions = [r.get("prediction") for r in data["results"]]
                    else:
                        st.error(data.get("error", "Unknown error"))
                        predictions = [None] * len(X_new)
                else:
                    st.error(f"API returned status code {response.status_code}")
                    predictions = [None] * len(X_new)

            # Progress bar while assigning predictions
            progress_bar = st.progress(0)
            df_new["predicted_income"] = None
            for i, pred in enumerate(predictions):
                df_new.at[i, "predicted_income"] = pred
                progress_bar.progress((i + 1) / len(predictions))
            progress_bar.empty()

            st.dataframe(df_new)

            # --- Compute metrics only on valid predictions ---
            if y_true is not None:
                valid_mask = [p in [">50K", "<=50K"] for p in predictions]
                y_true_valid = y_true[valid_mask]
                predictions_valid = [p for p, valid in zip(predictions, valid_mask) if valid]

                if len(predictions_valid) > 0:
                    from sklearn.metrics import accuracy_score, f1_score
                    correct_count = sum(y_true_valid == pd.Series(predictions_valid))
                    accuracy = accuracy_score(y_true_valid, predictions_valid)
                    f1 = f1_score(y_true_valid, predictions_valid, pos_label=">50K")
                    st.write(f"✅ Correct predictions: {correct_count}/{len(y_true_valid)}")
                    st.write(f"Accuracy: {accuracy:.4f}")
                    st.write(f"F1-score: {f1:.4f}")
                else:
                    st.warning("No valid predictions to compute metrics.")


                    
    # --- Profiling Report ---
    st.header("Profiling Report")
    profile_path = "artifacts/adult_dataset_profile.html"
    if os.path.exists(profile_path):
        with open(profile_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=800, scrolling=True)
    else:
        st.warning("Profiling report not found. Please run the profiling step first.")