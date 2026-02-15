# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Metrics
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(page_title="Auto ML Suite", layout="wide")

st.title("🤖 Automatic ML System")
st.caption("Regression | Classification | Clustering")

# ---------------- CSS ----------------

st.markdown("""
<style>
.best-model-card {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin-top: 20px;
}

.prediction-card {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# UPLOAD DATASET
# ---------------------------------------------------

uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    raw_data = pd.read_csv(uploaded_file)
    st.subheader("📌 Dataset Preview")
    st.dataframe(raw_data.head())

    data = raw_data.copy().drop_duplicates()

    for col in ["id", "Id", "ID", "date", "timestamp"]:
        if col in data.columns:
            data = data.drop(columns=[col])

    # ---------------------------------------------------
    # TARGET & INPUT SELECTION
    # ---------------------------------------------------

    st.subheader("🎯 Select Target Column")

    target_column = st.selectbox("Target Column", data.columns)

    input_columns = st.multiselect(
        "🧩 Select Input Features",
        [c for c in data.columns if c != target_column]
    )

    # ---------------------------------------------------
    # TRAIN MODELS
    # ---------------------------------------------------

    if st.button("🧠 Train Models"):

        if not input_columns:
            st.warning("⚠️ Please select at least one input feature")
            st.stop()

        X = data[input_columns]
        y = data[target_column]

        X = pd.get_dummies(X, drop_first=True)

        if pd.api.types.is_numeric_dtype(y):
            problem_type = "Regression"
        else:
            problem_type = "Classification"

        st.session_state.problem_type = problem_type

        label_encoder = None
        if problem_type == "Classification":
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y if problem_type == "Classification" else None
        )

        results = []

        if problem_type == "Regression":

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "KNN": KNeighborsRegressor()
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                results.append([
                    name,
                    mean_squared_error(y_test, y_pred),
                    np.sqrt(mean_squared_error(y_test, y_pred)),
                    r2_score(y_test, y_pred)
                ])

            results_df = pd.DataFrame(
                results, columns=["Algorithm", "MSE", "RMSE", "R2 Score"]
            )

            best_model_name = results_df.loc[
                results_df["R2 Score"].idxmax(), "Algorithm"
            ]

        else:

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "SVM": SVC(probability=True),
                "Naive Bayes": GaussianNB()
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                results.append([
                    name,
                    accuracy_score(y_test, y_pred),
                    precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    recall_score(y_test, y_pred, average="weighted"),
                    f1_score(y_test, y_pred, average="weighted")
                ])

            results_df = pd.DataFrame(
                results,
                columns=["Algorithm", "Accuracy", "Precision", "Recall", "F1 Score"]
            )

            best_model_name = results_df.loc[
                results_df["F1 Score"].idxmax(), "Algorithm"
            ]

        # Store models
        trained_models = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model

        st.session_state.trained_models = trained_models
        st.session_state.best_model = trained_models[best_model_name]
        st.session_state.best_model_name = best_model_name
        st.session_state.results_df = results_df
        st.session_state.input_columns = input_columns
        st.session_state.target_column = target_column
        st.session_state.label_encoder = label_encoder

        st.markdown(f"""
        <div class="best-model-card">
            <h2>🏆 Best Model Selected</h2>
            <h1>{best_model_name}</h1>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------
    # SHOW RESULTS
    # ---------------------------------------------------

    if "results_df" in st.session_state:
        st.subheader("📊 Model Comparison")
        st.dataframe(st.session_state.results_df)

    # ---------------------------------------------------
    # REGRESSION GRAPH
    # ---------------------------------------------------

    if (
        "problem_type" in st.session_state and
        st.session_state.problem_type == "Regression" and
        "best_model" in st.session_state
    ):

        st.subheader("📈 Regression Graph: Actual vs Predicted")

        X = pd.get_dummies(data[st.session_state.input_columns], drop_first=True)
        y = data[st.session_state.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        y_pred = st.session_state.best_model.predict(X_test)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_test, y_pred, alpha=0.7)

        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())

        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted")

        st.pyplot(fig)

    # ---------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------

    if "trained_models" in st.session_state:

        st.subheader("🔮 Make Prediction & Compare Models")

        user_input = []
        for col in st.session_state.input_columns:
            user_input.append(
                st.number_input(f"Enter value for {col}", value=0.0)
            )

        if st.button("🚀 Compare All Models"):

            user_df = pd.DataFrame(
                [user_input],
                columns=st.session_state.input_columns
            )

            predictions = []
            best_model_name = st.session_state.best_model_name

            for name, model in st.session_state.trained_models.items():

                pred = model.predict(user_df)[0]

                if st.session_state.problem_type == "Classification":
                    pred = st.session_state.label_encoder.inverse_transform([pred])[0]

                status = "🏆 Best Model" if name == best_model_name else ""
                predictions.append([name, pred, status])

            comparison_df = pd.DataFrame(
                predictions,
                columns=["Model", "Prediction", "Status"]
            )

            best_row = comparison_df[comparison_df["Status"] == "🏆 Best Model"]

            if not best_row.empty:
                best_prediction = best_row["Prediction"].values[0]
                best_model_used = best_row["Model"].values[0]

                st.markdown(f"""
                <div class="prediction-card">
                    <h2>🎯 Best Model Prediction</h2>
                    <h1>{best_prediction}</h1>
                    <p>Model Used: <b>{best_model_used}</b></p>
                </div>
                """, unsafe_allow_html=True)

            st.subheader("📊 All Model Predictions")
            st.dataframe(comparison_df)

else:
    st.info("⬆️ Upload a CSV file to begin")
