import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns


# ========= Data loading & preprocessing ========= #

@st.cache_data
def load_data(csv_path: str = "E Commerce Dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Impute numeric missing values with median
    for col in df.columns:
        if df[col].isnull().sum() > 0 and np.issubdtype(df[col].dtype, np.number):
            df[col].fillna(df[col].median(), inplace=True)

    return df


def build_preprocessor(df: pd.DataFrame):
    # Categorical columns from the notebook
    categorical_cols = [
        "PreferredLoginDevice",
        "PreferredPaymentMode",
        "Gender",
        "PreferedOrderCat",
        "MaritalStatus",
    ]

    # Numeric = everything else except target and categoricals
    numeric_cols = [c for c in df.columns if c not in categorical_cols + ["Churn"]]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ]
    )

    return preprocessor, categorical_cols, numeric_cols


# ========= Model training ========= #

@st.cache_resource
def train_models(df: pd.DataFrame):
    preprocessor, categorical_cols, numeric_cols = build_preprocessor(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        "SVM": SVC(
            probability=True,
            class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    trained_models = {}
    metrics_table = []
    reports = {}

    for name, model in models.items():
        clf = ImbPipeline(
            steps=[
                ("preprocess", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("model", model),
            ]
        )

        clf.fit(X_train, y_train)
        trained_models[name] = clf

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        metrics_table.append(
            {
                "Model": name,
                "Accuracy": acc,
                "ROC AUC": roc,
            }
        )

        reports[name] = classification_report(y_test, y_pred)

    metrics_df = pd.DataFrame(metrics_table).set_index("Model")
    return trained_models, metrics_df, reports, X_test, y_test, categorical_cols, numeric_cols


# ========= UI helpers ========= #

def show_eda(df: pd.DataFrame):
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write("Shape:", df.shape)

    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    df["Churn"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("Churn")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("Numeric Summary")
    st.dataframe(df.describe())


def show_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


def build_single_input_form(df, categorical_cols, numeric_cols):
    st.subheader("Customer Features")

    input_data = {}

    # Categorical inputs
    for col in categorical_cols:
        options = sorted(df[col].dropna().unique())
        if len(options) == 0:
            val = ""
        else:
            val = st.selectbox(col, options, index=0)
        input_data[col] = [val]

    # Numeric inputs
    for col in numeric_cols:
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        col_med = float(df[col].median())
        val = st.number_input(
            col,
            min_value=col_min,
            max_value=col_max,
            value=col_med,
        )
        input_data[col] = [val]

    return pd.DataFrame(input_data)


# ========= Streamlit app ========= #

def main():
    st.set_page_config(
        page_title="E-Commerce Churn Prediction",
        layout="wide"
    )

    st.title("E-Commerce Customer Churn Prediction")

    st.markdown(
        """
        This app loads the e-commerce dataset, performs basic EDA,
        trains several classification models with SMOTE and a preprocessing
        pipeline, and allows interactive churn predictions for a single customer.
        """
    )

    # Data
    df = load_data()

    # Train models
    (
        trained_models,
        metrics_df,
        reports,
        X_test,
        y_test,
        categorical_cols,
        numeric_cols,
    ) = train_models(df)

    tabs = st.tabs(
        ["Dataset & EDA", "Model Performance", "Single Prediction", "Feature Importance"]
    )

    # ---- Tab 1: Dataset & EDA ---- #
    with tabs[0]:
        show_eda(df)

    # ---- Tab 2: Model Performance ---- #
    with tabs[1]:
        st.subheader("Overall Metrics")
        st.dataframe(metrics_df.style.format({"Accuracy": "{:.3f}", "ROC AUC": "{:.3f}"}))

        model_name = st.selectbox(
            "Select a model to inspect detailed report",
            list(trained_models.keys()),
            index=list(trained_models.keys()).index("XGBoost")
            if "XGBoost" in trained_models
            else 0,
        )

        st.markdown(f"### Classification Report — {model_name}")
        st.text(reports[model_name])

        # Confusion matrix
        clf = trained_models[model_name]
        y_pred = clf.predict(X_test)
        st.markdown(f"### Confusion Matrix — {model_name}")
        show_confusion_matrix(y_test, y_pred)

    # ---- Tab 3: Single Prediction ---- #
    with tabs[2]:
        st.subheader("Predict Churn for a Single Customer")

        selected_model_name = st.selectbox(
            "Model for prediction",
            list(trained_models.keys()),
            index=list(trained_models.keys()).index("XGBoost")
            if "XGBoost" in trained_models
            else 0,
        )
        model = trained_models[selected_model_name]

        input_df = build_single_input_form(df, categorical_cols, numeric_cols)

        if st.button("Predict Churn"):
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0, 1]

            st.markdown("### Prediction Result")
            if pred == 1:
                st.success(f"Predicted: **Churn** (probability {proba:.2%})")
            else:
                st.info(f"Predicted: **No Churn** (probability {proba:.2%})")

            st.markdown("#### Raw Input")
            st.dataframe(input_df)

    # ---- Tab 4: Feature Importance ---- #
    with tabs[3]:
        st.subheader("Top Features by Model")

        # Let the user pick ANY model
        selected_model = st.selectbox(
            "Select model for feature importance",
            list(trained_models.keys()),
            index=list(trained_models.keys()).index("XGBoost")
            if "XGBoost" in trained_models
            else 0,
        )

        pipe = trained_models[selected_model]
        model = pipe.named_steps["model"]
        feature_names = pipe.named_steps["preprocess"].get_feature_names_out()

        # 1) Tree-based models: use feature_importances_
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

        # 2) Linear models (e.g., Logistic Regression): use |coef_|
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).ravel()

        # 3) Others (e.g., SVM with RBF): use permutation importance
        else:
            st.info(
                "This model has no built-in importance; "
                "using permutation importance (may take a few seconds)."
            )
            result = permutation_importance(
                pipe,
                X_test,
                y_test,
                n_repeats=10,
                random_state=42,
                n_jobs=-1,
            )
            importances = result.importances_mean

        fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)

        top_k = st.slider("Number of top features to show", 5, 30, 10)
        top_fi = fi.head(top_k)

        fig, ax = plt.subplots(figsize=(8, 0.4 * top_k + 2))
        top_fi.sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title(f"Top {top_k} Features — {selected_model}")
        st.pyplot(fig)

        st.dataframe(top_fi.to_frame(name="Importance"))


if __name__ == "__main__":
    main()
