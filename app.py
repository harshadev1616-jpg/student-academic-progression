from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "student_performance_model.pkl"
DATASET_PATH = PROJECT_DIR / "StudentPerformanceFactors.csv"
FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Previous_Scores",
    "Sleep_Hours",
    "Tutoring_Sessions",
    "Physical_Activity",
]
TARGET = "Exam_Score"
DAYS_PER_WEEK = 7

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_stylesheet() -> None:
    """Load the project theme, reporting a visible error if the file is absent."""
    stylesheet_path = PROJECT_DIR / "assets" / "style.css"
    try:
        stylesheet = stylesheet_path.read_text(encoding="utf-8")
    except OSError as error:
        st.error(f"Unable to load the dashboard styles ({error}).")
        return
    st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: str) -> Any:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained model file not found: {path.name}")

    model = joblib.load(path)
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None and list(model_features) != FEATURES:
        raise ValueError(
            "The saved model's feature names or order do not match the required "
            f"features: {FEATURES}."
        )
    return model


@st.cache_data
def load_dataset(dataset_path: str) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path.name}")

    data = pd.read_csv(path)
    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in data]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )
    if data[TARGET].isna().any():
        raise ValueError("Dataset contains missing target values.")
    return data


def evaluate_model(
    model: Any, data: pd.DataFrame
) -> tuple[dict[str, float], pd.Series, np.ndarray]:
    """Evaluate the saved estimator on the notebook's deterministic holdout split."""
    features = data[FEATURES]
    target = data[TARGET]
    _, X_test, _, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    predictions = model.predict(X_test)
    mse = float(mean_squared_error(y_test, predictions))
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_test, predictions)),
    }
    return metrics, y_test, np.asarray(predictions)


def make_actual_vs_predicted_chart(
    actual: pd.Series, predicted: np.ndarray
) -> plt.Figure:
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(
        actual,
        predicted,
        alpha=0.55,
        s=24,
        color="#7A5CFF",
        edgecolors="none",
    )
    score_min = min(float(actual.min()), float(predicted.min()))
    score_max = max(float(actual.max()), float(predicted.max()))
    axis.plot(
        [score_min, score_max],
        [score_min, score_max],
        linestyle="--",
        linewidth=1.6,
        color="#00BFFF",
        label="Perfect prediction",
    )
    axis.set(
        title="Actual vs. predicted exam scores",
        xlabel="Actual exam score",
        ylabel="Predicted exam score",
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.15)
    fig.tight_layout()
    return fig


def make_input_chart(values: dict[str, float]) -> plt.Figure:
    labels = ["Hours studied per day", "Attendance (%)", "Previous score"]
    keys = ["Hours_Studied_Per_Day", "Attendance", "Previous_Scores"]
    chart_values = [values[key] for key in keys]
    fig, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(labels, chart_values, color=["#7A5CFF", "#00BFFF", "#9B8AFF"])
    axis.set_title("Inputs used for this prediction")
    axis.set_ylabel("Input value (original units)")
    axis.grid(axis="y", alpha=0.15)
    axis.bar_label(bars, fmt="%.1f", padding=4, color="#E8E8F0")
    fig.tight_layout()
    return fig


def format_metric(metrics: dict[str, float] | None, key: str, digits: int) -> str:
    if metrics is None:
        return "Unavailable"
    return f"{metrics[key]:.{digits}f}"


def render_sidebar(model: Any | None, dataset: pd.DataFrame | None) -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="brand-mark">SP</div>
                <div>
                    <div class="brand-title">Academic Insights</div>
                    <div class="brand-caption">Student performance system</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("#### Project status")
        st.markdown(
            f'<div class="status-row"><span>Model</span><strong class="{"status-ok" if model is not None else "status-error"}">'
            f'{"Ready" if model is not None else "Unavailable"}</strong></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="status-row"><span>Dataset</span><strong class="{"status-ok" if dataset is not None else "status-error"}">'
            f'{"Ready" if dataset is not None else "Unavailable"}</strong></div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.caption("Robust Regression · Six study indicators")
        st.caption("Machine Learning Major Project")


def main() -> None:
    load_stylesheet()

    model: Any | None = None
    dataset: pd.DataFrame | None = None
    evaluation: tuple[dict[str, float], pd.Series, np.ndarray] | None = None

    try:
        model = load_model(str(MODEL_PATH))
    except FileNotFoundError as error:
        st.error(f"Model unavailable: {error}. Add the trained model to the project folder.")
    except Exception as error:
        st.error(f"Unable to load the saved model: {error}")

    try:
        dataset = load_dataset(str(DATASET_PATH))
    except FileNotFoundError as error:
        st.warning(
            f"Evaluation dataset unavailable: {error}. Predictions can still run, "
            "but evaluation metrics and charts cannot be shown."
        )
    except Exception as error:
        st.warning(f"Evaluation dataset could not be used: {error}")

    if model is not None and dataset is not None:
        try:
            evaluation = evaluate_model(model, dataset)
        except Exception as error:
            st.warning(f"Model evaluation is unavailable: {error}")

    render_sidebar(model, dataset)

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">TOPGRADE INNOVATIONS · MACHINE LEARNING</div>
            <div class="hero-title">Student Performance Prediction</div>
            <div class="hero-subtitle">Machine Learning-Based Academic Performance Analysis</div>
            <span class="powered-badge"><span class="badge-dot"></span> Powered by Machine Learning</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = evaluation[0] if evaluation is not None else None
    model_name = (
        "Huber Regression"
        if model is not None and type(model).__name__ == "Pipeline"
        else type(model).__name__
        if model is not None
        else "Unavailable"
    )
    st.markdown('<div class="section-kicker">CURRENT MODEL EVALUATION</div>', unsafe_allow_html=True)
    metric_columns = st.columns(4)
    overview = [
        ("Dataset size", f"{len(dataset):,}" if dataset is not None else "Unavailable", "records"),
        ("Model", model_name, "saved estimator"),
        ("Mean absolute error", format_metric(metrics, "mae", 2), "points · lower is better"),
        ("R² score", format_metric(metrics, "r2", 4), "test-set result"),
    ]
    for column, (label, value, caption) in zip(metric_columns, overview):
        with column:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label, value)
            st.caption(caption)
            st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Values above are from the current model evaluation on the deterministic test split.")

    st.markdown('<div class="section-heading" id="student-input">Make a prediction</div>', unsafe_allow_html=True)
    st.markdown(
        "<div class=\"section-description\">Enter the student's study habits and academic history.</div>",
        unsafe_allow_html=True,
    )
    form_column, result_column = st.columns([1.05, 0.95], gap="large")
    with form_column:
        with st.form("student_prediction_form", border=True):
            st.markdown("#### Study habits and academic history")
            hours_column, attendance_column = st.columns(2)
            with hours_column:
                hours_per_day = st.number_input(
                    "Hours studied per day",
                    min_value=1.0 / DAYS_PER_WEEK,
                    max_value=44.0 / DAYS_PER_WEEK,
                    value=20.0 / DAYS_PER_WEEK,
                    step=0.25,
                    help=(
                        "The dataset records weekly study hours. This daily value "
                        "is multiplied by 7 before prediction."
                    ),
                )
            with attendance_column:
                attendance = st.number_input(
                    "Attendance percentage",
                    min_value=60.0,
                    max_value=100.0,
                    value=80.0,
                    step=1.0,
                )

            previous_column, sleep_column = st.columns(2)
            with previous_column:
                previous_score = st.number_input(
                    "Previous exam score",
                    min_value=50.0,
                    max_value=100.0,
                    value=75.0,
                    step=1.0,
                )
            with sleep_column:
                sleep_hours = st.number_input(
                    "Sleep hours per night",
                    min_value=4,
                    max_value=10,
                    value=7,
                    step=1,
                )

            tutoring_column, activity_column = st.columns(2)
            with tutoring_column:
                tutoring_sessions = st.number_input(
                    "Tutoring sessions per month",
                    min_value=0,
                    max_value=8,
                    value=1,
                    step=1,
                )
            with activity_column:
                physical_activity = st.number_input(
                    "Physical activity hours per week",
                    min_value=0,
                    max_value=6,
                    value=3,
                    step=1,
                )

            submitted = st.form_submit_button(
                "Predict Student Score",
                type="primary",
                width="stretch",
            )

    with result_column:
        st.markdown('<div class="result-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-kicker">PREDICTION RESULT</div>', unsafe_allow_html=True)
        if submitted:
            hours_per_week = float(hours_per_day * DAYS_PER_WEEK)
            model_input_values = {
                "Hours_Studied": hours_per_week,
                "Attendance": float(attendance),
                "Previous_Scores": float(previous_score),
                "Sleep_Hours": int(sleep_hours),
                "Tutoring_Sessions": int(tutoring_sessions),
                "Physical_Activity": int(physical_activity),
            }
            chart_values = {
                "Hours_Studied_Per_Day": float(hours_per_day),
                "Attendance": float(attendance),
                "Previous_Scores": float(previous_score),
            }
            if not (1 <= hours_per_week <= 44 and 60 <= attendance <= 100 and 50 <= previous_score <= 100):
                st.error("Validation error: keep each input within its permitted range.")
            elif model is None:
                st.error("Prediction unavailable because the trained model could not be loaded.")
            else:
                try:
                    model_input = pd.DataFrame([model_input_values], columns=FEATURES)
                    prediction = float(model.predict(model_input)[0])
                    st.markdown(f'<div class="prediction-value">{prediction:.2f}<span>/ 100</span></div>', unsafe_allow_html=True)
                    st.progress(min(max(prediction / 100, 0.0), 1.0), text="Position on a 100-point scale")
                    st.caption(
                        "This is a model estimate, not a guaranteed exam result. "
                        "The progress indicator is bounded to the 100-point scale."
                    )
                    st.markdown("**Inputs used for this prediction**")
                    st.markdown(
                        f"""
                        <div class="input-summary">
                            <span>Hours studied per day <b>{hours_per_day:g}</b></span>
                            <span>Study hours used by model (per week) <b>{hours_per_week:.2f}</b></span>
                            <span>Attendance <b>{attendance:g}%</b></span>
                            <span>Previous score <b>{previous_score:g}</b></span>
                            <span>Sleep per night <b>{sleep_hours:g} hours</b></span>
                            <span>Tutoring sessions <b>{tutoring_sessions:g}</b></span>
                            <span>Physical activity <b>{physical_activity:g} hours/week</b></span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    input_fig = make_input_chart(chart_values)
                    st.pyplot(input_fig, width="stretch")
                    plt.close(input_fig)
                except Exception as error:
                    st.error(f"Prediction failed: {error}")
        else:
            st.markdown('<div class="empty-result">', unsafe_allow_html=True)
            st.markdown("Your estimate will appear here")
            st.caption("Complete the form and submit to calculate a prediction.")
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-heading" id="analytics">Visual analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">Test-set results are derived from the supplied dataset and saved model.</div>',
        unsafe_allow_html=True,
    )
    if evaluation is None:
        st.info(
            "Actual test-set predictions are unavailable. Add the original dataset and "
            "compatible saved model to display analytics; no sample or synthetic chart data is used."
        )
    else:
        actual, predicted = evaluation[1], evaluation[2]
        chart_column, note_column = st.columns([1.5, 0.8], gap="large")
        with chart_column:
            with st.container(border=True):
                fig = make_actual_vs_predicted_chart(actual, predicted)
                st.pyplot(fig, width="stretch")
                plt.close(fig)
        with note_column:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="section-kicker">TEST SET</div>
                    <div class="insight-number">{len(actual):,}</div>
                    <div class="section-description">held-out records from the 80/20 split</div>
                    <hr>
                    <p>Each point compares an actual exam score with a prediction from the saved model. The dashed line marks a perfect prediction.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-heading" id="evaluation">Model evaluation</div>', unsafe_allow_html=True)
    if metrics is None:
        st.info("Evaluation metrics cannot be calculated without the original test data and a compatible saved model.")
    else:
        evaluation_columns = st.columns(4)
        evaluation_cards = [
            ("Mean Absolute Error", metrics["mae"], "Average absolute difference between the prediction and actual score."),
            ("Mean Squared Error", metrics["mse"], "Average squared difference; larger errors receive more weight."),
            ("Root Mean Squared Error", metrics["rmse"], "Typical error size expressed in exam-score points."),
            ("R² Score", metrics["r2"], "Share of test-set score variation explained by the fitted model; not a guarantee of accuracy."),
        ]
        for column, (label, value, explanation) in zip(evaluation_columns, evaluation_cards):
            with column:
                with st.container(border=True):
                    st.metric(label, f"{value:.4f}")
                    st.caption(explanation)
        st.caption("All values are calculated from the saved model's predictions on the same 20% holdout split (random_state=42).")

    st.markdown('<div class="section-heading" id="about">About this project</div>', unsafe_allow_html=True)
    with st.container(border=True):
        about_left, about_right = st.columns(2, gap="large")
        with about_left:
            st.markdown("#### Problem & objective")
            st.write(
                "This project estimates a student's final exam score from selected "
                "academic indicators. Its objective is to demonstrate how supervised "
                "machine learning can model a continuous academic outcome."
            )
            st.markdown("#### Inputs and target")
            st.write(
                "**Inputs:** Study hours, attendance, previous exam score, "
                "sleep, tutoring, and physical activity.  \n"
                "**Target:** Exam_Score, a numerical score."
            )
        with about_right:
            st.markdown("#### Why regression?")
            st.write(
                "Regression is suitable because the target is numeric and the model "
                "estimates a score on a continuous scale. The displayed estimate "
                "should be interpreted with the evaluation metrics, not as a guarantee."
            )
            st.markdown("#### Technologies")
            st.write("Python · Streamlit · Pandas · NumPy · Scikit-learn · Matplotlib · Joblib")
        st.markdown("---")
        st.caption("Machine Learning Major Project · Topgrade Innovations")


if __name__ == "__main__":
    main()
