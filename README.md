# Student Performance Prediction

A Streamlit dashboard for estimating a student's final exam score from six
academic and study-habit indicators. It loads a saved
Huber Regression pipeline; the application does not retrain it.

## Project files

```text
student_performance_prediction/
├── app.py
├── student_performance_model.pkl
├── StudentPerformanceFactors.csv
├── notebook_prog_pred.ipynb
├── requirements.txt
├── README.md
└── assets/
    └── style.css
```

The CSV is used to reproduce the notebook's 80/20 test split and to calculate
real test-set metrics and chart points. If it is absent, predictions still work
with the saved model, but evaluation metrics and actual-vs-predicted analytics
are unavailable. No synthetic prediction data is generated.

## Setup

From this project directory, create and activate a virtual environment:

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The saved `student_performance_model.pkl` and `StudentPerformanceFactors.csv`
are included in this project folder. If you recreate the project elsewhere,
place the compatible trained model and original CSV beside `app.py`. The model
expects these features in this exact order:

```python
[
    "Hours_Studied",
    "Attendance",
    "Previous_Scores",
    "Sleep_Hours",
    "Tutoring_Sessions",
    "Physical_Activity",
]
```

The prediction form accepts hours studied per day and converts that input to
weekly study hours (daily hours multiplied by 7) for the model's
`Hours_Studied` feature. The permitted daily range corresponds to the dataset's
1–44 weekly-hour range. It also asks for attendance, previous exam score, sleep,
tutoring, and physical activity. The prediction result displays all values used.
The model uses numeric study and academic indicators only; numeric inputs are
scaled and missing training values are imputed within the saved pipeline.

The model file is a Python pickle artifact. Use a model file from a trusted
source and keep the scikit-learn version compatible with the one used to train
it (the included requirements pin the version verified with this model).

Run the application:

```bash
streamlit run app.py
```

## Evaluation and analytics

When both artifacts are present, the app selects the same 20% holdout data used
by the training notebook (`test_size=0.2`, `random_state=42`), then calls the
saved model to generate predictions. The model is trained only on the
remaining 80%, so the displayed holdout metrics are not measured on its
training examples. With the six numeric form inputs, the saved model achieves
approximately 1.26 MAE, 2.25 RMSE, and 0.640 R² on the included dataset and
split. A five-fold cross-validation comparison on the training portion found
Huber Regression reduced MAE from 1.327 to 1.317 points compared with Ridge.
RMSE did not improve, so this is a small improvement in average absolute error,
not a dramatic accuracy increase. Results are estimates, not guarantees. The
input bar chart shows daily study hours, attendance, and previous score from
the form.

To retrain the model in the original notebook, run it from this project
directory. Preserve the feature names/order and test split, then replace the
saved model in this folder. Keep the original CSV here to make the test-set
evaluation reproducible.

## Troubleshooting

- **Model unavailable:** confirm `student_performance_model.pkl` is in the same
  folder as `app.py`, is readable, and was created from a trusted source with a
  compatible scikit-learn version.
- **Evaluation dataset unavailable:** confirm
  `StudentPerformanceFactors.csv` is beside `app.py`, and that it contains
  the model input columns and `Exam_Score`.
- **Evaluation or prediction error:** verify that the saved estimator is the
  expected preprocessing pipeline and that its feature names match the exact
  names and order above.
- **Dependency issue:** activate the project virtual environment and reinstall
  with `python -m pip install -r requirements.txt`.

## Project credit

Machine Learning Major Project for **Topgrade Innovations**.
