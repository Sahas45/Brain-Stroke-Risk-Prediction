
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# =====================================
# 1. Load Dataset
# =====================================

df = pd.read_csv("stroke_data.csv")

print("Original Dataset Shape:", df.shape)


# =====================================
# 2. Handle Missing Values
# =====================================

df['bmi'] = df['bmi'].fillna(df['bmi'].median())

print("\nMissing Values After Preprocessing:")
print(df.isnull().sum())


# =====================================
# 3. Encode Categorical Columns
# =====================================

cat_cols = [
    'gender',
    'ever_married',
    'work_type',
    'Residence_type',
    'smoking_status'
]

encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le


# =====================================
# 4. Features and Target
# =====================================

X = df.drop(['id', 'stroke'], axis=1)
y = df['stroke']


# =====================================
# 5. Train-Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# =====================================
# 6. Train Logistic Regression Model
# =====================================

model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

model.fit(X_train, y_train)


# =====================================
# 7. Model Predictions
# =====================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# =====================================
# 8. Model Evaluation
# =====================================

accuracy = accuracy_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_prob)

print("\n========== MODEL PERFORMANCE ==========")

print("Accuracy:", round(accuracy, 4))

print("ROC-AUC Score:", round(roc_auc, 4))


# =====================================
# 9. Classification Report
# =====================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Stroke", "Stroke"],
        zero_division=0
    )
)


# =====================================
# 10. Confusion Matrix
# =====================================

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(y_test, y_pred)

print(cm)


# =====================================
# 11. Custom Patient Prediction
# =====================================

custom_input = pd.DataFrame([{
    'gender': encoders['gender'].transform(['Male'])[0],
    'age': 70,
    'hypertension': 1,
    'heart_disease': 1,
    'ever_married': encoders['ever_married'].transform(['Yes'])[0],
    'work_type': encoders['work_type'].transform(['Self-employed'])[0],
    'Residence_type': encoders['Residence_type'].transform(['Urban'])[0],
    'avg_glucose_level': 210.5,
    'bmi': 38.0,
    'smoking_status': encoders['smoking_status'].transform(['smokes'])[0]
}])

custom_pred = model.predict(custom_input)[0]

custom_prob = model.predict_proba(custom_input)[0]

stroke_probability = custom_prob[
    list(model.classes_).index(1)
]

label = "Stroke Class Predicted" if custom_pred == 1 else "No Stroke Class Predicted"

print("\n========== CUSTOM PATIENT PREDICTION ==========")

print("Prediction:", label)

print("Estimated Stroke Probability:", round(stroke_probability * 100, 2), "%")
