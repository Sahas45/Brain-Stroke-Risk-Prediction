
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("stroke_data.csv")

    # Handle missing BMI values using median
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())

    return df


data = load_data()


# Encode categorical columns
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
    data[col] = le.fit_transform(data[col])
    encoders[col] = le


# Features and target
X = data.drop(['id', 'stroke'], axis=1)
y = data['stroke']


# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

model.fit(X_train, y_train)


# Model predictions
y_pred = model.predict(X_test)
probs_test = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

fig_cm, ax_cm = plt.subplots()
disp.plot(ax=ax_cm)
plt.title("Confusion Matrix")


# Sidebar navigation
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["🧠 Stroke Risk Prediction", "📊 Data Insights"]
)


# =================== PAGE 1: Prediction ===================

if page == "🧠 Stroke Risk Prediction":

    st.title("🧠 Stroke Risk Predictor")

    st.write(
        "Enter patient details below to predict stroke risk."
    )

    gender = st.selectbox(
        "Gender",
        encoders['gender'].classes_
    )

    age = st.slider("Age", 0, 100, 30)

    hypertension = st.selectbox(
        "Hypertension (0 = No, 1 = Yes)",
        [0, 1]
    )

    heart_disease = st.selectbox(
        "Heart Disease (0 = No, 1 = Yes)",
        [0, 1]
    )

    ever_married = st.selectbox(
        "Ever Married",
        encoders['ever_married'].classes_
    )

    work_type = st.selectbox(
        "Work Type",
        encoders['work_type'].classes_
    )

    residence_type = st.selectbox(
        "Residence Type",
        encoders['Residence_type'].classes_
    )

    avg_glucose = st.slider(
        "Average Glucose Level",
        50.0, 300.0, 100.0
    )

    bmi = st.slider(
        "BMI",
        10.0, 50.0, 22.0
    )

    smoking_status = st.selectbox(
        "Smoking Status",
        encoders['smoking_status'].classes_
    )


    # Prepare patient input
    input_dict = {
        'gender': [
            encoders['gender'].transform([gender])[0]
        ],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [
            encoders['ever_married'].transform(
                [ever_married]
            )[0]
        ],
        'work_type': [
            encoders['work_type'].transform(
                [work_type]
            )[0]
        ],
        'Residence_type': [
            encoders['Residence_type'].transform(
                [residence_type]
            )[0]
        ],
        'avg_glucose_level': [avg_glucose],
        'bmi': [bmi],
        'smoking_status': [
            encoders['smoking_status'].transform(
                [smoking_status]
            )[0]
        ]
    }

    input_df = pd.DataFrame(input_dict)


    # Prediction
    if st.button("Predict Stroke Risk"):

        pred = model.predict(input_df)[0]

        prob = model.predict_proba(input_df)[0]

        stroke_probability = prob[
            list(model.classes_).index(1)
        ]

        st.subheader("🩺 Prediction Result:")

        if pred == 1:

            st.error(
                f"⚠️ Model predicts stroke class. "
                f"Estimated probability: "
                f"{stroke_probability:.2%}"
            )

        else:

            st.success(
                f"✅ Model predicts no-stroke class. "
                f"Estimated stroke probability: "
                f"{stroke_probability:.2%}"
            )


        # Probability bar chart
        st.subheader("📊 Prediction Probability")

        fig_prob, ax_prob = plt.subplots()

        ax_prob.bar(
            ['No Stroke', 'Stroke'],
            prob,
            color=['green', 'red']
        )

        ax_prob.set_ylim(0, 1)
        ax_prob.set_ylabel("Probability")
        ax_prob.set_title("Model Probability Estimates")

        st.pyplot(fig_prob)


# =================== PAGE 2: Data Insights ===================

elif page == "📊 Data Insights":

    st.title("📊 Data & Model Insights")


    # Dataset overview
    st.subheader("Dataset Overview")

    st.dataframe(data.head())


    # Stroke class distribution
    st.subheader("Stroke Class Distribution")

    stroke_counts = data['stroke'].value_counts().sort_index()

    fig1, ax1 = plt.subplots()

    ax1.pie(
        stroke_counts,
        labels=["No Stroke", "Stroke"],
        autopct='%1.1f%%',
        colors=['green', 'red']
    )

    ax1.axis('equal')

    st.pyplot(fig1)


    # Age distribution
    st.subheader("Age Distribution by Stroke Outcome")

    fig2, ax2 = plt.subplots()

    sns.histplot(
        data=data,
        x="age",
        hue="stroke",
        multiple="stack",
        bins=30,
        palette="husl",
        ax=ax2
    )

    st.pyplot(fig2)


    # Glucose vs BMI
    st.subheader("Glucose vs BMI (Colored by Stroke)")

    fig3, ax3 = plt.subplots()

    sns.scatterplot(
        data=data,
        x="avg_glucose_level",
        y="bmi",
        hue="stroke",
        palette="coolwarm",
        ax=ax3
    )

    st.pyplot(fig3)


    # Confusion matrix
    st.subheader("Confusion Matrix")

    st.pyplot(fig_cm)


    # Model accuracy
    st.subheader("Model Accuracy")

    st.metric(
        "Accuracy on Test Set",
        f"{accuracy:.2%}"
    )


# Medical disclaimer
st.sidebar.warning(
    "Educational project only. "
    "Predictions are not medical diagnoses."
)
