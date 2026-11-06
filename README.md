# 🧠 Brain Stroke Risk Prediction

A Machine Learning project that estimates the likelihood of stroke based on patient health and lifestyle information using Logistic Regression.

## Try Live Demo: https://brain-stroke-risk-prediction-dmgpcqcmta3jgnk3u5ermb.streamlit.app/
## 📌 Project Overview

Stroke is a serious medical condition that occurs when blood supply to part of the brain is interrupted or reduced.

This project uses machine learning to explore patterns in patient health data and predict stroke outcomes based on selected health indicators.

The application provides an interactive Streamlit interface where users can enter patient details, view a model prediction, and explore dataset insights.

## ✨ Features

- Stroke risk prediction using Logistic Regression.
- Interactive patient information form.
- Prediction probability visualization.
- Dataset overview and class distribution.
- Age distribution analysis.
- Glucose level vs BMI visualization.
- Confusion matrix and model accuracy.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data processing |
| Scikit-learn | Machine Learning |
| Logistic Regression | Classification algorithm |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Web application |

## 📂 Project Structure

```text
Stroke/
│
├── app.py
├── lr.py
├── stroke_data.csv
├── requirements.txt
└── README.md
```

## ⚙️ Machine Learning Workflow

1. Load the stroke dataset.
2. Handle missing values.
3. Encode categorical variables.
4. Separate features and target variable.
5. Split data into training and testing sets (80:20).
6. Train a Logistic Regression model.
7. Evaluate using accuracy, classification report and confusion matrix.
8. Generate predictions through the Streamlit application.

## 📊 Input Features

- Gender
- Age
- Hypertension
- Heart Disease
- Ever Married
- Work Type
- Residence Type
- Average Glucose Level
- BMI
- Smoking Status

## 🚀 Run Locally

**1. Clone the repository**

```bash
git clone https://github.com/Sahas45/Brain-Stroke-Risk-Prediction.git
cd Brain-Stroke-Risk-Prediction
```

**2. Create a virtual environment**

```bash
python -m venv venv
```

**3. Activate the environment (Windows)**

```bash
venv\Scripts\activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Run the Streamlit application**

```bash
streamlit run app.py
```

## ⚠️ Disclaimer

This project is developed for educational and machine learning demonstration purposes only. Its predictions are not medical diagnoses and must not be used to make healthcare decisions. Always consult a qualified healthcare professional for medical advice.

## 👨‍💻 Author

**Sahas Bhosale**

GitHub: https://github.com/Sahas45

---

⭐ If you find this project useful, consider giving the repository a star!
