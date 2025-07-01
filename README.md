# Revolutionizing-Liver-Care-Predicting-Liver-Cirrhosis-using-Advanced-Machine-Learning-Techniques
A machine learning project to predict liver cirrhosis using clinical data. Models like Random Forest and XGBoost are trained, evaluated, and deployed via a Flask web app. Users can input patient data and get real-time predictions to support early diagnosis and better treatment planning.

📊 Dataset

Source: Collected healthcare data related to liver conditions.

Format: .xlsx Excel file containing 41 columns — 40 input features and 1 target column.

Target Variable:

1 → Patient has liver cirrhosis

0 → Patient does NOT have liver cirrhosis

Key Features:

Demographics: Age, Gender, Location

Alcohol Usage: Duration, Quantity

Viral Infections: Hepatitis B, Hepatitis C

Health Indicators: Diabetes, Obesity, Blood Pressure

Lab Results: RBC, WBC, Hemoglobin, Bilirubin, SGOT, SGPT, etc.

Full list available in the dataset description.

🛠️ Technologies Used
Frontend: HTML, CSS, Bootstrap (in index.html)

Backend: Flask (Python Web Framework)

ML Model: Random Forest Classifier

Libraries: Pandas, NumPy, scikit-learn, joblib

⚙️ How It Works
The user fills in patient details via a web form.

The Flask backend processes the input and passes it to the trained ML model.

The model predicts whether the patient has liver cirrhosis.

The result is displayed to the user on the interface.

🚀 Getting Started
Prerequisites: Python 3.9 or higher

Steps to Run the Project:

Install dependencies from the requirements file.

Navigate to the training folder and run the training script to train and save the model.

Go to the app folder and start the Flask application.

Open http://127.0.0.1:5000/ in your browser to use the app.

🧠 Model Details
Algorithm Used: RandomForestClassifier

Data Preprocessing:

Categorical features are encoded using OneHotEncoder

Numerical features are scaled using StandardScaler

Model Evaluation: Accuracy score, Confusion Matrix (optional)

✅ Future Enhancements
Add user login and authentication system

Deploy the application on cloud platforms like Heroku or Render

Add EDA dashboards and integrate model explainability tools such as SHAP or LIME

📬 Contact
Sasi kumar Yeripilli St. Ann's College of Engineering and Technology
