A machine learning project to predict liver cirrhosis using clinical data. Models like Random Forest and XGBoost are trained, evaluated, and deployed via a Flask web application. Users can input patient data and get real-time predictions to support early diagnosis and improve treatment planning.

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

Full feature list is available in the dataset description.

🛠️ Technologies Used
Frontend: HTML, CSS, Bootstrap (index.html)

Backend: Flask (Python Web Framework)

Machine Learning Model: Random Forest Classifier

Libraries: Pandas, NumPy, scikit-learn, joblib

⚙️ How It Works
The user fills in patient details through a web form.

The Flask backend processes the input and sends it to the trained machine learning model.

The model predicts whether the patient has liver cirrhosis.

The result is displayed on the user interface.

🚀 Getting Started
Prerequisites

Python 3.9 or higher

Steps to Run the Project

Install dependencies from the requirements.txt file.

Navigate to the training folder and run the training script to train and save the model.

Move to the app folder and start the Flask application.

Open your browser and visit http://127.0.0.1:5000/ to use the application.

🧠 Model Details
Algorithm Used: RandomForestClassifier

Data Preprocessing:

Categorical features → OneHotEncoder

Numerical features → StandardScaler

Model Evaluation:

Accuracy Score

Confusion Matrix (optional)

✅ Future Enhancements
Add user login and authentication

Deploy the application on cloud platforms like Heroku or Render

Integrate EDA dashboards and model explainability tools like SHAP or LIME

📬 Contact
Sasi Kumar Yeripilli
St. Ann's College of Engineering and Technology
