import numpy as np
from flask import Flask, render_template, request
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and column names from the training directory
model = joblib.load('../training/model.pkl')
train_columns = joblib.load('../training/train_columns.joblib')  # 39 features
print("Training columns:", train_columns)  # Debug

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect all input values from the form (39 features, excluding 'S.NO')
        inputs = [
            request.form.get('Age', ''),
            request.form.get('Gender', ''),
            request.form.get('Place(location where the patient lives)', ''),
            request.form.get('Duration of alcohol consumption(years)', ''),
            request.form.get('Quantity of alcohol consumption (quarters/day)', ''),
            request.form.get('Type of alcohol consumed', ''),
            request.form.get('Hepatitis B infection', ''),
            request.form.get('Hepatitis C infection', ''),
            request.form.get('Diabetes Result', ''),
            request.form.get('Blood pressure (mmhg)', ''),
            request.form.get('Obesity', ''),
            request.form.get('Family history of cirrhosis/ hereditary', ''),
            request.form.get('TCH', ''),
            request.form.get('TG', ''),
            request.form.get('LDL', ''),
            request.form.get('HDL', ''),
            request.form.get('Hemoglobin (g/dl)', ''),
            request.form.get('PCV (%)', ''),
            request.form.get('RBC (million cells/microliter)', ''),
            request.form.get('MCV (femtoliters/cell)', ''),
            request.form.get('MCH (picograms/cell)', ''),
            request.form.get('Total Count', ''),
            request.form.get('MCHC (grams/deciliter)', ''),
            request.form.get('Polymorphs (%)', ''),
            request.form.get('Lymphocytes (%)', ''),
            request.form.get('Monocytes (%)', ''),
            request.form.get('Eosinophils (%)', ''),
            request.form.get('Basophils (%)', ''),
            request.form.get('Platelet Count (lakhs/mm)', ''),
            request.form.get('Total Bilirubin (mg/dl)', ''),
            request.form.get('Direct (mg/dl)', ''),
            request.form.get('Indirect (mg/dl)', ''),
            request.form.get('Total Protein (g/dl)', ''),
            request.form.get('Albumin (g/dl)', ''),
            request.form.get('Globulin (g/dl)', ''),
            request.form.get('A/G Ratio', ''),
            request.form.get('AL.Phosphatase (U/L)', ''),
            request.form.get('SGOT/AST (U/L)', ''),
            request.form.get('SGPT/ALT (U/L)', ''),
            request.form.get('USG Abdomen (diffuse liver or  not)', '')  # Match exact name
        ]

        # Define column names (exactly same order as in training data, excluding 'S.NO')
        columns = [
            'Age', 'Gender', 'Place(location where the patient lives)',
            'Duration of alcohol consumption(years)', 'Quantity of alcohol consumption (quarters/day)',
            'Type of alcohol consumed', 'Hepatitis B infection', 'Hepatitis C infection',
            'Diabetes Result', 'Blood pressure (mmhg)', 'Obesity',
            'Family history of cirrhosis/ hereditary', 'TCH', 'TG', 'LDL', 'HDL',
            'Hemoglobin (g/dl)', 'PCV (%)', 'RBC (million cells/microliter)',
            'MCV (femtoliters/cell)', 'MCH (picograms/cell)', 'Total Count',
            'MCHC (grams/deciliter)', 'Polymorphs (%)', 'Lymphocytes (%)',
            'Monocytes (%)', 'Eosinophils (%)', 'Basophils (%)',
            'Platelet Count (lakhs/mm)', 'Total Bilirubin (mg/dl)', 'Direct (mg/dl)',
            'Indirect (mg/dl)', 'Total Protein (g/dl)', 'Albumin (g/dl)',
            'Globulin (g/dl)', 'A/G Ratio', 'AL.Phosphatase (U/L)', 'SGOT/AST (U/L)',
            'SGPT/ALT (U/L)', 'USG Abdomen (diffuse liver or  not)'  # Match exact name
        ]

        # Create a DataFrame with the form inputs
        input_df = pd.DataFrame([inputs], columns=columns)

        # Fill missing values (match train.py)
        categorical_cols = [
            'Gender', 'Place(location where the patient lives)', 'Type of alcohol consumed',
            'Hepatitis B infection', 'Hepatitis C infection', 'Diabetes Result',
            'Blood pressure (mmhg)', 'Obesity', 'Family history of cirrhosis/ hereditary',
            'USG Abdomen (diffuse liver or  not)'  # Match exact name
        ]
        for col in input_df.columns:
            if col in categorical_cols:
                input_df[col] = input_df[col].astype(str).fillna('missing')
            else:
                # Convert to numeric, coercing errors to NaN
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                mean_val = input_df[col].mean()  # Calculate mean after conversion
                input_df[col] = input_df[col].fillna(mean_val)

        # Align columns with training data
        input_df = input_df.reindex(columns=train_columns, fill_value=0)

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Interpret the result (assuming "YES" = 1, "NO" = 0 implicitly handled by RandomForest)
        result = "✅ Patient has Liver Cirrhosis" if prediction == 1 else "🟢 Patient does NOT have Liver Cirrhosis"

        # Return the result page
        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"❌ Error occurred: {str(e)}", 400

# Start the Flask server
if __name__ == '__main__':
    app.run(debug=True)