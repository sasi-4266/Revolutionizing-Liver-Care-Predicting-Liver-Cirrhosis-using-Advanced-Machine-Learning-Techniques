import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Starting train.py")

# Load dataset
df = pd.read_excel('../Data/HealthCareData.xlsx')
print("Columns in the dataset:")
print(df.columns.tolist())  # Debug: Print raw columns

# Target column (match the exact name from the dataset)
target_col = 'Predicted Value(Out Come-Patient suffering from liver  cirrosis or not)'  # Use the exact name with spaces
if target_col not in df.columns:
    print(f"Target column '{target_col}' not found. Checking normalized names...")
    normalized_cols = [col.strip() for col in df.columns]
    if target_col.strip() in normalized_cols:
        target_col = next(col for col in df.columns if col.strip() == target_col.strip())
        print(f"Adjusted target column to: {target_col}")
    else:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame. Available columns: {df.columns.tolist()}")

# Drop rows where target is NaN
df = df.dropna(subset=[target_col])
print(f"\nRows after dropping NaN in {target_col}: {len(df)}")  # Debug

# Define X and y
X = df.drop(target_col, axis=1)
y = df[target_col]

# Fix column names with extra spaces
X.columns = [col.strip() for col in X.columns]
print(f"\nFeatures after dropping target (should be 39): {X.columns.tolist()}")  # Debug

# Define categorical columns (use exact names with spaces from dataset)
categorical_cols = [
    'Gender',
    'Place(location where the patient lives)',
    'Type of alcohol consumed',
    'Hepatitis B infection',
    'Hepatitis C infection',
    'Diabetes Result',
    'Blood pressure (mmhg)',
    'Obesity',
    'Family history of cirrhosis/ hereditary',
    'USG Abdomen (diffuse liver or  not)'  # Match exact name with trailing space
]

# Ensure categorical_cols matches X.columns
categorical_cols = [col for col in categorical_cols if col in X.columns]
print(f"\nCategorical columns (matched): {categorical_cols}")  # Debug

# Get numerical columns
numerical_cols = [col for col in X.columns if col not in categorical_cols]
print(f"\nNumerical columns: {numerical_cols}")  # Debug

# Handle numeric missing values by filling with mean
for col in numerical_cols:
    X[col] = pd.to_numeric(X[col], errors='coerce')  # convert to numbers
    X[col] = X[col].fillna(X[col].mean())

# Handle missing categorical with 'missing'
for col in categorical_cols:
    X[col] = X[col].astype(str).fillna('missing')

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
pipeline.fit(X_train, y_train)

# Save model and feature columns
joblib.dump(pipeline, 'model.pkl')
joblib.dump(X.columns.tolist(), 'train_columns.joblib')

print("✅ Model trained and saved successfully!")