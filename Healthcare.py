import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------------------
# Load healthcare dataset (replace 'healthcare_data.csv' with your dataset)
data = pd.read_csv('healthcare_data.csv')

# ---------------------------------------------------------------------------------------
# Data preprocessing and feature selection
# Add your data preprocessing steps here
# Prepare the data
X = data.drop('medical_condition', axis=1)  # Features
y = data['medical_condition']  # Target

# ---------------------------------------------------------------------------------------
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------------------------------------
# Train a Random Forest classifier with hyperparameter tuning
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------------------
# Model evaluation with cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print(f'Cross-validated accuracy scores: {cv_scores}')

# ---------------------------------------------------------------------------------------
# Make predictions on the test set
y_pred = model.predict(X_test)

# ---------------------------------------------------------------------------------------
# Evaluate the model
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
