import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the data
df = pd.read_csv('labeled_squat_data.csv')
X = df.drop('label', axis=1)
y = df['label']

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Forest
# 'n_estimators=100' creates 100 different decision trees that vote on the result
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

#  Audit the accuracy 
print("FINAL MODEL PERFORMANCE")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

 #Save the model
joblib.dump(model, 'final_squat_model.joblib')
print("\nSuccess! 'final_squat_model.joblib' is ready for deployment.")