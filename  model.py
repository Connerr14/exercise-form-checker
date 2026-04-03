import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Loading the data from the CSV
df = pd.read_csv('labeled_squat_data.csv')

# Separating the data, creating a table with everything but the answer
X = df.drop('label', axis=1)

# Creating a list with the "answers"
y = df['label']

# Splitting the data, using a 80/20 split. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Random Forest Classifier
# 'n_estimators=100' creates 100 different decision trees that vote on the result
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

#  Audit the accuracy 
print("Final Model Performance")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

 #Saving the model to be used in the program
joblib.dump(model, 'final_squat_model.joblib')
print("\nSuccess!")