# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# load dataset
df = pd.read_csv(r'C:\Users\Acer\OneDrive\Desktop\applied ml lab\lab9-normal and abnormal physiological signals\mitbih_train.csv.zip')
# Show dataset info
print("Dataset Shape:", df.shape)
print(df.head())
# Separate features and labels
X = df.iloc[:, :-1]   # all columns except last
y = df.iloc[:, -1]    # last column (class label)

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Create Random Forest model
model = RandomForestClassifier(n_estimators=100)

# Train model
model.fit(X_train, y_train)
# Make predictions
predictions = model.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("\nModel Accuracy:", accuracy)

# Classification report
from sklearn.metrics import classification_report

print("\nClassification Report:")
report = classification_report(y_test, predictions)
print(report)
