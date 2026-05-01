# Step 1: Import Libraries
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Step 2: Load Dataset
data = load_breast_cancer()

X = data.data      # Features
y = data.target    # Labels

print("Dataset Shape:", X.shape)
# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# Step 4: Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Step 5: Train Model (SVM)
model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale'
)

model.fit(X_train, y_train)
# Step 6: Prediction
y_pred = model.predict(X_test)

# Step 7: Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
