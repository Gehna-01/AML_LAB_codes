import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv(r'C:\Users\Acer\OneDrive\Desktop\applied ml lab\lab7-Credit Risk Assessment\Credit_Data.csv')

# Fix column names (IMPORTANT)
data.columns = data.columns.str.strip().str.lower()

# Show columns to verify
print("Columns in dataset:", data.columns)

# Target column (adjust if needed)
target_col = 'default'   # after converting to lowercase

# Check distribution
print(data[target_col].value_counts())

# Plot distribution
sns.countplot(x=target_col, data=data)
plt.title("Distribution of Default")
plt.show()

# Histograms
data.hist(figsize=(12,10), bins=20)
plt.suptitle("Feature Distributions")
plt.show()

# Pairplot
sns.pairplot(data, hue=target_col)
plt.show()

# Split features and target
X = data.drop(target_col, axis=1)
y = data[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC AUC Score (only for binary classification)
try:
    y_prob = model.predict_proba(X_test)[:, 1]
    print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))
except:
    print("\nROC-AUC not applicable")