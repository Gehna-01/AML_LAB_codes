# Step 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert to pandas DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)

# Add target column
df['species'] = y

# Display first few rows
print(df.head())

# Count of each class
print("\nClass distribution:\n", df['species'].value_counts())

# Step 3: Data Visualization

# Histogram
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# Pairplot
sns.pairplot(df, hue="species")
plt.show()

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 7: Prediction
y_pred = model.predict(X_test)

# Step 8: Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))