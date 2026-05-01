import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
# load dataset
data = pd.read_csv(r'C:\Users\Acer\OneDrive\Desktop\applied ml lab\lab7-Credit Risk Assessment\creditcard.csv')
# Display first 5 rows
print("Dataset Preview:")
print(df.head())
# take sample (dataset is big)
df = df.sample(5000, random_state=42)
# Separate features (remove target column)
X = df.drop('Class', axis=1)
# model
# Create Isolation Forest model
# contamination = expected anomaly percentage
model = IsolationForest(contamination=0.01)
model.fit(X)
# Train model
df['anomaly'] = model.predict(X)
# Predict anomalies
# -1 = anomaly , 1 = normal
df['Anomaly'] = model.predict(X)
# Count anomalies
print("\nAnomaly Count:")
print(df['Anomaly'].value_counts())
# Plot anomalies using 2 features
plt.figure(figsize=(8,6))
plt.scatter(df['V1'], df['V2'], c=df['Anomaly'])
plt.title("Anomaly Detection using Isolation Forest")
plt.xlabel("Feature V1")
plt.ylabel("Feature V2")
plt.show()
