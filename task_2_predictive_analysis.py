"""
Task‑2 – Predictive Analysis (Titanic)
Creates text, CSV and PNG outputs in the chosen folder.
"""

import os
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ─────────────────────────────────────────────────────────────
# 1.  OUTPUT DIRECTORY  ( your exact path )
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR = (r"C:\Users\pedda\OneDrive\Desktop\ESAIP CLG SUBJECTS\Internship\INTERNSHIP\Task-2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 2.  LOAD DATASET
# ─────────────────────────────────────────────────────────────
print("Loading Titanic dataset …")
df = pd.read_csv(
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
print("Rows, columns:", df.shape)

# ─────────────────────────────────────────────────────────────
# 3.  CLEAN & ENCODE
# ─────────────────────────────────────────────────────────────
df = df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"])
df.loc[:, "Age"] = df["Age"].fillna(df["Age"].median())
df.loc[:, "Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

enc = LabelEncoder()
df["Sex"] = enc.fit_transform(df["Sex"])
df["Embarked"] = enc.fit_transform(df["Embarked"])

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X, y = df[features], df["Survived"]

# ─────────────────────────────────────────────────────────────
# 4.  TRAIN / TEST
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# ─────────────────────────────────────────────────────────────
# 5.  EVALUATION
# ─────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)
cr  = classification_report(y_test, y_pred)

print("\nAccuracy:", acc)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", cr)

# Save metrics to TXT
txt_path = os.path.join(OUTPUT_DIR, "evaluation_results.txt")
with open(txt_path, "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))
    f.write("\n\nClassification Report:\n")
    f.write(cr)

# Save metrics to CSV
csv_path = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
rows = [["Metric", "Value"],
        ["Accuracy", f"{acc:.4f}"],
        ["", ""],
        ["Confusion Matrix", ""],
        ["True Negative", cm[0, 0]],
        ["False Positive", cm[0, 1]],
        ["False Negative", cm[1, 0]],
        ["True Positive", cm[1, 1]],
        ["", ""],
        ["Classification Report", ""]]

for label, metrics in classification_report(
        y_test, y_pred, output_dict=True).items():
    if isinstance(metrics, dict):
        for m_name, val in metrics.items():
            rows.append([f"{label} - {m_name}", f"{val:.4f}"])
    else:
        rows.append([label, f"{metrics:.4f}"])

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

# ─────────────────────────────────────────────────────────────
# 6.  VISUALISATIONS
# ─────────────────────────────────────────────────────────────
## 6‑A Correlation heatmap
plt.figure(figsize=(9, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"),
            dpi=300, bbox_inches="tight")
plt.show()

## 6‑B Confusion‑matrix heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Not Survived", "Survived"],
            yticklabels=["Not Survived", "Survived"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted"), plt.ylabel("Actual")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
            dpi=300, bbox_inches="tight")
plt.show()

## 6‑C Feature importance bar chart
importances = model.feature_importances_
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=features)
plt.title("Random‑Forest Feature Importances")
plt.xlabel("Importance"), plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importances.png"),
            dpi=300, bbox_inches="tight")
plt.show()

print("\n All outputs saved to:", OUTPUT_DIR)
