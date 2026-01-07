"""
Title   : Intrusion Detection using XGBoost on UNSW-NB15 (AI)
Purpose : Binary classification (Normal vs Attack)
Author  : <Your Name>
Dataset : UNSW-NB15
"""

# =============================
# 1. Import Required Libraries
# =============================
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from imblearn.over_sampling import SMOTE
import xgboost as xgb

# =============================
# 2. Create Output Directory
# =============================
os.makedirs("results", exist_ok=True)
print("\n[INFO] Results directory checked/created.")

# =============================
# 3. Load UNSW-NB15 Dataset
# =============================
print("\n[INFO] Loading datasets...")
train_df = pd.read_csv("UNSW_NB15_training-set.csv")
test_df  = pd.read_csv("UNSW_NB15_testing-set.csv")

print(f"[INFO] Training set shape : {train_df.shape}")
print(f"[INFO] Testing set shape  : {test_df.shape}")

# Merge training and testing data
df = pd.concat([train_df, test_df], ignore_index=True)
print(f"[INFO] Combined dataset shape: {df.shape}")

# =============================
# 4. Separate Features and Target
# =============================
print("\n[INFO] Separating features and target variable...")
y = df["label"]          # 0 = Normal, 1 = Attack
X = df.drop(columns=["label"])

print(f"[INFO] Feature matrix shape: {X.shape}")
print(f"[INFO] Target distribution:\n{y.value_counts()}")

# =============================
# 5. Encode Categorical Features
# =============================
print("\n[INFO] Encoding categorical features...")
cat_cols = X.select_dtypes(include=['object']).columns
print(f"[INFO] Categorical columns detected: {len(cat_cols)}")

X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
print(f"[INFO] Shape after encoding: {X.shape}")

# =============================
# 6. Keep Numeric Features Only
# =============================
print("\n[INFO] Selecting numeric features only...")
X = X.select_dtypes(include=[np.number])
print(f"[INFO] Numeric feature matrix shape: {X.shape}")

# =============================
# 7. Feature Scaling
# =============================
print("\n[INFO] Applying StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("[INFO] Feature scaling completed.")

# =============================
# 8. Train-Test Split
# =============================
print("\n[INFO] Performing train-test split (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"[INFO] Training samples: {X_train.shape[0]}")
print(f"[INFO] Testing samples : {X_test.shape[0]}")

# =============================
# 9. Handle Class Imbalance (SMOTE)
# =============================
print("\n[INFO] Applying SMOTE...")
print("[INFO] Class distribution before SMOTE:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("[INFO] Class distribution after SMOTE:")
print(pd.Series(y_train_res).value_counts())

# =============================
# 10. Initialize XGBoost Model
# =============================
print("\n[INFO] Initializing XGBoost classifier...")
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=1,
    reg_alpha=0.01,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)
print("[INFO] Model initialized.")

# =============================
# 11. Model Training
# =============================
print("\n[INFO] Training XGBoost model...")
model.fit(X_train_res, y_train_res)
print("[INFO] Model training completed.")

# =============================
# 12. Prediction
# =============================
print("\n[INFO] Generating predictions...")
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# =============================
# 13. Evaluation
# =============================
print("\n[INFO] Evaluating model...")

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
cm        = confusion_matrix(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("======================================")

# =============================
# 14. Save Results to Disk
# =============================
print("\n[INFO] Saving evaluation results...")

with open("results/metrics.txt", "w") as f:
    f.write("=== Intrusion Detection Results (XGBoost) ===\n\n")
    f.write(f"Accuracy  : {accuracy * 100:.2f}%\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1-Score  : {f1:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred))

cm_df = pd.DataFrame(
    cm,
    columns=["Pred_Normal", "Pred_Attack"],
    index=["Actual_Normal", "Actual_Attack"]
)
cm_df.to_csv("results/confusion_matrix.csv")

print("[INFO] Numerical results saved.")

# =============================
# 15. Graphical Analysis
# =============================
print("\n[INFO] Generating graphical analysis...")

# ---- Confusion Matrix Heatmap ----
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Attack"],
            yticklabels=["Normal", "Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

# ---- ROC Curve ----
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", color="darkorange")
plt.plot([0,1], [0,1], linestyle="--", color="navy")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("results/roc_curve.png")
plt.close()

# ---- Feature Importance ----
importance = model.feature_importances_
feature_names = X.columns

fi_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False).head(20)

plt.figure(figsize=(8,6))
sns.barplot(x="Importance", y="Feature", data=fi_df)
plt.title("Top 20 Feature Importances - XGBoost")
plt.tight_layout()
plt.savefig("results/feature_importance.png")
plt.close()

print("[INFO] Graphical results saved in 'results/' directory.")
print("\n[INFO] Pipeline execution completed successfully.")
