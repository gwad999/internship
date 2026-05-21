
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

CSV_FILE = "data_mobile_price_range.csv"   # <-- change if needed

df = pd.read_csv(CSV_FILE)
print(df.head())
print("\n Missing values:", df.isnull().sum().sum())
X = df.drop("price_range", axis=1)   
y = df["price_range"]                
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
models = {
    "Decision Tree"               : (DecisionTreeClassifier(random_state=42),   False),
    "Random Forest (Bagging)"     : (RandomForestClassifier(n_estimators=100, random_state=42), False),
    "Gradient Boosting (Boosting)": (GradientBoostingClassifier(n_estimators=100, random_state=42), False),
    "Logistic Regression"         : (LogisticRegression(max_iter=1000, random_state=42), True),}
results = {}
for name, (model, use_scaled) in models.items():
    Xtr = X_train_scaled if use_scaled else X_train
    Xte = X_test_scaled  if use_scaled else X_test

    model.fit(Xtr, y_train)
    preds    = model.predict(Xte)
    accuracy = accuracy_score(y_test, preds)
    results[name] = {"model": model, "preds": preds, "accuracy": accuracy, "scaled": use_scaled}
    print(f"   {name:<35} → {accuracy*100:.2f}%")
best_name = "Random Forest (Bagging)"
best      = results[best_name]
labels      = ["Low (0)", "Medium (1)", "High (2)", "Very High (3)"]
short_labels = ["Low", "Medium", "High", "Very High"]
print("\n Classification Report:")
print(classification_report(y_test, best["preds"], target_names=short_labels))
rf_model  = results["Random Forest (Bagging)"]["model"]
feat_imp  = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_imp  = feat_imp.sort_values(ascending=False)




