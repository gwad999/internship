import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

wine=load_wine()
df=pd.DataFrame(wine.data,columns=wine.feature_names)
print(df) 
df['Wine_type']=wine.target
print(df)
df.shape
df.info()
X=wine.data
Y=wine.target
scalar=StandardScaler()
X_scaled=scalar.fit_transform(X)


k_values = range(1, 15)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled,Y, cv=5)
    cv_scores.append(scores.mean())
    
results = pd.DataFrame({'K Value': k_values, 'Mean CV Accuracy': cv_scores})
print("Cross-Validation Scores for Different K Values:")
print(results)

import matplotlib.pyplot as plt
plt.plot(k_values,cv_scores,marker='o')
plt.title(" K values Cross validation-Accuracy vs K")
plt.xlabel("K Values")
plt.ylabel("Mean CV Accuracy")
plt.savefig("K_values_CV_Accuracy.png")

best_k = k_values[np.argmax(cv_scores)]
print(f" Best K Value: {best_k}")
print(f"Highest Mean CV Accuracy: {max(cv_scores):.3f}")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy of KNN:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

sample = X_scaled[10].reshape(1, -1)
pred = knn.predict(sample)
print("Predicted Wine Type:", wine.target_names[pred][0])