import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

iris=datasets.load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
print(df)
print(df.head(5))

df['target']=iris.target
df['target_names']=iris.target_names[iris.target]
print(df.head(5))

x=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scalar=StandardScaler()
x_train=scalar.fit_transform(x_train)
x_test=scalar.transform(x_test)

svm_model=SVC(kernel='rbf',C=1.0,gamma='scale')
svm_model.fit(x_train,y_train)
y_pred=svm_model.predict(x_test)

print("Training Accuracy:",svm_model.score(x_train,y_train))
print("Testing Accuracy:",svm_model.score(x_test,y_test))

from sklearn.model_selection import cross_val_score

scores=cross_val_score(svm_model,x,y,cv=5)
print("Cross validation scores:",scores)
print("Average score:",scores.mean())
print("accuracy score:",accuracy_score(y_test,y_pred))
print("\nConfusion matrix:\n",confusion_matrix(y_test,y_pred))
print("\n Classification report:\n",classification_report(y_test,y_pred,target_names=iris.target_names))
test_results = pd.DataFrame(scalar.inverse_transform(x_test), columns=iris.feature_names)
test_results['actual_flower'] = iris.target_names[y_test]
test_results['predicted_flower'] = iris.target_names[y_pred]
print(test_results)