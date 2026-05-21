import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

st.title("Iris Flower Prediction")
st.write("Enter the flower measurements below to predict the species.")

# Train model
iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel="rbf", C=1.0, gamma="scale")
model.fit(X_train, y_train)

# Input sliders
sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sw = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.0)
pl = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
pw = st.slider("Petal Width (cm)",  0.1, 2.5, 1.2)

# Predict
if st.button("Predict"):
    sample = scaler.transform([[sl, sw, pl, pw]])
    prediction = iris.target_names[model.predict(sample)[0]]
    st.write("Predicted Species:", prediction)
