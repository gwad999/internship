import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = pd.DataFrame({
    'Rooms': [2, 3, 4, 3, 5],
    'Area': [1000, 1500, 2000, 1200, 2500],
    'Location': [1, 2, 3, 2, 1],  
    'Price': [200000, 300000, 400000, 250000, 500000]
})

X = data[['Rooms', 'Area', 'Location']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted Prices:", y_pred)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
