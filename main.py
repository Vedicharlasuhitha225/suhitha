import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("data.csv")

# Features and target
X = data[['area', 'bedrooms']]
y = data['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", model.score(X_test, y_test))

# Input
area = int(input("Enter area: "))
bedrooms = int(input("Enter bedrooms: "))

# Prediction
prediction = model.predict([[area, bedrooms]])
print("Predicted Price:", int(prediction[0]))