
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load the dataset
df = pd.read_csv("sales.csv")

print(df.head())
print("\nDataset shape:", df.shape)

df.info()


# Check the data
print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nDataset summary:")
print(df.describe())


# Check the relationship between advertising and sales
correlation = df["Advertising_Spend"].corr(df["Sales"])
print("\nCorrelation:", correlation)

plt.scatter(df["Advertising_Spend"], df["Sales"])
plt.xlabel("Advertising Spend")
plt.ylabel("Sales")
plt.title("Advertising Spend vs Sales")
plt.grid(True)
plt.show()


# Separate input and output
X = df[["Advertising_Spend"]]
y = df["Sales"]

print("\nX shape:", X.shape)
print("y shape:", y.shape)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# Create a simple baseline model
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)


# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel coefficient:", model.coef_[0])
print("Model intercept:", model.intercept_)


# Evaluate model performance
def evaluate(actual, predicted, name):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = mse ** 0.5
    r2 = r2_score(actual, predicted)

    print(f"\n{name}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.3f}")


evaluate(y_test, baseline_pred, "Baseline Results")
evaluate(y_test, y_pred, "Linear Regression Results")


# Compare actual and predicted sales
results = pd.DataFrame({
    "Advertising_Spend": X_test["Advertising_Spend"].values,
    "Actual_Sales": y_test.values,
    "Predicted_Sales": y_pred
})

results["Error"] = results["Actual_Sales"] - results["Predicted_Sales"]

print("\nPrediction results:")
print(results.round(2))


# Plot actual and predicted values
plt.scatter(
    X_test["Advertising_Spend"],
    y_test,
    label="Actual Sales"
)

plt.scatter(
    X_test["Advertising_Spend"],
    y_pred,
    marker="x",
    label="Predicted Sales"
)

plt.xlabel("Advertising Spend")
plt.ylabel("Sales")
plt.title("Actual vs Predicted Sales")
plt.legend()
plt.grid(True)
plt.show()


# Check prediction errors
plt.scatter(y_pred, results["Error"])
plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Sales")
plt.ylabel("Error")
plt.title("Residual Plot")
plt.grid(True)
plt.show()


# Train the final model using all available data
final_model = LinearRegression()
final_model.fit(X, y)

# Predict sales for a new advertising budget
new_data = pd.DataFrame({
    "Advertising_Spend": [750]
})

predicted_sales = final_model.predict(new_data)[0]

print("\nNew prediction:")
print("Advertising Spend: 750")
print(f"Predicted Sales: {predicted_sales:.2f}")

print("\nDone!")
```