import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. Get stock data (e.g., Apple)
stock_symbol = "AAPL"  # You can change to "TSLA", "MSFT", etc.
data = yf.Ticker(stock_symbol).history(start="2022-01-01", end="2024-01-01")


# 2. Show basic info
print("Stock Data Sample:")
print(data.head())


# 3. Select features and target
features = data[["Open", "High", "Low", "Volume"]]
target = data["Close"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# 5. Train Linear Regression model
model = RandomForestRegressor()
model.fit(X_train, y_train.values.ravel())



# 6. Predict on test data
predictions = model.predict(X_test)


# 7. Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual Close Price", color="blue")
plt.plot(predictions, label="Predicted Close Price", color="red")
plt.title(f"{stock_symbol} - Actual vs Predicted Closing Prices")
plt.xlabel("Time (Days)")
plt.ylabel("Price")
plt.legend()
plt.show()
