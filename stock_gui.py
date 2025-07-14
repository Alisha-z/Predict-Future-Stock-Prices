import tkinter as tk
from tkinter import messagebox
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def predict_stock():
    symbol = entry.get().upper()
    if not symbol:
        messagebox.showerror("Error", "Please enter a stock symbol!")
        return

    try:
        # Fetch data
        data = yf.download(symbol, start="2022-01-01", end="2024-01-01")
        if data.empty:
            messagebox.showerror("Error", "No data found for that symbol.")
            return

        # Prepare features
        features = data[["Open", "High", "Low", "Volume"]]
        target = data["Close"]
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(y_test.values, label="Actual", color="blue")
        plt.plot(predictions, label="Predicted", color="red")
        plt.title(f"{symbol} - Actual vs Predicted Close Prices")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI Setup
window = tk.Tk()
window.title("Stock Price Predictor")
window.geometry("400x300")
window.configure(bg="#f0f0f0")

title = tk.Label(window, text="📈 Stock Predictor", font=("Helvetica", 16), bg="#f0f0f0")
title.pack(pady=20)

entry = tk.Entry(window, font=("Arial", 14), width=20, justify='center')
entry.insert(0, "AAPL")  # default symbol
entry.pack(pady=10)

predict_btn = tk.Button(window, text="Predict Closing Prices", font=("Arial", 12), command=predict_stock)
predict_btn.pack(pady=20)

signature = tk.Label(window, text="Created by Alisha", font=("Arial", 10, "italic"), bg="#f0f0f0", fg="gray")
signature.pack(side="bottom", pady=10)

window.mainloop()
