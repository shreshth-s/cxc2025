import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Load the dataset
df = pd.read_csv("Dataset1.csv")

# Rename columns for Prophet
df = df.rename(columns={'year': 'ds', 'total_investment': 'y'})
df['ds'] = pd.to_datetime(df['ds'], format='%Y')

# Force an upper-bound (capacity) to control growth
df['cap'] = df['y'].max() * 1.5  # Increase this if needed to push more growth
df['floor'] = df['y'].min() * 0.8  # Ensures floor doesn't go to zero

# Initialize Prophet with logistic growth
model = Prophet(growth="logistic")
model.fit(df)

# Predict the next 6 years (2025-2030)
future = model.make_future_dataframe(periods=6, freq='Y')
future['cap'] = df['cap'].max()  # Apply the same capacity
future['floor'] = df['floor'].min()  # Apply floor

forecast = model.predict(future)

# Convert values to billions for better readability
df['y'] = df['y'] / 1e9
forecast['yhat'] = forecast['yhat'] / 1e9
forecast['yhat_lower'] = forecast['yhat_lower'] / 1e9
forecast['yhat_upper'] = forecast['yhat_upper'] / 1e9

# Plot the forecast with forced growth
plt.figure(figsize=(10, 6))
plt.plot(df['ds'], df['y'], 'bo-', label="Actual Data")
plt.plot(forecast['ds'], forecast['yhat'], 'r--', label="Predicted Trend")
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='purple', alpha=0.2)
plt.xlabel("Year")
plt.ylabel("Total Investment (Billions $)")
plt.title("Predicted Investment Trends (2025-2030)")
plt.legend()
plt.show()
