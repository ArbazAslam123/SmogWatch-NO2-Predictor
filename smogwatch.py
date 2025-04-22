
import zipfile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Loading data from the zip file
zip_path = "archive.zip"  

# Loading the CSV directly from the zip without extracting the full file
with zipfile.ZipFile(zip_path) as zip_ref:
    with zip_ref.open("pollution_us_2000_2016.csv") as f:
        df = pd.read_csv(f)  

# Print the column names to check for any discrepancies
print(df.columns)

# 3. Preprocess
df['Date Local'] = pd.to_datetime(df['Date Local'])  # Convert 'Date Local' to datetime format
df = df[['Date Local', 'City', 'State', 'NO2 Mean']]  # Select relevant columns (use NO2 Mean instead of PM2.5 or PM10)
df.dropna(inplace=True)  # Drop rows with missing values

# 4. EDA - Visualize NO2 levels for Los Angeles over time
sns.lineplot(data=df[df['City'] == 'Los Angeles'], x='Date Local', y='NO2 Mean')
plt.title("NO2 Levels in Los Angeles Over Time")
plt.xlabel('Date')
plt.ylabel('NO2 Mean')
plt.show()

# 5. Feature Engineering - Extract useful time-based features
df['Year'] = df['Date Local'].dt.year
df['Month'] = df['Date Local'].dt.month
df['Day'] = df['Date Local'].dt.day
df['Weekday'] = df['Date Local'].dt.weekday

# 6. Modeling - Train a model using NO2 Mean as the target variable
city_df = df[df['City'] == 'Los Angeles']  # Filter for Los Angeles only
X = city_df[['Year', 'Month', 'Day', 'Weekday']]  # Feature matrix
y = city_df['NO2 Mean']  # Target variable (NO2 Mean)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)  # Fit the model

# Make predictions
preds = model.predict(X_test)

# 7. Evaluation - Evaluate model performance
print("MAE:", mean_absolute_error(y_test, preds))  # Mean Absolute Error
print("RMSE:", mean_squared_error(y_test, preds, squared=False))  # Root Mean Squared Error
print("R² Score:", r2_score(y_test, preds))  # R² Score

# Optionally, visualize predicted vs actual values
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, preds, label='Predicted', color='red', linestyle='--')
plt.title('Actual vs Predicted NO2 Levels in Los Angeles')
plt.xlabel('Index')
plt.ylabel('NO2 Mean')
plt.legend()
plt.show()




