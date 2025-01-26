import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the model and scaler
model_filename = 'model_Period1.joblib'  # Replace with your model file path
scaler_filename = 'scaler_Period1.joblib'  # Replace with your scaler file path

model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

# Load new data
new_data_file_path = 'data/TrainingData/Period2/Period2/A/market_data_A_0.csv'  # Replace with your new data CSV file path
new_data = pd.read_csv(new_data_file_path)

# Convert timestamp to a numerical feature
def convert_timestamp_to_seconds(ts):
    h, m, s = map(float, ts.split(':'))
    return h * 3600 + m * 60 + s

new_data['timestamp_seconds'] = new_data['timestamp'].apply(lambda x: convert_timestamp_to_seconds(x.split('.')[0]))

# Feature engineering: calculate spread, mid-price
new_data['spread'] = new_data['askPrice'] - new_data['bidPrice']
new_data['mid_price'] = (new_data['askPrice'] + new_data['bidPrice']) / 2

# Convert stock feature to numeric if it exists
if 'stock' in new_data.columns:
    new_data['stock'] = new_data['stock'].astype('category').cat.codes
else:
    new_data['stock'] = -1  # or handle it appropriately

# Define features
features = ['bidVolume', 'bidPrice', 'askVolume', 'askPrice', 'spread', 'mid_price', 'timestamp_seconds', 'stock']

# Normalize features
X_new = new_data[features]
X_new_scaled = scaler.transform(X_new)

# Make predictions
predictions = model.predict(X_new_scaled)

# Add predictions to the new data
new_data['predictions'] = predictions

# Save the predictions to a new CSV file
new_data.to_csv('predictions.csv', index=False)

print("Predictions saved to predictions.csv")