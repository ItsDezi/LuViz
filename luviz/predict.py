import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from luviz.parser import read_market_data

def predict(period: int, stock: str):
    # Load the model and scaler
    model_filename = 'model_Period1.joblib'  # Replace with your model file path
    scaler_filename = 'scaler_Period1.joblib'  # Replace with your scaler file path

    model = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)

    # Load new data
    new_data = read_market_data(stock, period)

    # Check if necessary columns are present
    required_columns = ['timestamp', 'askPrice', 'bidPrice', 'askVolume', 'bidVolume']
    for col in required_columns:
        if col not in new_data.columns:
            raise KeyError(f"Column '{col}' not found in the data")

    # Convert timestamp to a numerical feature
    def convert_timestamp_to_seconds(ts):
        time_part = ts.split(' ')[1]  # Extract the time part
        h, m, s = map(float, time_part.split(':'))
        return h * 3600 + m * 60 + s

    new_data['timestamp_seconds'] = new_data['timestamp'].apply(lambda x: convert_timestamp_to_seconds(str(x)))

    # Feature engineering: calculate spread, mid-price
    new_data['spread'] = new_data['askPrice'] - new_data['bidPrice']
    new_data['mid_price'] = (new_data['askPrice'] + new_data['bidPrice']) / 2

    # Print the first few rows to debug
    print(new_data.head())

    # Convert stock feature to numeric if it exists
    if 'stock' in new_data.columns:
        new_data['stock'] = new_data['stock'].astype('category').cat.codes
    else:
        new_data['stock'] = -1  # or handle it appropriately

    # Select features
    features = ['bidVolume', 'bidPrice', 'askVolume', 'askPrice', 'spread', 'mid_price', 'timestamp_seconds', 'stock']
    X_new = new_data[features]

    # Normalize features
    X_new_scaled = scaler.transform(X_new)

    # Make predictions
    predictions = model.predict(X_new_scaled)

    # Add predictions to the DataFrame
    new_data['predictions'] = predictions

    return new_data

# Example usage
if __name__ == "__main__":
    period = 1
    stock = 'A'
    predictions = predict(period, stock)
    print(predictions)