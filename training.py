import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import datetime
import joblib

# Function to convert timestamp to seconds
def convert_timestamp_to_seconds(ts):
    h, m, s = map(float, ts.split(':'))
    return h * 3600 + m * 60 + s

# Base directory containing period directories
base_dir = 'data/train'

# Loop through each period directory
for period_dir in os.listdir(base_dir):
    period_path = os.path.join(base_dir, period_dir)
    if os.path.isdir(period_path):
        print(f"Processing period: {period_dir}")
        
        # Initialize an empty DataFrame to hold all data for the current period
        all_data = pd.DataFrame()
        columns = None
        
        # Loop through all CSV files in the nested directories within the period directory
        for root, dirs, files in os.walk(period_path):
            for file in files:
                if file.endswith('.csv') and file.startswith('market_data'):
                    file_path = os.path.join(root, file)
                    print(file)
                    if columns is None:
                        # Read the first file with headers
                        data = pd.read_csv(file_path)
                        columns = data.columns
                    else:
                        # Read subsequent files without headers
                        data = pd.read_csv(file_path, header=None)
                        data.columns = columns
                    
                    # Extract stock name from the folder path
                    stock_name = os.path.basename(os.path.dirname(file_path))
                    data['stock'] = stock_name
                    
                    # Convert timestamp to a numerical feature
                    if 'timestamp' in data.columns:
                        try:
                            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
                            data = data.dropna(subset=['timestamp'])
                            data['timestamp_seconds'] = data['timestamp'].apply(lambda x: convert_timestamp_to_seconds(x.strftime('%H:%M:%S')))
                        except Exception as e:
                            print(f"Error converting timestamp in file {file_path}: {e}")
                    
                    # Convert askPrice and bidPrice to numeric
                    data['askPrice'] = pd.to_numeric(data['askPrice'], errors='coerce')
                    data['bidPrice'] = pd.to_numeric(data['bidPrice'], errors='coerce')
                    
                    # Feature engineering: calculate spread, mid-price
                    data['spread'] = data['askPrice'] - data['bidPrice']
                    data['mid_price'] = (data['askPrice'] + data['bidPrice']) / 2
                    
                    # Append to all_data DataFrame
                    all_data = pd.concat([all_data, data], ignore_index=True)
        
        # Target: Define a significant event
        # Example: Event is 1 if there is a large askVolume or bidVolume change (> 20%)
        # Ensure bidVolume and askVolume are numeric
        all_data['bidVolume'] = pd.to_numeric(all_data['bidVolume'], errors='coerce')
        all_data['askVolume'] = pd.to_numeric(all_data['askVolume'], errors='coerce')
        
        # Drop rows with NaN values in bidVolume and askVolume
        all_data.dropna(subset=['bidVolume', 'askVolume'], inplace=True)
        
        # Define the event based on significant changes in bidVolume or askVolume
        all_data['event'] = np.where(
            (all_data['bidVolume'].pct_change().abs() > 0.2) | 
            (all_data['askVolume'].pct_change().abs() > 0.2), 
            1, 
            0
        )
        
        # Drop rows with NaN (caused by pct_change)
        all_data.dropna(inplace=True)
        
        # Define features and target
        features = ['bidVolume', 'bidPrice', 'askVolume', 'askPrice', 'spread', 'mid_price', 'timestamp_seconds', 'stock']
        target = 'event'
        
        # Convert stock feature to numeric
        all_data['stock'] = all_data['stock'].astype('category').cat.codes
        
        X = all_data[features]
        y = all_data[target]
        
        # Normalize features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train a Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        print(f"Period: {period_dir}")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importances = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        
        print("\nFeature Importances:\n", feature_importances)
        
        # Save the model and scaler
        model_filename = f'model_{period_dir}.joblib'
        scaler_filename = f'scaler_{period_dir}.joblib'
        joblib.dump(model, model_filename)
        joblib.dump(scaler, scaler_filename)
        print(f"Model and scaler saved for period {period_dir}")
