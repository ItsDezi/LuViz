import pandas as pd
import plotly.express as px

# Function to read CSV and plot two columns
def plot_csv(file_path, timestamp, askVolume, bidVolume):
    # Read the CSV file and parse the x column as datetime
    data = pd.read_csv(file_path, parse_dates=[timestamp])
    print(data.columns)
    # Plot the data using Plotly
    fig = px.line(data, x=timestamp, y=[askVolume,bidVolume], title=f'{timestamp} vs {askVolume}')
    #fig.add_selection(px.line(data, x=timestamp, y=bidVolume, title=f'{timestamp} vs {bidVolume}'))
    fig.show()

# Example usage
if __name__ == "__main__":
    file_path = 'data/TrainingData/Period1/A/market_data_A_0.csv'  # Replace with your CSV file path
    time = 'timestamp'
    askVolume = 'askVolume' 
    bidVolume = 'bidVolume'
    plot_csv(file_path, time, askVolume, bidVolume)
