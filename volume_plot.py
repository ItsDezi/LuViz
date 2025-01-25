import pandas as pd
import plotly.graph_objects as go
import os
from plotly.subplots import make_subplots

# Function to read CSV and plot two columns
def plot_csv(file_path1, timestamp, askVolume, bidVolume):
    # Read the first CSV file and parse the timestamp column as datetime
    first_csv = [file for file in os.listdir(file_path1) if file.startswith('market_data') and file.endswith('0.csv')][0]
    #print(first_csv)
    files = [file for file in os.listdir(file_path1) if file.startswith('market_data') and not file.endswith('0.csv')]
    #print(files)
    data1 = pd.read_csv(file_path1 + '/' + first_csv, parse_dates=[timestamp])

    # Rename the columns of the second DataFrame to match the first
    for file in files:
        data2 = pd.read_csv(file_path1 + '/' + file, parse_dates=[timestamp], header=None, names=data1.columns)
        data2.columns = data1.columns
        data1 = pd.concat([data1, data2], ignore_index=True)
    
    # Plot the data using Plotly Graph Objects
    fig = make_subplots(rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

    fig.add_trace(go.Scatter(x=data1[timestamp], y=data1[askVolume], mode='lines', marker=dict(color='red'), name=askVolume), row=1, col=1)
    fig.add_trace(go.Scatter(x=data1[timestamp], y=data1[bidVolume], mode='lines', marker=dict(color='green'), name=bidVolume), row=1, col=1)
    
    fig.update_layout(title=f'{timestamp} vs {askVolume} and {bidVolume}',
                      xaxis_title=timestamp,
                      yaxis_title='Volume')
    #-----------------------------------------
    trade_data_file = [file for file in os.listdir(file_path1) if file.startswith('trade_data')][0]
    trade_data = pd.read_csv(file_path1 + '/' + trade_data_file, parse_dates=['timestamp'])
    fig.add_trace(go.Scatter(x=data1['timestamp'], y=data1['bidPrice'], mode='lines', name='Bid Price'), row=2, col=1)
    fig.add_trace(go.Scatter(x=data1['timestamp'], y=data1['askPrice'], mode='lines', name='Ask Price'), row=2, col=1)
    fig.add_trace(go.Scatter(x=trade_data['timestamp'], y=trade_data['price'], mode='lines', name='Trade Price'), row=2, col=1)

    #-----------------------------------------
    return fig
    #fig.show()

# Example usage
if __name__ == "__main__":
    file_path1 = 'data/TrainingData/Period1/C'  # Replace with your CSV file path
    time = 'timestamp'
    askVolume = 'askVolume' 
    bidVolume = 'bidVolume'
    plot_csv(file_path1, time, askVolume, bidVolume)
