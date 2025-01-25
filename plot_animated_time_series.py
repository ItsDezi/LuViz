import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import time

# Function to read CSV data
def read_csv_data(file_path, header=True, columns=None):
    if header:
        return pd.read_csv(file_path, parse_dates=['timestamp'])
    else:
        return pd.read_csv(file_path, header=None, names=columns, parse_dates=['timestamp'])

# File paths
file_path1 = '/home/mael/mchacks_2025/LuViz/data/TrainingData/Period1/A/market_data_A_0.csv'
file_path2 = '/home/mael/mchacks_2025/LuViz/data/TrainingData/Period1/A/market_data_A_1.csv'
trade_file_path = '/home/mael/mchacks_2025/LuViz/data/TrainingData/Period1/A/trade_data__A.csv'

# Read initial data
market_data_A_0_df = read_csv_data(file_path1)
market_data_A_1_df = read_csv_data(file_path2, header=False, columns=market_data_A_0_df.columns)
market_data_combined_df = pd.concat([market_data_A_0_df, market_data_A_1_df], ignore_index=True)
trade_data__A_df = read_csv_data(trade_file_path)

# Create figure
fig = make_subplots(rows=1, cols=1)

# Add traces for bid price and trade price
fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Bid Price'), row=1, col=1)
fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Trade Price'), row=1, col=1)

# Function to update the plot
def update_plot():
    for k in range(1, len(market_data_combined_df)):
        fig.data[0].x = market_data_combined_df['timestamp'][:k]
        fig.data[0].y = market_data_combined_df['bidPrice'][:k]
        fig.data[1].x = trade_data__A_df['timestamp'][:k]
        fig.data[1].y = trade_data__A_df['price'][:k]
        fig.update_layout(title='Bid Price and Trade Price over time',
                          xaxis_title='Timestamp',
                          yaxis_title='Price',
                          legend_title='Legend',
                          xaxis=dict(tickformat='%Y-%m-%d %H:%M:%S', tickangle=45),
                          template='plotly_white')
        pio.show(fig, auto_open=False)
        time.sleep(0.1)

# Call the update function
update_plot()