import pandas as pd
import plotly.graph_objects as go

market_data_A_0_df = pd.read_csv('/home/mael/mchacks_2025/LuViz/data/TrainingData/Period1/A/market_data_A_0.csv', 
                                 parse_dates=['timestamp'])
market_data_A_1_df = pd.read_csv('/home/mael/mchacks_2025/LuViz/data/TrainingData/Period1/A/market_data_A_1.csv', header=None, 
                                 names=market_data_A_0_df.columns, parse_dates=['timestamp'])
market_data_combined_df = pd.concat([market_data_A_0_df, market_data_A_1_df], ignore_index=True)
trade_data__A_df = pd.read_csv('/home/mael/mchacks_2025/LuViz/data/TrainingData/Period1/A/trade_data__A.csv', parse_dates=['timestamp'])

print(market_data_combined_df)

fig = go.Figure()

fig.add_trace(go.Scatter(x=market_data_combined_df['timestamp'], y=market_data_combined_df['bidPrice'], mode='lines', name='Bid Price'))
fig.add_trace(go.Scatter(x=trade_data__A_df['timestamp'], y=trade_data__A_df['price'], mode='lines', name='Trade Price'))
# fig.add_trace(go.Scatter(x=trade_data__A_df['timestamp'], y=trade_data__A_df['price'], mode='markers', name='Trade Volume', 
                        #  marker=dict(color='red', size=5, opacity=0.5)))

fig.update_layout(title='Bid Price and Trade Price over time',
                  xaxis_title='Timestamp',
                  yaxis_title='Price',
                  legend_title='Legend',
                  xaxis=dict(tickformat='%Y-%m-%d %H:%M:%S', tickangle=45),
                  template='plotly_white')

fig.show()

# market_data_A_0_df = pd.read_csv('/home/mael/mchacks_2025/LuViz/data/TrainingData/Period1/A/market_data_A_0.csv', 
#                                  parse_dates=['timestamp'])
# market_data_A_1_df = pd.read_csv('/home/mael/mchacks_2025/LuViz/data/TrainingData/Period1/A/market_data_A_1.csv', header=None, 
#                                  names=market_data_A_0_df.columns, parse_dates=['timestamp'])
# market_data_combined_df = pd.concat([market_data_A_0_df, market_data_A_1_df], ignore_index=True)
# trade_data__A_df = pd.read_csv('/home/mael/mchacks_2025/LuViz/data/TrainingData/Period1/A/trade_data__A.csv', parse_dates=['timestamp'])

# print(market_data_combined_df)

# plt.figure(figsize=(10, 6))
# plt.plot(market_data_combined_df['timestamp'], market_data_combined_df['bidPrice'], label='Bid Price')
# plt.plot(trade_data__A_df['timestamp'], trade_data__A_df['price'], label='Trade Price')
# plt.scatter(trade_data__A_df['timestamp'], trade_data__A_df['price'], c='red', s=10, label='Trade Volume', alpha=0.5)

# plt.xlabel('Timestamp')
# plt.ylabel('Price')
# plt.title('Bid Price and Trade Price over time')
# plt.legend()
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()

# plt.savefig('plot.png')
# plt.show()