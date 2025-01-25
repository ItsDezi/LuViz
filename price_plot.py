import pandas as pd
import plotly.graph_objects as go
from utils.read_stock_data import read_market_data, read_trade_data

market_data_combined_df = read_market_data('A', 1)
trade_data__A_df = read_trade_data('A', 1)

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
