import pandas as pd
import plotly.graph_objects as go
from luviz.parser import read_market_data, read_trade_data
from luviz.plotter import plot_price_time_series_scatter

fig = plot_price_time_series_scatter('A', 1, 'ask')

fig.show()
