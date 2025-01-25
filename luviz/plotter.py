import plotly.graph_objects as go
from luviz.parser import read_market_data, read_trade_data

def plot_price_time_series_scatter(stock:str, period: int, price_type: str) -> go.Figure:

    if price_type in ['ask', 'bid']:
        price_id = price_type + 'Price'
        df = read_market_data(stock, period)
    elif price_type == 'trade':
        price_id = 'price'
        df = read_trade_data(stock, period)
    else:
        raise ValueError("Unknown price type error")
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['timestamp'], y=df[price_id], mode='lines', name=price_type.capitalize() + ' Price'))

    fig.update_layout(title=f'{price_type.capitalize()} Price vs time',
                    xaxis_title='Time',
                    yaxis_title='Price',
                    xaxis=dict(tickformat='%Y-%m-%d %H:%M:%S', tickangle=45),
                    template='plotly_white')
    return fig


def plot_group_price_time_series_scatter(stock:list[str], period: int, price_type: str) -> go.Figure:

    if price_type in ['ask', 'bid']:
        price_id = price_type + 'Price'
        df = read_market_data(stock, period)
    elif price_type == 'trade':
        price_id = 'price'
        df = read_trade_data(stock, period)
    else:
        raise ValueError("Unknown price type error")
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['timestamp'], y=df[price_id], mode='lines', name=price_type.capitalize() + ' Price'))

    fig.update_layout(title=f'{price_type.capitalize()} Price vs time',
                    xaxis_title='Time',
                    yaxis_title='Price',
                    xaxis=dict(tickformat='%Y-%m-%d %H:%M:%S', tickangle=45),
                    template='plotly_white')
    return fig
