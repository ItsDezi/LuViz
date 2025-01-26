import plotly.graph_objects as go
from luviz.parser import read_market_data, read_trade_data
import pandas as pd 

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
                    xaxis=dict(tickformat='%H:%M:%S', tickangle=45),
                    template='plotly_white',
                    xaxis_rangeslider_visible=True,
                    autosize=False,
                    width=1000,
                    height=600)
    return fig

def plot_std_dev(stock: str, period: int, interval: str = '30S') -> go.Figure:
    if interval not in ['30S', '60S']:
        raise ValueError("Interval must be '30S' or '60S'")
    
    df = read_market_data(stock, period)
    df = df.set_index('timestamp')
    
    std_dev = df['bidPrice'].resample(interval).std()
    std_dev_df = std_dev.reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=std_dev_df['timestamp'], y=std_dev_df['bidPrice'], mode='lines', name='Standard Deviation'))
    
    fig.update_layout(title=f'Standard Deviation of Bid Price over {interval} intervals',
                        xaxis_title='Time',
                        yaxis_title='Standard Deviation',
                        xaxis=dict(tickformat='%H:%M:%S', tickangle=45),
                        template='plotly_white',
                        xaxis_rangeslider_visible=True,
                        autosize=False,
                        width=1000,
                        height=600)
    
    return fig

def compute_ohlc(df, freq='1S'):
    df = df.set_index('timestamp')
    ohlc_dict = {
        'openPrice': df['bidPrice'].resample(freq).first(),
        'highPrice': df['bidPrice'].resample(freq).max(),
        'lowPrice': df['bidPrice'].resample(freq).min(),
        'closePrice': df['bidPrice'].resample(freq).last()
    }
    ohlc_df = pd.DataFrame(ohlc_dict)
    ohlc_df.reset_index(inplace=True)
    return ohlc_df

def plot_candlestick(stock: str, period: int, freq='1S') -> go.Figure:
    df = read_market_data(stock, period)
    df = compute_ohlc(df, freq)

    fig = go.Figure(data=[go.Candlestick(x=df['timestamp'],
                                            open=df['openPrice'],
                                            high=df['highPrice'],
                                            low=df['lowPrice'],
                                            close=df['closePrice'])])

    fig.update_layout(title=f'Candlestick chart for {stock}',
                        xaxis_title='Time',
                        yaxis_title='Price',
                        xaxis=dict(tickformat='%H:%M:%S', tickangle=45),
                        template='plotly_white',
                        xaxis_rangeslider_visible=True,
                        autosize=False,
                        width=1000,
                        height=600)
    return fig

def plot_group_price_time_series_scatter(stocks:list[str], period: int, price_types: list[str]) -> go.Figure:

    fig = go.Figure()
    for stock in stocks:
        tracker = 0
        for price_type in price_types:
            if price_type in ['ask', 'bid']:
                price_id = price_type + 'Price'
                if tracker > 0: continue
                df = read_market_data(stock, period)
                tracker += 1
            elif price_type == 'trade':
                price_id = 'price'
                df = read_trade_data(stock, period)
            else:
                raise ValueError("Unknown price type error")

            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[price_id], mode='lines', name=stock + ' ' + price_type.capitalize() + ' Price', legendgroup=price_type))

    fig.update_layout(title=f'{" ".join([pt.capitalize() for pt in price_types])} Price vs time',
                    xaxis_title='Time',
                    yaxis_title='Price',
                    legend=dict(
                        title='Stock',
                        traceorder='normal'
                    ),
                    xaxis=dict(tickformat='%H:%M:%S', tickangle=45),
                    template='plotly_white',
                    xaxis_rangeslider_visible=True,
                    autosize=False,
                    width=1000,
                    height=600)
    
    return fig
