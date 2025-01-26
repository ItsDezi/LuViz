import plotly.graph_objects as go
from luviz.parser import read_market_data, read_trade_data
import pandas as pd 
from plotly.subplots import make_subplots

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

def plot_price_time_series_scatter_subplots(stock: str, period: int) -> go.Figure:
    df_market = read_market_data(stock, period)
    df_trade = read_trade_data(stock, period)

    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    fig.add_trace(go.Scatter(x=df_market['timestamp'], y=df_market['bidVolume'], mode='lines', marker=dict(color='blue'), name='Bid Volume'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_market['timestamp'], y=df_market['askVolume'], mode='lines', marker=dict(color='red'), name='Ask Volume'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_market['timestamp'], y=df_market['bidPrice'], mode='lines', marker=dict(color='green'), name='Bid Price'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_market['timestamp'], y=df_market['askPrice'], mode='lines', marker=dict(color='orange'), name='Ask Price'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_trade['timestamp'], y=df_trade['price'], mode='lines', marker=dict(color='purple'), name='Trade Price'), row=5, col=1)

    fig.update_layout(title=f'Volume and Price over Time',
                      xaxis_title='Time',
                      xaxis_rangeslider_visible=False,
                      height=1600)

    return fig

def plot_toggleable_volume_price(stock: str, period: int) -> go.Figure:
    df_market = read_market_data(stock, period)
    df_trade = read_trade_data(stock, period)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    # Add volume traces
    fig.add_trace(go.Scatter(x=df_market['timestamp'], y=df_market['bidVolume'], mode='lines', marker=dict(color='blue'), name='Bid Volume', visible=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_market['timestamp'], y=df_market['askVolume'], mode='lines', marker=dict(color='red'), name='Ask Volume', visible=True), row=1, col=1)

    # Add price traces
    fig.add_trace(go.Scatter(x=df_market['timestamp'], y=df_market['bidPrice'], mode='lines', marker=dict(color='green'), name='Bid Price', visible=True), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_market['timestamp'], y=df_market['askPrice'], mode='lines', marker=dict(color='orange'), name='Ask Price', visible=True), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_trade['timestamp'], y=df_trade['price'], mode='lines', marker=dict(color='purple'), name='Trade Price', visible=True), row=2, col=1)

    initial_visible = [True, True, True, True, True]
    # Update layout with checkboxes
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=list([
                    dict(label="Bid Volume",
                         method="update",
                         args=[{"visible": [True, False, True, True, True]},
                               {"title": "Bid Volume"}],
                         args2=[{"visible": [False, False, False, False, False]},
                                {"title": "Bid Volume"}]),
                    dict(label="Ask Volume",
                         method="update",
                         args=[{"visible": [False, True, True, True, True]},
                               {"title": "Ask Volume"}],
                         args2=[{"visible": [False, False, False, False, False]},
                                {"title": "Ask Volume"}]),
                    dict(label="Both Volumes",
                         method="update",
                         args=[{"visible": [True, True, True, True, True]},
                               {"title": "Both Volumes"}],
                         args2=[{"visible": [False, False, False, False, False]},
                                {"title": "Both Volumes"}])
                ]),
                direction="down",
                showactive=False,
                x=-0.3,
                xanchor="left",
                y=1,
                yanchor="top",
                bgcolor='rgba(50, 50, 50, 0.8)',
                bordercolor='white',
                font=dict(color='white')
            ),
            dict(
                type="buttons",
                buttons=list([
                    dict(label="Bid Price",
                         method="update",
                         args=[{"visible": [True, True, True, False, False]},
                               {"title": "Bid Price"}],
                         args2=[{"visible": [False, False, False, False, False]},
                                {"title": "Bid Price"}]),
                    dict(label="Ask Price",
                         method="update",
                         args=[{"visible": [True, True, False, True, False]},
                               {"title": "Ask Price"}],
                         args2=[{"visible": [False, False, False, False, False]},
                                {"title": "Ask Price"}]),
                    dict(label="Trade Price",
                         method="update",
                         args=[{"visible": [True, True, False, False, True]},
                               {"title": "Trade Price"}],
                         args2=[{"visible": [False, False, False, False, False]},
                                {"title": "Trade Price"}]),
                    dict(label="All Prices",
                         method="update",
                         args=[{"visible": [True, True, True, True, True]},
                               {"title": "All Prices"}],
                         args2=[{"visible": [False, False, False, False, False]},
                                {"title": "All Prices"}])
                ]),
                direction="down",
                showactive=False,
                x=-0.3,
                xanchor="left",
                y=0.5,
                yanchor="top",
                bgcolor='rgba(50, 50, 50, 0.8)',
                bordercolor='white',
                font=dict(color='white')
            )
        ]
    )

    fig.update_layout(title=f'Volume and Price over Time',
                      xaxis_title='Time',
                      xaxis_rangeslider_visible=False,
                      height=800,
                      template='plotly_dark')

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
