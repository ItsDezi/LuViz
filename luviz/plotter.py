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
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Add traces with consistent names and indices
    traces = [
        ('Bid Volume', df_market['timestamp'], df_market['bidVolume'], 1),
        ('Ask Volume', df_market['timestamp'], df_market['askVolume'], 1),
        ('Bid Price', df_market['timestamp'], df_market['bidPrice'], 2),
        ('Ask Price', df_market['timestamp'], df_market['askPrice'], 2),
        ('Trade Price', df_trade['timestamp'], df_trade['price'], 2)
    ]

    for i, (name, x, y, row) in enumerate(traces):
        fig.add_trace(
            go.Scatter(
                x=x, 
                y=y,
                mode='lines',
                marker=dict(color=colors[i]),
                name=name,
                visible=True
            ),
            row=row,
            col=1
        )

    # Create update menus (buttons) for each trace
    um = []
    menuadjustment = 0.15
    buttonX = -0.1
    buttonY = 1 + menuadjustment

    for i, (name, _, _, _) in enumerate(traces):
        button = dict(
            method='restyle',
            label=name,
            visible=True,
            args=[{'visible': True, 'line.color': colors[i]}, [i]],
            args2=[{'visible': False, 'line.color': colors[i]}, [i]]
        )
        
        buttonY = buttonY - menuadjustment
        um.append({
            'buttons': [button],
            'showactive': False,
            'y': buttonY,
            'x': buttonX,
            'type': 'buttons'
        })

    # Add "All" button
    # button_all = dict(
    #     method='restyle',
    #     label='All',
    #     visible=True,
    #     args=[{'visible': True}],
    #     args2=[{'visible': False}]
    # )
    
    # um.append({
    #     'buttons': [button_all],
    #     'showactive': True,
    #     'y': buttonY - menuadjustment,
    #     'x': buttonX,
    #     'type': 'buttons'
    # })

    fig.update_layout(
        title='Volume and Price over Time',
        xaxis_title='Time',
        xaxis_rangeslider_visible=False,
        height=800,
        template='plotly_dark',
        showlegend=True,
        updatemenus=um
    )

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


def plot_stock_comparison_tool(stock1: str, period1: int, stock2: str, period2: int) -> go.Figure:
    df_market1 = read_market_data(stock1, period1)
    df_trade1 = read_trade_data(stock1, period1)
    df_market2 = read_market_data(stock2, period2)
    df_trade2 = read_trade_data(stock2, period2)
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.02)
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Add traces with consistent names and indices
    traces = [
        ('Bid Volume', df_market1['timestamp'], df_market1['bidVolume'], 1),
        ('Ask Volume', df_market1['timestamp'], df_market1['askVolume'], 1),
        ('Bid Price', df_market1['timestamp'], df_market1['bidPrice'], 2),
        ('Ask Price', df_market1['timestamp'], df_market1['askPrice'], 2),
        ('Trade Price', df_trade1['timestamp'], df_trade1['price'], 2)
    ]
    traces2 = [
        ('Bid Volume', df_market2['timestamp'], df_market2['bidVolume'], 1),
        ('Ask Volume', df_market2['timestamp'], df_market2['askVolume'], 1),
        ('Bid Price', df_market2['timestamp'], df_market2['bidPrice'], 2),
        ('Ask Price', df_market2['timestamp'], df_market2['askPrice'], 2),
        ('Trade Price', df_trade2['timestamp'], df_trade2['price'], 2)
    ]
    for i, (name, x, y, col) in enumerate(traces):
        fig.add_trace(
            go.Scatter(
                x=x, 
                y=y,
                mode='lines',
                marker=dict(color=colors[i]),
                name=name,
                visible=True
            ),
            row=1,
            col=col
        )
    for i, (name, x, y, col) in enumerate(traces2):
        fig.add_trace(
            go.Scatter(
                x=x, 
                y=y,
                mode='lines',
                marker=dict(color=colors[i]),
                name=name,
                visible=True
            ),
            row=2,
            col=col
        )
    # Create update menus (buttons) for each trace
    um = []
    menuadjustment = 0.15
    buttonX = 0.5
    buttonY = -0.2

    for i, (name, _, _, _) in enumerate(traces):
        # For each metric, we need to control visibility of same metric in both stocks
        # i is for stock1, i+5 is for stock2 (since there are 5 traces per stock)
        button = dict(
            method='restyle',
            label=name,
            visible=True,
            args=[{'visible': True, 'line.color': colors[i]}, [i, i+5]],
            args2=[{'visible': False, 'line.color': colors[i]}, [i, i+5]]
        )
        
        buttonX = buttonX - menuadjustment
        um.append({
            'buttons': [button],
            'showactive': False,
            'y': buttonY,
            'x': buttonX,
            'type': 'buttons',
            'xanchor': 'center',
            'yanchor': 'top'
        })

    fig.update_layout(
        title='Volume and Price over Time',
        xaxis_title='Time',
        xaxis_rangeslider_visible=False,
        autosize=True,
        template='plotly_dark',
        showlegend=True,
        updatemenus=um
    )

    return fig

def plot_trade_predicted_subplots(stock: str, period: int, predicitions) -> go.Figure:
    df_trade = read_trade_data(stock, period)
    predicitions = predicitions[predicitions['predictions'] != 0]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    fig.add_trace(go.Scatter(x=df_trade['timestamp'], y=df_trade['price'], mode='lines', marker=dict(color='purple'), name='Trade Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_trade['timestamp'], y=df_trade['mid_price'], mode='lines', marker=dict(color='green'), name='Predicted Trade Price'), row=2, col=1)

    fig.update_layout(title=f'Price over Time',
                      xaxis_title='Time',
                      xaxis_rangeslider_visible=False,
                      height=800)

    return fig