import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from luviz.parser import read_market_data
from luviz.plotter import plot_candlestick, plot_std_dev, plot_group_price_time_series_scatter
from scipy.signal import correlate
# Replace the problematic model import with:
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel
import torch

def train_model(train_data, target_stock):
    """Updated model training with aligned lengths"""
    config = TimeSeriesTransformerConfig(
        prediction_length=10,
        context_length=70,  # Must match history_length - lags_sequence
        input_size=1,
        num_time_features=1,
        encoder_layers=2,
        decoder_layers=2,
        lags_sequence=[1, 7, 14],  # Reduced lags to fit context
        num_static_real_features=0,
        scaling=False
    )
    
    model = TimeSeriesTransformerModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for _ in range(5):
        model.train()
        for batch in create_batches(train_data, target_stock):
            outputs = model(
                past_values=batch['past_values'],
                past_time_features=batch['past_time_features'],
                past_observed_mask=batch['past_observed_mask'],
                future_values=batch['future_values'],
                future_time_features=batch['future_time_features']
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    return model

def create_batches(data, target_stock, batch_size=32):
    """Create aligned batches with exact length matching"""
    history_length = 70  # Matches context_length + max(lags_sequence)
    prediction_length = 10
    total_length = history_length + prediction_length
    
    num_samples = len(data) - total_length
    
    for i in range(0, num_samples, batch_size):
        batch = {
            'past_values': [],
            'past_time_features': [],
            'past_observed_mask': [],
            'future_values': [],
            'future_time_features': []
        }
        
        for j in range(i, min(i+batch_size, num_samples)):
            window = data[target_stock].iloc[j:j+total_length]
            
            # Strict length enforcement
            past_values = window.values[:history_length]
            past_time = np.arange(history_length).reshape(-1, 1)
            future_time = np.arange(history_length, history_length+prediction_length).reshape(-1, 1)
            
            batch['past_values'].append(past_values)
            batch['past_time_features'].append(past_time)
            batch['past_observed_mask'].append(np.ones_like(past_values))
            batch['future_values'].append(window.values[history_length:])
            batch['future_time_features'].append(future_time)
        
        yield {
            k: torch.tensor(np.array(v), dtype=torch.float32 if k != 'past_observed_mask' else torch.bool)
            for k, v in batch.items()
        }

# Custom preprocessing and analysis functions
def preprocess_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Process raw data from parser into analysis-ready format"""
    processed = raw_data.rename(columns={
        'bidPrice': 'best_bid',
        'askPrice': 'best_ask',
        'price': 'trade_price'
    }).set_index('timestamp')  # Add this to set datetime index
    processed['mid_price'] = (processed['best_bid'] + processed['best_ask']) / 2
    return processed

def detect_volatility_events(data: pd.DataFrame, window: int = 10, threshold: float = 2.0):
    """Enhanced volatility detection with multi-timeframe analysis"""
    # Ensure we're working with a time-sorted index
    data = data.sort_index()
    
    # Calculate rolling std using time-based windows
    data['30s_std'] = data['mid_price'].rolling(f'{30*int(window)}S').std()
    data['60s_std'] = data['mid_price'].rolling(f'{60*int(window)}S').std()
    
    data['volatility_event'] = np.where(
        (data['30s_std'] > threshold) | (data['60s_std'] > threshold), 1, 0
    )
    return data

def identify_leading_stock(all_stocks_data: dict, target_stock: str):
    """Cross-correlation analysis for market leadership detection"""
    leads = {}
    target_prices = all_stocks_data[target_stock]['mid_price']
    
    for stock, data in all_stocks_data.items():
        if stock == target_stock:
            continue
        correlation = correlate(target_prices, data['mid_price'], mode='full')
        max_lag = np.argmax(correlation) - len(target_prices) + 1
        leads[stock] = max_lag
    
    return max(leads, key=lambda x: abs(leads[x]))

# Model-related functions
def prepare_train_data(all_stocks_data: dict):
    """Prepare aligned multivariate time series data"""
    # Resample all stocks to common frequency (1 second)
    resampled = {}
    for stock, df in all_stocks_data.items():
        # Handle duplicates by taking last value
        deduped = df[~df.index.duplicated(keep='last')]
        # Resample to 1S frequency with forward fill
        resampled[stock] = deduped['mid_price'].resample('1S').last().ffill()
    
    # Combine into single dataframe
    combined = pd.concat(resampled.values(), axis=1, keys=resampled.keys())
    combined.ffill(inplace=True)
    combined.bfill(inplace=True)
    
    return combined

# Streamlit app layout
st.set_page_config(layout="wide")
st.title("Advanced Stock Market Analysis Toolkit")

# Sidebar controls
with st.sidebar:
    st.header("Analysis Parameters")
    selected_stock = st.selectbox("Select Stock", ['A', 'B', 'C', 'D', 'E'])
    selected_period = st.slider("Select Period", 1, 15, 1)
    plot_type = st.selectbox("Chart Type", [
        'Candlestick', 'Volatility', 'Comparative', 
        'Standard Deviation', 'Market Depth'
    ])
    
# Data loading and preprocessing
@st.cache_data
def load_all_data(period: int):
    stocks = ['A', 'B', 'C', 'D', 'E']
    return {
        stock: preprocess_data(read_market_data(stock, period))
        for stock in stocks
    }

all_stocks_data = load_all_data(selected_period)
processed_data = all_stocks_data[selected_stock]
processed_data = detect_volatility_events(processed_data)

# Main visualization section
st.header("Market Data Visualization")
col1, col2 = st.columns([3, 1])

with col1:
    if plot_type == 'Candlestick':
        fig = plot_candlestick(selected_stock, selected_period)
    elif plot_type == 'Standard Deviation':
        fig = plot_std_dev(selected_stock, selected_period)
    elif plot_type == 'Comparative':
        fig = plot_group_price_time_series_scatter(
            ['A', 'B', 'C', 'D', 'E'], selected_period, ['bid', 'ask']
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=processed_data.index, 
            y=processed_data['mid_price'], 
            name='Mid Price'
        ))
        
    # Highlight volatility events
    volatility_periods = processed_data[processed_data['volatility_event'] == 1]
    for idx, row in volatility_periods.iterrows():
        fig.add_vrect(
            x0=idx - pd.Timedelta(seconds=30), 
            x1=idx + pd.Timedelta(seconds=30),
            fillcolor="red", 
            opacity=0.2,
            annotation_text="Volatility Event" if idx == volatility_periods.index[0] else None
        )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("Market Insights")
    leading_stock = identify_leading_stock(all_stocks_data, selected_stock)
    st.metric("Market Leader", f"{leading_stock}")
    
    st.subheader("Volatility Statistics")
    st.metric("30s Std Dev", f"{processed_data['30s_std'].mean():.2f}")
    st.metric("60s Std Dev", f"{processed_data['60s_std'].mean():.2f}")
    
    st.subheader("Price Summary")
    st.metric("Current Mid Price", f"{processed_data['mid_price'].iloc[-1]:.2f}")
    st.metric("Daily Range", 
        f"{processed_data['mid_price'].min():.2f} - {processed_data['mid_price'].max():.2f}")

# Model training and prediction section
st.header("Predictive Analytics")
# Updated Streamlit training section
if st.button("Train Prediction Model"):
    with st.spinner("Training model..."):
        train_data = prepare_train_data(all_stocks_data)
        model = train_model(train_data, selected_stock)
        
        # Generate predictions
        test_window = train_data[selected_stock].values[-70:-10]
        predictions = model(torch.tensor(test_window, dtype=torch.float32).unsqueeze(0))
        
        # Display results
        st.line_chart(pd.DataFrame({
            "Actual": train_data[selected_stock].values[-60:],
            "Predicted": predictions.last_hidden_state.detach().numpy().flatten()[:60]
        }))
        
        # Generate trading signals
        signals = np.where(predictions > 0.5, "BUY", "SELL")
        st.subheader("Trading Signals")
        st.write(pd.Series(signals).value_counts())
        
        # P&L Calculation
        initial_capital = 1_000_000
        positions = initial_capital // train_data[selected_stock].values[-100:]
        pnl = positions * (train_data[selected_stock].values[-1] - train_data[selected_stock].values[-100])
        st.metric("Projected P&L", f"${pnl:,.2f}")

# Help section
with st.expander("User Guide"):
    st.markdown("""
    **Toolkit Features:**
    - Select different stocks and periods using sidebar controls
    - Choose from multiple visualization types
    - Real-time volatility detection and market leadership analysis
    - Transformer-based predictive modeling
    - Automated trading strategy backtesting
    """)