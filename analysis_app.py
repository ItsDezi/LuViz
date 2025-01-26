import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from luviz.parser import read_market_data, read_trade_data
from luviz.plotter import plot_candlestick, plot_std_dev, plot_group_price_time_series_scatter
from scipy.signal import correlate
# from transformers import TimeSeriesTransformerForPrediction, Trainer, TrainingArguments
# Replace the Hugging Face model with a simpler PyTorch model
import torch
import torch.nn as nn

class StockPredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_model(train_data):
    model = StockPredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Convert data to tensors
    X = torch.tensor(train_data.values[:-1], dtype=torch.float32).unsqueeze(0)
    y = torch.tensor(train_data.values[1:], dtype=torch.float32).unsqueeze(0)
    
    # Training loop
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
    return model

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

# def train_model(train_data, val_data):
#     """Train Hugging Face time series transformer"""
#     model = TimeSeriesTransformerForPrediction.from_pretrained("ibm-automl/ts-transformer")
    
#     training_args = TrainingArguments(
#         output_dir='./results',
#         learning_rate=1e-4,
#         num_train_epochs=10,
#         per_device_train_batch_size=32,
#     )
    
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_data,
#         eval_dataset=val_data,
#     )
#     trainer.train()
#     return model

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
if st.button("Train Prediction Model"):
    with st.spinner("Training model..."):
        train_data = prepare_train_data(all_stocks_data)
        model = train_model(train_data)
        
        # Generate predictions
        with torch.no_grad():
            test_input = torch.tensor(train_data.values[-100:], dtype=torch.float32).unsqueeze(0)
            predictions = model(test_input).numpy().flatten()
            
        st.line_chart(pd.DataFrame({
            "Actual": train_data[selected_stock].values[-100:],
            "Predicted": predictions
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