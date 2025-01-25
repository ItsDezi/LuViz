"""
This module provides utility functions to read stock market and trade data from CSV files.
Functions:
    read_market_data(stock: str, period: int, parsing_header: str = 'timestamp') -> pd.DataFrame:
        Reads market data CSV files for a given stock and period, concatenates them, and returns a DataFrame.
        Args:
            stock (str): The stock identifier, must be one of ['A', 'B', 'C', 'D', 'E'].
            period (int): The period identifier, must be in the range 2 to 15.
            parsing_header (str): The column name to parse as datetime, default is 'timestamp'.
        Returns:
            pd.DataFrame: A DataFrame containing the concatenated market data.
        Raises:
            ValueError: If the stock or period is not valid.
    read_trade_data(stock: str, period: int, parsing_header: str = 'timestamp') -> pd.DataFrame:
        Reads trade data CSV file for a given stock and period and returns a DataFrame.
        Args:
            stock (str): The stock identifier, must be one of ['A', 'B', 'C', 'D', 'E'].
            period (int): The period identifier, must be in the range 2 to 15.
            parsing_header (str): The column name to parse as datetime, default is 'timestamp'.
        Returns:
            pd.DataFrame: A DataFrame containing the trade data.
        Raises:
            ValueError: If the stock or period is not valid.
"""

import os
import pandas as pd

# Function to read CSV and plot two columns
def read_market_data(stock:str, period: int, parsing_header:str ='timestamp') -> pd.DataFrame:
    
    if stock not in ['A', 'B', 'C', 'D', 'E']:
        raise ValueError("Unknown stock error")
    if period not in range(1, 16):
        raise ValueError("Unknown period error")
    
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../data/TrainingData/Period{period}/{stock}')

    # Read the first CSV file and parse the timestamp column as datetime
    first_csv = f'market_data_{stock}_0.csv'  

    other_csvs = [file for file in os.listdir(folder_path) if file.startswith('market_data') and not file.endswith('0.csv')]

    data = pd.read_csv(folder_path + '/' + first_csv, parse_dates=[parsing_header])
    
    columns = data.columns
    
    # Rename the columns of other DataFrames to match the first
    for file in other_csvs:
        curr_data = pd.read_csv(folder_path + '/' + file, parse_dates=[parsing_header], header=None, names=columns)
        curr_data.columns = columns
        data = pd.concat([data, curr_data], ignore_index=True)
    
    return data


def read_trade_data(stock: str, period: int, parsing_header: str = 'timestamp') -> pd.DataFrame:
    
    if stock not in ['A', 'B', 'C', 'D', 'E']:
        raise ValueError("Unknown stock error")
    if period not in range(1, 16):
        raise ValueError("Unknown period error")
    
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../data/TrainingData/Period{period}/{stock}')

    # Read the trade csv file and parse the timestamp column as datetime
    trade_csv = f'trade_data__{stock}.csv'  

    data = pd.read_csv(folder_path + '/' + trade_csv, parse_dates=[parsing_header])
    
    return data