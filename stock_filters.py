import pandas as pd
import numpy as np
import argparse

def get_stock_data(filename):
    df = pd.read_csv(filename, parse_dates=['Date'], index_col='Date', date_format='%d-%b-%y')
    print("Data read from CSV file:")
    print(df.head())
    return df

def calculate_indicators(df):
    # Check if necessary columns are present
    required_columns = ['Close', 'Volume', 'High']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in data")

    # Simple Moving Average of Volume over 20 days
    df['SMA20'] = df['Volume'].rolling(window=20).mean()

    # Exponential Moving Averages
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA13'] = df['Close'].ewm(span=13, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # Maximum High over the last 222 days
    df['Max222High'] = df['High'].rolling(window=222).max()

    return df

def apply_filters(df):
    filters = (
        (df['Close'] >= 50) &
        (df['Close'] >= df['Close'].shift(1) * 1.01) &
        (df['Volume'] > df['SMA20'] * 1.1) & 
        (df['High'] == df['Max222High'].fillna(0)) &  
        (df['Close'].shift(1) > df['Close'].shift(2) * 0.98) &
        (df['EMA13'] > df['EMA50']) &
        (df['EMA50'] > df['EMA200']) &
        (df['Close'] > df['EMA9'])
    )
    filtered_df = df[filters]
    print("Filtered DataFrame:")
    print(filtered_df)
    return filtered_df

def filter_stocks_from_csv(filename):
    df = get_stock_data(filename)
    df = calculate_indicators(df)
    filtered_df = apply_filters(df)

    return filtered_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter stocks from a CSV file.')
    parser.add_argument('csv_file', type=str, help='The path to the CSV file containing stock data.')
    args = parser.parse_args()

    filtered_stocks = filter_stocks_from_csv(args.csv_file)
    print("Stocks that pass all filters:", args.csv_file if not filtered_stocks.empty else "None")
