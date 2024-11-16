from pathlib import Path
import pandas as pd
import numpy as np
import typer
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

app = typer.Typer()

def generate_features(data):
    # moving averages
    data['SMA_50'] = data["Close"].rolling(window=50).mean()
    data['SMA_200'] = data["Close"].rolling(window=200).mean()
    data['price_change'] = data['Close'].diff()
    data['price_volatility'] = data['Close'].rolling(window=30).std()

    # economic indicators placeholders
    inflation_data = pd.DataFrame({'CPI': [2.5] * len(data)}, index=data.index) 
    gdp_data = pd.DataFrame({'growth_rate': [2.0] * len(data)}, index=data.index)
    unemployment_data = pd.DataFrame({'rate': [5.0] * len(data)}, index=data.index)
    data['inflation_rate'] = inflation_data["CPI"]
    data['gdp_growth'] = gdp_data["growth_rate"]
    data['unemployment_rate'] = unemployment_data["rate"]

    # relative strength index (rsi)
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # exponential moving averages (ema)
    data['EMA_50'] = data["Close"].ewm(span=50, adjust=False).mean()

    # lag features
    data["lagged_price"] = data["Close"].shift(1)
    data["lagged_yield"] = data["Close"].shift(1)

    return data

def scale_features(data, features):
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    return data