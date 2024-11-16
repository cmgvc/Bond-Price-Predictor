from pathlib import Path
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import typer
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from bond_price_predictor.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from bond_price_predictor.features import generate_features, scale_features
from bond_price_predictor.plots import plot_bond_prices

app = typer.Typer()

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    logger.info("Processing dataset...")

    bonds = ['^TNX', 'AAPl', 'MSFT', 'EMB', 'F']
    data_dict = {}
    
    for bond in bonds:
        data_dict[bond] = yf.download(bond, start="2010-01-01", end="2024-01-01", progress=False)

    for bond, data in data_dict.items():
        data = generate_features(data)

        sp500 = yf.download("^GSPC", start="2010-01-01", end="2024-01-01", progress=False)["Close"]
        vix = yf.download("^VIX", start="2010-01-01", end="2024-01-01", progress=False)["Close"]
        data["spt500"] = sp500
        data["vix"] = vix

        data.dropna(inplace=True)
        scaler = StandardScaler()
        features = [
            "Close",
            "SMA_50",
            "SMA_200",
            "price_change",
            "price_volatility",
            "inflation_rate",
            "gdp_growth",
            "unemployment_rate",
            "RSI",
            "lagged_price",
            "lagged_yield",
            "spt500",
            "vix",
        ]
        data = scale_features(data, features)
        data.to_csv(output_path, mode='a', header=not Path(output_path).exists(), index=False)

        logger.info(f"Processed data for {bond}:")
        logger.info(data.head())

        plot_bond_prices(data, bond)

        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Close'], label=f'{bond} Price')
        plt.plot(data.index, data['SMA_50'], label='50-day SMA')
        plt.plot(data.index, data['SMA_200'], label='200-day SMA')
        plt.title(f'{bond} Price with 50-day and 200-day SMAs')
        plt.legend()
        plt.show()

    logger.success("Dataset processing complete.")

if __name__ == "__main__":
    app()
