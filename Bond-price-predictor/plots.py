from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from bond_price_predictor.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def plot_bond_prices(data, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label=f'{bond} Price')
    plt.plot(data.index, data['SMA_50'], label='50-day SMA')
    plt.plot(data.index, data['SMA_200'], label='200-day SMA')
    plt.title(f'{bond} Price with 50-day and 200-day SMAs')
    plt.legend()
    plt.show()