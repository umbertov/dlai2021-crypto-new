import mplfinance as mpf
import pandas as pd
from matplotlib import pyplot as plt

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--symbol", default="BTC-PERP")
parser.add_argument("--show-last", default=200)
args = parser.parse_args()
symbol = args.symbol

prices = pd.read_csv(f".tmp.{symbol}.csv")
prices.index = pd.to_datetime(prices.Date, utc=True)
preds = pd.read_csv(f"preds.{symbol}.txt", sep=";")
preds.index = pd.to_datetime(preds.Time, utc=True)
preds = preds.resample("5min").agg("first").drop("Time", axis="columns")
preds = preds.iloc[-min(args.show_last, len(preds)) :]
prices = prices.iloc[-len(preds) :]
ap = [
    # mpf.make_addplot(preds["Neutral"], color="gray", panel=1),
    mpf.make_addplot(preds["Buy"], color="green", panel=1, secondary_y=False),
    mpf.make_addplot(preds["Sell"], color="red", panel=1, secondary_y=False),
    mpf.make_addplot(preds["Neutral"], color="grey", panel=1, secondary_y=False),
]
mpf.plot(prices, type="candle", addplot=ap, title=symbol)
# preds.plot(ax=ax2)
plt.show()
