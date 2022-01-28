import mplfinance as mpf
import pandas as pd
from matplotlib import pyplot as plt

prices = pd.read_csv(".tmpcsv")
prices.index = pd.to_datetime(prices.Date, utc=True)
preds = pd.read_csv("preds.txt", sep=";")
preds.index = pd.to_datetime(preds.Time, utc=True)
preds = preds.resample("5min").agg("first").drop("Time", axis="columns")
ap = [
    # mpf.make_addplot(preds["Neutral"], color="gray", panel=1),
    mpf.make_addplot(preds["Buy"], color="green", panel=1, secondary_y=False),
    mpf.make_addplot(preds["Sell"], color="red", panel=1, secondary_y=False),
]
prices = prices.iloc[-len(preds) :]
mpf.plot(prices, type="candle", addplot=ap)
# preds.plot(ax=ax2)
plt.show()
