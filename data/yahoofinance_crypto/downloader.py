"""
Script which downloads, from Yahoo! Finance, 5-minutely data
for the last 60 days of the selected assets
"""

import os
import yfinance as yf
from random import random
from time import sleep
from sys import exit  # or iPython complains
import pandas as pd

from tqdm.contrib.concurrent import thread_map

from typing import Optional

# All tickers are as represented on the Yahoo! Finance website
CRYPTOS = (
    "BTC",
    "LTC",
    "ETH",
    "ADA",
    "SOL1",
    "DOT1",
)
FIATS = ("USD", "EUR")

# flat list of all crypto/fiat pairs
PAIRS = [f"{crypto}-{fiat}" for crypto in CRYPTOS for fiat in FIATS]


def download_ticker(ticker_name: str, jitter: bool = False):
    # When concurrently downloading lots of data, jitter avoids rate limits
    if jitter:
        sleep(min(0.1, 1.2 * random()))

    ticker = yf.Ticker(ticker_name)
    # best we can do is 5m data for last 60 days.
    history = ticker.history(period="60d", interval="5m")
    # cryptos don't have dividends and stock splits
    history = history.drop(["Dividends", "Stock Splits"], axis="columns")

    save_dataframe(history, ticker_name)

    return history


def save_dataframe(df: "pandas.DataFrame", name):
    path = f"{name}.{fmt_start_end(df)}.csv"
    df.to_csv(path)


def map_exec(f, elements, **kwargs):
    return list(map(f, elements))


def fmt_timestamp(timestamp) -> str:
    return str(timestamp)[:10]


def fmt_start_end(df: "pandas.DataFrame") -> str:
    start = fmt_timestamp(df.index[0])
    end = fmt_timestamp(df.index[-1])
    return f"{start}.{end}"


def concat_dedupe(df1, df2):
    return pd.concat([df1, df2[~df2.index.isin(df1.index)]], verify_integrity=True)


if __name__ == "__main__":
    # useful info
    print("I will download:", *PAIRS, sep="\n - ")
    # confirmation
    if not input("continue? [y/N] ") == "y":
        print("confirmation failed. aborting.")
        exit(0)

    # concurrent download of all pairs
    dataframes = thread_map(
        lambda ticker: download_ticker(ticker, jitter=True),
        PAIRS,
        max_workers=4,
    )

    # concurrent serialization of all downloaded dataframes
    # thread_map(
    #    lambda item: save_dataframe(item[0], f"{item[1]}.csv"),
    #    list(zip(dataframes, PAIRS)),
    #    max_workers=4,
    # )
