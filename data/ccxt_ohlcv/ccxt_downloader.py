import ccxt
from collections import namedtuple
import time
from pathlib import Path
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm


LIMITS = {
    "ftx": 5000,
    "binance": 1000,
}

START_DATE, END_DATE = datetime(2019, 7, 22), datetime(2021, 11, 23)
TIMEFRAME = "5m"


def timedeltas(start_date, end_date, timeframe="5m", n=5000):
    """
    Args:
        - start_date (str|datetime): starting date and time
        - end_date (str|datetime): ending date and time
        - interval (str|timedelta): the resolution of the data to get
        - n (int): the length, in `interval`s, of the timedeltas to get
    Returns: list of datetime couples (start_i, end_i)
    """
    delta = n * pd.to_timedelta(timeframe)
    out = []
    date = start_date
    while date < end_date:
        start = date
        end = start + delta
        out.append((start, end))
        date += delta
    return out


def process_ccxt_ohlcv(ccxt_data: list[list]):
    df = pd.DataFrame(
        ccxt_data, columns=["Date", "Open", "High", "Low", "Close", "Volume"]
    )
    df.index = pd.to_datetime(df.Date, unit="ms", utc=True)
    df.drop(columns="Date", inplace=True)
    return df


def get_fetch_ohlcv_params(exchange, start_date, end_date):
    if isinstance(exchange, ccxt.ftx):
        return {
            "start_time": datetime_to_seconds(start_date),
            "end_time": datetime_to_seconds(end_date),
        }
    raise NotImplementedError


def fetch_multi_ohlcv(
    exchange, symbol, start_date, end_date, timeframe="5m", limit=5000
):
    deltas = timedeltas(start_date, end_date, timeframe, limit)
    dataframes = []
    for start, end in tqdm(deltas, total=len(deltas)):
        ohlcv = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=limit,
            since=datetime_to_ms(start),
            # params=get_fetch_ohlcv_params(exchange, start, end),
        )
        df = process_ccxt_ohlcv(ohlcv)
        dataframes.append(df)
    return dataframes


def datetime_to_seconds(dt: datetime):
    return int(time.mktime(dt.timetuple()))


def datetime_to_ms(dt: datetime):
    return int(time.mktime(dt.timetuple())) * 1000


def join_dataframes(ohlcv_dataframes: list[pd.DataFrame]):
    df = pd.concat(ohlcv_dataframes)
    # deduplicating the index
    df = df[~df.index.duplicated(keep="first")]
    return df


def csv_filename(symbol, start_date, end_date, exchange_name):
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")
    if "/" in symbol:
        symbol = "-".join(symbol.split("/"))
    return Path(f"{symbol}.{exchange_name}.{start_date}.{end_date}.csv")


SymbolData = namedtuple("SymbolData", ["name", "dataframe", "path", "exchange"])


def download_symbol(
    exchange,
    symbol,
    start_date,
    end_date,
    timeframe="5m",
    destdir=".",
    limit=5000,
    save=False,
) -> SymbolData:
    exchange_name = type(exchange).__name__

    ohlcv_dataframes = fetch_multi_ohlcv(
        exchange, symbol, start_date, end_date, timeframe, limit=limit
    )
    df = join_dataframes(ohlcv_dataframes)

    csv_path = destdir / csv_filename(symbol, start_date, end_date, exchange_name)

    if save:
        df.to_csv(str(csv_path))
    return SymbolData(name=symbol, dataframe=df, path=csv_path, exchange=exchange_name)


def download_symbols(symbols: list[str], **kwargs) -> list[SymbolData]:
    save = kwargs.pop("save", False)
    symbols_data = [
        download_symbol(symbol=symbol, save=False, **kwargs) for symbol in symbols
    ]
    if save:
        for symbol_data in symbols_data:
            print(f"saving {symbol_data.name} to {symbol_data.path}...")
            symbol_data.dataframe.to_csv(str(symbol_data.path))
    return symbols_data


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--symbol", default="BTC-PERP")
    parser.add_argument("--destdir", default=".", type=str)
    parser.add_argument("--exchange", default="ftx", type=str)
    args = parser.parse_args()

    SYMBOL = args.symbol
    DESTDIR = Path(args.destdir)
    assert DESTDIR.exists() and DESTDIR.is_dir()

    exchange = getattr(ccxt, args.exchange)({"enableRateLimit": True})
    symbol_data = download_symbol(
        exchange,
        SYMBOL,
        START_DATE,
        END_DATE,
        TIMEFRAME,
        DESTDIR,
        limit=LIMITS[args.exchange],
        save=True,
    )
