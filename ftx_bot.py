import ccxt
from getpass import getpass
import pandas as pd
from pathlib import Path
import torch
import hydra
from datetime import datetime
from pprint import pprint
import time

info = hydra.utils.logging.info

from src.dataset_readers import goodfeatures_reader
from src.evaluation.common import get_cfg_model, PROJECT, RUN_ID, ENTITY
from src.dataset import from_pandas
from src.ui.ui_utils import get_run_dir

# We don't ever need gradient descent here
torch.set_grad_enabled(False)

SIGNALS = {
    "sell": 0,
    "neutral": 1,
    "buy": 2,
}

CLOSE_ON_SIGNAL = True


def process_ccxt_ohlcv(ccxt_data: list[list]):
    df = pd.DataFrame(
        ccxt_data, columns=["Date", "Open", "High", "Low", "Close", "Volume"]
    )
    df.index = pd.to_datetime(df.Date, unit="ms", utc=True)
    df.drop(columns="Date", inplace=True)
    return df


def is_time_to_close(position, max_candles, timeframe):
    timeframe = pd.to_timedelta(timeframe)
    delta = datetime.now() - pd.to_datetime(position["datetime"])
    return delta > (max_candles * timeframe)


def close_position(exchange, symbol):
    positions = exchange.fetch_positions()
    positions_by_symbol = exchange.index_by(positions, "symbol")
    position = exchange.safe_value(positions_by_symbol, symbol)

    if position is None:
        info(f"No {symbol} open position")

    type = "market"
    side = "sell" if position["side"] == "long" else "buy"
    amount = position["contracts"]
    price = None
    params = {"reduceOnly": True}
    info(f"closing position for {symbol}")
    order = exchange.create_order(symbol, type, side, amount, price, params)
    pprint(order)
    return order


def go_long(exchange, symbol, amount, price=None):
    type = "market" if price is None else "limit"
    side = "buy"
    params = {}
    info(f"opening LONG position for {symbol} @ {price:.2f}, amount = {amount:.8f}")
    order = exchange.create_order(symbol, type, side, amount, price, params)
    info("done")
    pprint(order)
    return order


def go_short(exchange, symbol, amount, price=None):
    type = "market" if price is None else "limit"
    side = "sell"
    params = {}
    info(f"opening SHORT position for {symbol} @ {price:.2f}, amount = {amount:.8f}")
    order = exchange.create_order(symbol, type, side, amount, price, params)
    info("done")
    pprint(order)
    return order


def loop(exchange):
    info("fetching new OHLCV data")
    ohlcv = process_ccxt_ohlcv(exchange.fetchOHLCV("BTC-PERP", "5m"))
    info("done fetching new OHLCV data")
    ohlcv.to_csv(".tmpcsv")

    incols = cfg.dataset_conf.input_columns

    info("building feature dataframe")
    nn_in_df = goodfeatures_reader(
        trend_period=20, alpha=0.015, zscore_periods=[10, 30, 50, 100, 200]
    )(".tmpcsv")[incols].iloc[-64:]

    nn_in = (
        from_pandas(nn_in_df).unsqueeze(0).transpose(-1, -2).float().to(model.device)
    )
    info("computing model output")
    model_out = model(nn_in)

    probabs = torch.softmax(model_out["classification_logits"], -1)[0, -1]
    info(f"predicted probabilities: {probabs.cpu()}")

    next_prediction = probabs.argmax()

    info("canceling all pending orders")
    exchange.cancelAllOrders()

    positions = exchange.fetch_positions()
    positions_by_symbol = exchange.index_by(positions, "symbol")
    position = exchange.safe_value(positions_by_symbol, "BTC-PERP")

    if position is None:
        info(f"No BTC-PERP open position")
    is_long = position is not None and position["side"] == "long"
    is_short = position is not None and position["side"] != "long"
    is_flat = (not is_short) and (not is_long)

    time_to_close = position is not None and is_time_to_close(
        position,
        max_candles=cfg.dataset_conf.dataset_reader.trend_period,
        timeframe=cfg.dataset_conf.dataset_reader.resample[:2],
    )

    last_closing_price = ohlcv.Close[-1]

    if time_to_close:
        info("closing position due to time target reached")
        close_position(exchange, symbol="BTC-PERP")
        return

    if next_prediction == SIGNALS["buy"]:
        if is_flat:
            usd_balance = exchange.fetchBalance()["USD"]["free"]
            price = last_closing_price * 0.9998
            amount = max(0.0001, (usd_balance / 2) / price)
            go_long(exchange, "BTC-PERP", amount=amount, price=price)
        elif is_short and CLOSE_ON_SIGNAL:
            close_position(exchange, symbol="BTC-PERP")
    elif next_prediction == SIGNALS["sell"]:
        if is_flat:
            usd_balance = exchange.fetchBalance()["USD"]["free"]
            price = last_closing_price * 1.0001
            amount = max(0.0001, (usd_balance / 2) / price)
            go_short(exchange, "BTC-PERP", amount=amount, price=price)
        elif is_long and CLOSE_ON_SIGNAL:
            close_position(exchange, symbol="BTC-PERP")
    else:
        info("predicted neutral class")


if __name__ == "__main__":
    RUN_ID = "2k2hrsos"

    ##### LOAD HYDRA CFG, MODEL CHECKPOINT FROM WANDB RUN
    run_dir: Path = get_run_dir(project=PROJECT, entity=ENTITY, run_id=RUN_ID)
    checkpoint_paths: list[Path] = list(run_dir.rglob("checkpoints/*"))
    cfg, model = get_cfg_model(checkpoint_path=checkpoint_paths[0], run_dir=run_dir)
    dataset_reader = hydra.utils.instantiate(cfg.dataset_conf.dataset_reader)

    EXCHANGE = "ftx"
    private_credentials = (
        {
            "apiKey": getpass("Insert API Key: "),
            "secret": getpass("Insert API Secret: "),
            "headers": {
                "FTX-SUBACCOUNT": "Bot",
            },
        }
        if "y" in input("Authenticate to exchange? (y/N) ")
        else {}
    )
    exchange = getattr(ccxt, EXCHANGE)({"enableRateLimit": True, **private_credentials})
    exchange.checkRequiredCredentials()

    while True:
        loop(exchange)
        minutesToSleep = 5 - datetime.now().minute % 5
        info(f"going to sleep for {minutesToSleep} minutes")
        time.sleep(minutesToSleep * 60)
