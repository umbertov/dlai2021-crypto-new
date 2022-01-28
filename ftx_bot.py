from dataclasses import dataclass
from typing import Callable, Optional
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
from src.common.utils import get_env, load_envs

# We don't ever need gradient descent here
torch.set_grad_enabled(False)

SIGNALS = {
    "sell": 0,
    "neutral": 1,
    "buy": 2,
}

CLOSE_ON_SIGNAL = True


@dataclass
class LoopState:
    trade_start = None
    side = None
    dataset_reader: Callable

    def trade_started(self, side):
        assert side in ("long", "short")
        self.trade_start = datetime.now().replace(second=0, microsecond=0)
        self.side = side

    def trade_ended(self):
        self.trade_start = None
        self.side = None


def process_ccxt_ohlcv(ccxt_data: list[list]):
    df = pd.DataFrame(
        ccxt_data, columns=["Date", "Open", "High", "Low", "Close", "Volume"]
    )
    df.index = pd.to_datetime(df.Date, unit="ms", utc=True)
    df.drop(columns="Date", inplace=True)
    return df


def is_time_to_close(trade_start, max_candles, timeframe):
    if trade_start is None:
        info(
            "trade start is none in input to is_time_to_close; silently returning false"
        )
        return False
    timeframe = pd.to_timedelta(timeframe)
    delta = datetime.now() - trade_start
    return delta > (max_candles * timeframe)


def close_position(exchange, symbol, position):
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


def predict_on_df(model, features_dataframe, cfg, past_context=64):
    assert not past_context > len(features_dataframe)
    feature_columns = cfg.dataset_conf.input_columns
    info("building feature dataframe")
    features_dataframe = features_dataframe[feature_columns].iloc[-past_context:]

    feature_tensor = (
        from_pandas(features_dataframe)
        .unsqueeze(0)
        .transpose(-1, -2)
        .float()
        .to(model.device)
    )
    info("computing model output")
    model_out = model(feature_tensor)

    probabs = torch.softmax(model_out["classification_logits"], -1)[0, -1]
    info(f"predicted probabilities: {probabs.cpu()}")
    return probabs


def loop(exchange, loop_state: LoopState, symbol):
    info("fetching new OHLCV data")
    ohlcv = process_ccxt_ohlcv(exchange.fetchOHLCV(symbol, "5m"))
    info("done fetching new OHLCV data")
    ohlcv.to_csv(".tmpcsv")

    # incols = cfg.dataset_conf.input_columns

    # info("building feature dataframe")
    # nn_in_df = goodfeatures_reader(
    #     trend_period=20, alpha=0.015, zscore_periods=[10, 30, 50, 100, 200]
    # )(".tmpcsv")[incols].iloc[-64:]

    # nn_in = (
    #     from_pandas(nn_in_df).unsqueeze(0).transpose(-1, -2).float().to(model.device)
    # )
    # info("computing model output")
    # model_out = model(nn_in)

    # probabs = torch.softmax(model_out["classification_logits"], -1)[0, -1]
    # info(f"predicted probabilities: {probabs.cpu()}")
    features_dataframe = loop_state.dataset_reader(".tmpcsv")
    probabs = predict_on_df(model, features_dataframe, cfg)
    with open("preds.txt", "a") as f:
        f.write(
            f'{datetime.utcnow()};{";".join(map(str,probabs.squeeze().tolist()))}\n'
        )

    next_prediction = probabs.argmax()
    info(f"predicted class index: {next_prediction}")

    info("canceling all pending orders")
    exchange.cancelAllOrders()

    info("fetching positions")
    positions = exchange.fetch_positions()
    positions_by_symbol = exchange.index_by(positions, "symbol")
    position = exchange.safe_value(positions_by_symbol, symbol)

    if position is None:
        info(f"No {symbol} open position")
    elif position["timestamp"] is None:
        info("position is not none but position['timestamp'] is.")
        position = None
        info("position was set to none")

    position_size = 0 if position is None else position["info"]["size"]

    is_long = position is not None and position["side"] == "long" and position_size != 0
    is_short = (
        position is not None and position["side"] != "long" and position_size != 0
    )
    is_flat = (not is_short) and (not is_long)

    time_to_close = position_size > 0 and is_time_to_close(
        loop_state.trade_start,
        max_candles=cfg.dataset_conf.dataset_reader.trend_period,
        timeframe=cfg.dataset_conf.dataset_reader.resample[:2],
    )

    last_closing_price = ohlcv.Close[-1]

    if time_to_close:
        info("closing position due to time target reached")
        close_position(exchange, symbol=symbol, position=position)
        info("closed")
        loop_state.trade_ended()
        return loop_state

    if next_prediction == SIGNALS["buy"]:
        if is_flat:
            info("buy signal and flat position. entering long")
            usd_balance = exchange.fetchBalance()["USD"]["free"]
            price = last_closing_price * 0.9998
            amount = max(0.0001, (usd_balance / 2) / price)
            order = go_long(exchange, symbol, amount=amount, price=price)
            loop_state.trade_started("long")
        elif is_short and CLOSE_ON_SIGNAL:
            info("buy signal and short position. closing long")
            close_position(exchange, symbol=symbol, position=position)
            loop_state.trade_ended()
    elif next_prediction == SIGNALS["sell"]:
        if is_flat:
            info("sell signal and flat position. entering short")
            usd_balance = exchange.fetchBalance()["USD"]["free"]
            price = last_closing_price * 1.0001
            amount = max(0.0001, (usd_balance / 2) / price)
            order = go_short(exchange, symbol, amount=amount, price=price)
            loop_state.trade_started("short")
        elif is_long and CLOSE_ON_SIGNAL:
            info("sell signal and long position. closing short")
            close_position(exchange, symbol=symbol, position=position)
            loop_state.trade_ended()
    else:
        info("predicted neutral class")

    return loop_state


def synchronize_clocks():
    minutesToSleep = 4 - datetime.now().minute % 5
    secondsToSleep = 60 * minutesToSleep
    secondsToSleep += 60 - datetime.now().second % 60
    secondsToSleep += 2
    info(
        f"going to sleep for {secondsToSleep // 60} minutes, {secondsToSleep % 60} seconds"
    )
    time.sleep(secondsToSleep)


def get_exchange(exchange_name):
    private_credentials = {
        "apiKey": get_env("FTX_API_KEY"),  # getpass("Insert API Key: "),
        "secret": get_env("FTX_API_SECRET"),  # getpass("Insert API Secret: "),
        "headers": {
            "FTX-SUBACCOUNT": "Bot",
        },
    }
    import os

    del os.environ["FTX_API_KEY"]
    del os.environ["FTX_API_SECRET"]

    exchange = getattr(ccxt, exchange_name)(
        {"enableRateLimit": True, **private_credentials}
    )
    exchange.checkRequiredCredentials()
    return exchange


if __name__ == "__main__":
    RUN_ID = "2k2hrsos"
    EXCHANGE = "ftx"
    load_envs(".secrets")

    ##### LOAD HYDRA CFG, MODEL CHECKPOINT FROM WANDB RUN
    run_dir: Path = get_run_dir(project=PROJECT, entity=ENTITY, run_id=RUN_ID)
    checkpoint_paths: list[Path] = list(run_dir.rglob("checkpoints/*"))
    cfg, model = get_cfg_model(checkpoint_path=checkpoint_paths[0], run_dir=run_dir)
    dataset_reader = hydra.utils.instantiate(cfg.dataset_conf.dataset_reader)

    exchange = get_exchange(EXCHANGE)

    loop_state = LoopState(dataset_reader=dataset_reader)

    while True:
        synchronize_clocks()
        loop_state = loop(exchange, loop_state=loop_state, symbol="BTC-PERP")
        info("loop exited.")
        info(f"loop state: {loop_state}")
