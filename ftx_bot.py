from dataclasses import dataclass, field
from random import randint
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
import logging

info = logging.info
warning = logging.warning
error = logging.error

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

MIN_AMOUNTS = {
    "BTC-PERP": 0.0001,
    "ETH-PERP": 0.001,
}

CLOSE_ON_SIGNAL = True
INVERT_ON_SIGNAL = False


@dataclass
class LoopState:
    trade_start = None
    side = None
    dataset_reader: Callable = field(repr=False)
    symbol: str
    min_amount: float
    prediction_threshold: Optional[float] = None

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
    price_str = f"{price:.2f}" if price else "market"
    info(f"opening LONG position for {symbol} @ {price_str}, amount = {amount:.8f}")
    order = exchange.create_order(symbol, type, side, amount, price, params)
    info("done")
    pprint(order)
    return order


def go_short(exchange, symbol, amount, price=None):
    type = "market" if price is None else "limit"
    side = "sell"
    params = {}
    price_str = f"{price:.2f}" if price else "market"
    info(f"opening SHORT position for {symbol} @ {price_str}, amount = {amount:.8f}")
    order = exchange.create_order(symbol, type, side, amount, price, params)
    info("done")
    pprint(order)
    return order


def set_take_profit_stop_loss(exchange, symbol, amount, entry_price, mode):
    assert mode in ("long", "short")
    side = "sell" if mode == "long" else "buy"
    tp_price_pct = (1 + 0.015) if mode == "long" else (1 - 0.015)
    sl_price_pct = (1 - 0.015) if mode == "long" else (1 + 0.015)
    tp_trigger_price = tp_price_pct * entry_price
    sl_trigger_price = sl_price_pct * entry_price
    info(
        f"setting take profit to {tp_trigger_price:.2f}$ and stop loss to {sl_trigger_price:.2f} for {mode.upper()} position on {symbol}"
    )
    try:
        info("sending take profit")
        take_profit_order = exchange.private_post_conditional_orders(
            {
                "market": symbol,
                "side": side,
                "triggerPrice": tp_trigger_price,
                "type": "takeProfit",
                "size": amount,
                "reduceOnly": True,
            }
        )
    except Exception as e:
        error(e, exc_info=True)
        take_profit_order = None
    try:
        info("sending stop loss")
        stop_loss_order = exchange.private_post_conditional_orders(
            {
                "market": symbol,
                "side": side,
                "triggerPrice": sl_trigger_price,
                "size": amount,
                "type": "stop",
                "reduceOnly": True,
            }
        )
    except Exception as e:
        error(e, exc_info=True)
        stop_loss_order = None
    return take_profit_order, stop_loss_order


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
    with open(f"preds.{symbol}.txt", "a") as f:
        f.write(
            f'{datetime.utcnow()};{";".join(map(str,probabs.squeeze().tolist()))}\n'
        )

    next_prediction = probabs.argmax()
    if loop_state.prediction_threshold is not None:
        # if predicted non-neutral, but probability is not above threshold, reset to neutral
        if probabs.max() < loop_state.prediction_threshold:
            next_prediction = 1
            # next_prediction = (
            #     0  # sell
            #     if next_prediction == 0 and probabs[0] > loop_state.prediction_threshold
            #     else 2  # buy
            #     if next_prediction == 2 and probabs[2] > loop_state.prediction_threshold
            #     else 1  # neutral
            # )
    return next_prediction, probabs


def get_symbol_position(exchange, symbol):
    positions = exchange.fetch_positions()
    positions_by_symbol = exchange.index_by(positions, "symbol")
    position = exchange.safe_value(positions_by_symbol, symbol)
    return position


def get_current_price(exchange, symbol):
    ticker = exchange.fetchTicker(symbol)
    bid, ask = float(ticker["bid"]), float(ticker["ask"])
    midprice = (bid + ask) / 2
    return midprice


def symbol_loop(exchange, loop_state: LoopState):
    symbol = loop_state.symbol
    info("fetching new OHLCV data")
    ohlcv = process_ccxt_ohlcv(exchange.fetchOHLCV(symbol, "5m"))
    info("done fetching new OHLCV data")
    ohlcv.to_csv(f".tmp.{symbol}.csv")

    features_dataframe = loop_state.dataset_reader(f".tmp.{symbol}.csv")
    next_prediction, probabs = predict_on_df(model, features_dataframe, cfg)
    info(f"predicted class index: {next_prediction}")

    # info("canceling all pending orders")
    # exchange.cancelAllOrders()

    info("fetching positions")
    position = get_symbol_position(exchange, symbol)
    current_price = get_current_price(exchange, symbol)
    info(f"current {symbol} price is {current_price:.2f}")

    position_size = 0 if position is None else float(position["info"]["size"])
    if position_size == 0:
        info(f"No {symbol} open position")
    else:
        position_entry_price = float(position["info"]["recentBreakEvenPrice"])
        calc_pnl = current_price / position_entry_price - 1
        info(
            f"{symbol} position open with size {position_size}, @{position_entry_price:.2f} "
            + f"recentPnl = {100*float(position['info']['recentPnl']):.2f}%, calculated pnl = {100*calc_pnl:.2f}%"
        )

    is_long = position_size != 0 and position["side"] == "long"
    is_short = position_size != 0 and position["side"] != "long"
    is_flat = (not is_short) and (not is_long)

    time_to_close = position_size > 0 and (
        is_time_to_close(
            loop_state.trade_start,
            max_candles=int(cfg.dataset_conf.dataset_reader.trend_period),
            timeframe=cfg.dataset_conf.dataset_reader.resample[:2],
        )
        # or (float(position["info"]["recentPnl"]) > 0.015)
        # or (float(position["info"]["recentPnl"]) < -0.015)
    )

    if time_to_close:
        info("time to close")
        close_position(exchange, symbol=symbol, position=position)
        info("closed")
        info("canceling all orders")
        exchange.cancelAllOrders()
        loop_state.trade_ended()
        return loop_state

    if datetime.now().minute % 5 != 0:
        return loop_state

    if next_prediction == SIGNALS["buy"]:
        if is_flat:
            info("buy signal and flat position. entering long")
            usd_balance = exchange.fetchBalance()["USD"]["free"]
            # market order
            price = None  # last_closing_price * 0.9998
            amount = max(loop_state.min_amount, (usd_balance / 2) / current_price)
            order = go_long(exchange, symbol, amount=amount, price=price)
            loop_state.trade_started("long")
            set_take_profit_stop_loss(exchange, symbol, amount, current_price, "long")
        elif is_short and CLOSE_ON_SIGNAL:
            info("buy signal and short position. closing long")
            close_position(exchange, symbol=symbol, position=position)
            loop_state.trade_ended()
            info("canceling all orders")
            exchange.cancelAllOrders()
            if INVERT_ON_SIGNAL:
                usd_balance = exchange.fetchBalance()["USD"]["free"]
                amount = max(loop_state.min_amount, (usd_balance / 2) / current_price)
                order = go_long(exchange, symbol, amount=amount, price=None)
                loop_state.trade_started("short")
                set_take_profit_stop_loss(
                    exchange, symbol, amount, current_price, "long"
                )
    elif next_prediction == SIGNALS["sell"]:
        if is_flat:
            info("sell signal and flat position. entering short")
            usd_balance = exchange.fetchBalance()["USD"]["free"]
            # market order
            price = None  # last_closing_price * 1.0001
            amount = max(loop_state.min_amount, (usd_balance / 2) / current_price)
            order = go_short(exchange, symbol, amount=amount, price=price)
            loop_state.trade_started("short")
            set_take_profit_stop_loss(exchange, symbol, amount, current_price, "short")
        elif is_long and CLOSE_ON_SIGNAL:
            info("sell signal and long position. closing short")
            close_position(exchange, symbol=symbol, position=position)
            info("canceling all orders")
            exchange.cancelAllOrders()
            loop_state.trade_ended()
            if INVERT_ON_SIGNAL:
                usd_balance = exchange.fetchBalance()["USD"]["free"]
                amount = max(loop_state.min_amount, (usd_balance / 2) / current_price)
                order = go_short(exchange, symbol, amount=amount, price=None)
                loop_state.trade_started("short")
                set_take_profit_stop_loss(
                    exchange, symbol, amount, current_price, "short"
                )

    else:
        info("predicted neutral class")

    return loop_state


def synchronize_clocks(minutes):
    minutesToSleep = (minutes - 1) - datetime.now().minute % (minutes)
    secondsToSleep = 60 * minutesToSleep
    secondsToSleep += 60 - datetime.now().second % 60
    secondsToSleep += randint(0, 50) / 10
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
    SYMBOLS = [
        "BTC-PERP",
    ]
    RUN_ID = "2k2hrsos"  # "1467oz8h"  # "1467oz8h"  # "2k2hrsos" #
    EXCHANGE = "ftx"
    PREDICTION_THRESHOLD = 0.5
    load_envs(".secrets")

    exchange = get_exchange(EXCHANGE)

    ##### LOAD HYDRA CFG, MODEL CHECKPOINT FROM WANDB RUN
    run_dir: Path = get_run_dir(project=PROJECT, entity=ENTITY, run_id=RUN_ID)
    checkpoint_paths: list[Path] = list(run_dir.rglob("checkpoints/*"))
    cfg, model = get_cfg_model(checkpoint_path=checkpoint_paths[0], run_dir=run_dir)
    model.eval()
    dataset_reader = hydra.utils.instantiate(cfg.dataset_conf.dataset_reader)

    loop_states = {
        symbol: LoopState(
            dataset_reader=dataset_reader,
            symbol=symbol,
            min_amount=MIN_AMOUNTS[symbol],
            prediction_threshold=PREDICTION_THRESHOLD,
        )
        for symbol in SYMBOLS
    }

    while True:
        print("\n\n")
        synchronize_clocks(minutes=1)
        for symbol, loop_state in loop_states.items():
            info("=" * 20)
            info(f"starting to do {symbol}")
            info("=" * 20)
            info(f"with loop state: {loop_states[symbol]}")
            try:
                loop_states[symbol] = symbol_loop(exchange, loop_state=loop_state)
            except Exception as e:
                error(e, exc_info=True)

            info("loop exited.")
            info(f"new loop state: {loop_states[symbol]}")
