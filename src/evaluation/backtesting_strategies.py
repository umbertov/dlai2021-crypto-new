from backtesting import Strategy
from backtesting.backtesting import Backtest
from einops import rearrange
from backtesting.lib import SignalStrategy, TrailingStrategy, barssince

from random import random, randint
from typing import Callable, Optional
from tqdm.auto import tqdm

import torch
import numpy as np
import pandas as pd

from src.common.utils import get_model


def pnl_to_signals(pnl: pd.Series, up_threshold=0.0, down_threshold=0.0):
    # where
    # pnl > 0 -> 1
    # pnl < 0 -> -1
    # pnl == 0 -> 0
    signal = (pnl > up_threshold).astype(int).fillna(0)
    signal = signal - (pnl < down_threshold).astype(int).fillna(0)
    return signal


from abc import ABCMeta, abstractmethod


class ModelStrategyBase(Strategy, metaclass=ABCMeta):
    go_long: bool = True
    go_short: bool = True
    price_delta_pct: Optional[float] = None
    position_size_pct: float = 0.999
    cfg: "omegaconf.DictConfig"
    model: torch.nn.Module = None

    def init(self):
        super().init()

        self.model_output = self.I(self.get_model_output)
        # self.model_output = self.get_model_output()

    @abstractmethod
    def get_model_output(self):
        raise NotImplementedError

    def _long_sl_tp_prices(self, price):
        sl, tp = None, None
        if self.price_delta_pct is not None:
            tp = price * (1 + self.price_delta_pct)
            sl = price * (1 - self.price_delta_pct)
        return sl, tp

    def _short_sl_tp_prices(self, price):
        sl, tp = None, None
        if self.price_delta_pct is not None:
            sl, tp = price * (1 + np.r_[1, -1] * self.price_delta_pct)
        return sl, tp

    def next(self):
        price = self.data.Close[-1]
        prediction = self.model_output[-1]
        position_duration = 0
        if len(self.trades) > 0:
            last_trade = self.trades[-1]
            is_open = last_trade.exit_bar is None
            if is_open:
                position_duration = len(self.data) - last_trade.entry_bar
        if prediction == 1:
            if self.position.is_short:
                self.position.close()
            if self.go_long and not self.position:
                sl, tp = self._long_sl_tp_prices(price)
                self.buy(size=self.position_size_pct, tp=tp, sl=sl)
        elif prediction == -1:
            if self.position.is_long:
                self.position.close()
            if self.go_short and not self.position:
                sl, tp = self._short_sl_tp_prices(price)
                self.sell(size=self.position_size_pct, tp=tp, sl=sl)
        if position_duration > self.cfg.dataset_conf.dataset_reader.trend_period:
            self.position.close()


class Monke(Strategy):
    def init(self):
        super().init()

    def next(self):
        if self.position:
            self.position.close()
        if random() > 0.5:
            self.buy(size=0.999)
        else:
            self.sell(size=0.999)


class PerfectOracle(ModelStrategyBase):
    def get_model_output(self):
        open_price = self.data.Open.s
        close_price = self.data.Close.s
        # daily_pnl = 1 - (close_price / open_price).shift(-1)
        self.pred_pnl = self.I(
            lambda: (open_price.shift(-1) / open_price).shift(-1) - 1
        )
        signal = pnl_to_signals(self.pred_pnl.s)
        return signal


class FaultyOracle(PerfectOracle):
    error_rate: float = 0.01

    def get_model_output(self):
        signal = super().get_model_output()
        self.errors = self.I(self.get_error_signal)
        return signal

    def get_error_signal(self):
        errors = self.data.Open.s.map(lambda _: random() < self.error_rate)
        return errors

    def next(self):
        if self.errors[-1]:
            # do the opposite of the perfect strategy (or nothing if it's 0)
            self.model_output[-1] = -self.model_output[-1]
            # do something random
            # self.model_output[-1] = randint(-1, 1)
        return super().next()


class SequenceTaggerStrategy(ModelStrategyBase):
    cfg: "omegaconf.DictConfig" = None
    class2idx = {0: "sell", 1: "hold", 2: "buy"}
    and_predictions = 0

    @torch.no_grad()
    def get_model_output(self):
        channels_last = self.cfg.dataset_conf.channels_last

        model: "src.lightning_modules.TimeSeriesClassifier" = self.model or get_model(
            self.cfg
        )

        assert model is not None
        model = model.eval()

        device = model.device
        model_output = pd.Series([0] * len(self.data))

        input_columns = self.cfg.dataset_conf.input_columns
        input_length = self.cfg.dataset_conf.window_length // 2
        print(f"{input_columns=}, {input_length=}")

        feature_dataframe = self.data.df[input_columns]

        print("beginning to compute model predictions")
        with tqdm(
            total=(len(feature_dataframe) - 2 * input_length) // (input_length // 2)
        ) as pbar:
            for candle_idx in range(
                input_length + 1,
                len(feature_dataframe) - input_length,
                input_length // 2,
            ):
                input_dataframe = feature_dataframe.iloc[
                    (candle_idx - input_length) : candle_idx
                ]
                input_tensor = (
                    torch.tensor(input_dataframe.values, device=device)
                    .view(1, input_length, -1)  #### CHANNELS LAST
                    .float()
                )
                ### THIS BLOCK UNTESTED
                if self.cfg.dataset_conf.zscore_scale_windows == "by_open":
                    open_col = input_tensor[..., 0]
                    mean, std = open_col.mean(), open_col.std()
                    input_tensor = (input_tensor - mean) / std
                if not channels_last:
                    input_tensor = rearrange(input_tensor, "b s c -> b c s")
                # class_indices :: [Batch, SeqLen]
                class_indices = model.predict(input_tensor)
                for i, class_idx in enumerate(class_indices[0, input_length // 2 :]):
                    prediction_step = candle_idx - i
                    class_idx = class_idx.item()
                    class_name = self.class2idx[class_idx]
                    if class_name == "buy":
                        model_output[prediction_step] = 1
                    elif class_name == "sell":
                        model_output[prediction_step] = -1

                pbar.update()

        return model_output


class OptimalSequenceTaggerStrategy(ModelStrategyBase):
    cfg: "omegaconf.DictConfig" = None
    class2idx = {0: "sell", 1: "hold", 2: "buy"}
    and_predictions = 0

    @torch.no_grad()
    def get_model_output(self):
        channels_last = self.cfg.dataset_conf.channels_last

        model: "src.lightning_modules.TimeSeriesClassifier" = self.model or get_model(
            self.cfg
        )

        assert model is not None
        model = model.eval()

        model_output = pd.Series([0] * len(self.data))

        label_column = self.cfg.dataset_conf.categorical_targets
        labels = self.data.df[label_column]

        input_length = self.cfg.dataset_conf.window_length

        print("beginning to compute model predictions")
        with tqdm(total=len(self.data.df) - input_length) as pbar:
            for candle_idx in range(
                input_length + 1,
                len(self.data.df) - input_length,
            ):
                class_idx = int(labels.iloc[candle_idx])
                class_name = self.class2idx[class_idx]
                if class_name == "buy":
                    model_output[candle_idx] = 1
                elif class_name == "sell":
                    model_output[candle_idx] = -1
                pbar.update()

        return model_output


class RegressionStrategy(ModelStrategyBase):
    cfg: "omegaconf.DictConfig" = None
    threshold: float = 1
    mult_factor = 1

    @torch.no_grad()
    def get_model_output(self):

        model = self.model or get_model(self.cfg)

        assert model is not None
        model = model.eval()

        device = model.device
        model_output = pd.Series([0] * len(self.data))

        input_columns = self.cfg.data.dataset_conf.input_columns
        input_length = self.cfg.data.dataset_conf.window_length
        print(f"{input_columns=}, {input_length=}")

        feature_dataframe = self.data.df[input_columns]

        print("beginning to compute model predictions")
        for candle_idx in range(
            input_length + 1, len(feature_dataframe) - input_length
        ):
            input_dataframe = feature_dataframe.iloc[
                (candle_idx - input_length) : candle_idx
            ]
            input_tensor = (
                torch.tensor(input_dataframe.values, device=device)
                .view(1, input_length, -1)
                .float()
            )
            expected_return = torch.tanh(
                self.mult_factor * model(input_tensor)["logits"]
            ).item()
            if expected_return > self.threshold:
                model_output[candle_idx] = 1
            elif expected_return < -self.threshold:
                model_output[candle_idx] = -1

        return model_output


if __name__ == "__main__":
    from src.common.utils import get_hydra_cfg, get_model, get_datamodule
    from sys import argv

    cfg = get_hydra_cfg(overrides=argv[1:])
    model = get_model(cfg).cuda()
    datamodule = get_datamodule(cfg)
    dss = datamodule.train_dataset.datasets
    dfs = [ds.dataframe for ds in dss]
    df = dataframe = dfs[0]
    ds = dataset = dss[0]

    backtest = Backtest(
        dataframe.iloc[10_000:10_500],
        SequenceTaggerStrategy,
        cash=100_000,
        commission=0.002,
        exclusive_orders=True,
    )
    stats = backtest.run(
        model=model, cfg=cfg, go_short=False, go_long=True, and_predictions=2
    )
    print(stats)
