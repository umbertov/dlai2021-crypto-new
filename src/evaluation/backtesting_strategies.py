from backtesting import Strategy
from backtesting.lib import SignalStrategy, TrailingStrategy

from random import random, randint
from typing import Callable
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
    cfg: "omegaconf.DictConfig"
    model: torch.nn.Module = None

    def init(self):
        super().init()

        self.model_output = self.I(self.get_model_output)
        # self.model_output = self.get_model_output()

    @abstractmethod
    def get_model_output(self):
        raise NotImplementedError

    def next(self):
        if self.position:
            self.position.close()

        prediction = self.model_output[-1]
        if prediction == 1 and self.go_long:
            self.buy(size=0.999)
        elif prediction == -1 and self.go_short:
            self.sell(size=0.999)


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

    @torch.no_grad()
    def get_model_output(self):

        model: "torch.nn.Module" = self.model or get_model(self.cfg)

        assert model is not None
        model = model.eval()

        device = model.device
        model_output = pd.Series([0] * len(self.data))

        input_columns = self.cfg.data.dataset_conf.input_columns
        input_length = self.cfg.data.dataset_conf.input_length
        print(f"{input_columns=}, {input_length=}")

        feature_dataframe = self.data.df[input_columns]

        print("beginning to compute model predictions")
        for candle_idx in range(
            input_length + 1,
            len(feature_dataframe) - input_length,
        ):
            input_dataframe = feature_dataframe.iloc[
                (candle_idx - input_length) : candle_idx
            ]
            input_tensor = (
                torch.tensor(input_dataframe.values, device=device)
                .view(1, input_length, -1)  #### CHANNELS LAST
                .float()
            )
            # class_indices :: [Batch, SeqLen]
            class_indices = model.predict(input_tensor)
            class_idx = class_indices[0, -1]
            if class_idx == self.class2idx["buy"]:
                model_output[candle_idx] = 1
            elif class_idx == self.class2idx["sell"]:
                model_output[candle_idx] = -1

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
        input_length = self.cfg.data.dataset_conf.input_length
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
