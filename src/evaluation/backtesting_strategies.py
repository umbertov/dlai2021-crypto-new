from backtesting import Strategy
from backtesting.backtesting import Backtest
from einops import rearrange
from backtesting.lib import SignalStrategy, TrailingStrategy, barssince
from typing import Optional

from tqdm.auto import tqdm

import torch
import pandas as pd

from src.common.utils import get_model

from abc import ABCMeta, abstractmethod


class ModelStrategyBase(TrailingStrategy, metaclass=ABCMeta):
    name = "no_name"
    go_long: bool = True
    go_short: bool = True
    take_profit_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    volatility_std_mult: Optional[float] = None
    position_size_pct: float = 0.999
    cfg: "omegaconf.DictConfig"
    model: torch.nn.Module = None
    trailing_mul: Optional[int] = None
    close_on_signal: bool = False
    max_entries = 1

    def init(self):
        super().init()
        # Counts how many signal-based entries were made
        self.entry_count = 0

        self.model_output = self.I(self.get_model_output)
        if self.volatility_std_mult is not None:
            assert self.take_profit_pct is None
            self.volatility = self.I(self.get_volatility)
        if self.trailing_mul:
            self.set_trailing_sl(self.trailing_mul)
        # self.model_output = self.get_model_output()

    @abstractmethod
    def get_volatility(self):
        raise NotImplementedError

    @abstractmethod
    def get_model_output(self):
        raise NotImplementedError

    def next(self):
        price = self.data.Close[-1]
        prediction = self.model_output[-1]
        position_duration = 0
        # first, cancel any non-executed order
        for order in self.orders:
            order.cancel()

        # then, optionally take profit from all good positions
        if self.take_profit_pct is not None:
            for trade in self.trades:
                if trade.pl_pct > self.take_profit_pct:
                    trade.close()
                elif trade.pl_pct < -self.stop_loss_pct:
                    trade.close()

        if self.position:
            position_duration = len(self.data) - self.trades[-1].entry_bar

        # if present, compute length of current trade (in bars)
        if position_duration > self.cfg.dataset_conf.dataset_reader.trend_period:
            self.position.close()

        if prediction == 1:
            if self.close_on_signal and self.position.is_short:
                self.position.close()
            elif self.go_long and len(self.trades) < self.max_entries:
                self.buy(size=self.position_size_pct)
        elif prediction == -1:
            if self.close_on_signal and self.position.is_long:
                self.position.close()
            elif self.go_short and len(self.trades) < self.max_entries:
                self.sell(size=self.position_size_pct)


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


class SequenceTaggerStrategy(ModelStrategyBase):
    cfg: "omegaconf.DictConfig" = None
    class2idx = {0: "sell", 1: "hold", 2: "buy"}
    prediction_threshold = None
    verbose = True

    @torch.no_grad()
    def get_volatility(self):
        return self.data.df["RollingReturnVolatility"]

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
        print(f"{self.prediction_threshold=} ")

        feature_dataframe = self.data.df[input_columns]

        print("beginning to compute model predictions")
        pbar = None
        if self.verbose:
            pbar = tqdm(
                total=(len(feature_dataframe) - 2 * input_length) // (input_length // 2)
            )
        for candle_idx in range(
            input_length + 1,
            len(feature_dataframe) - input_length,
            input_length // 2,
        ):
            input_dataframe = feature_dataframe.iloc[
                (candle_idx - input_length) : candle_idx
            ]
            # input_tensor :: [ Batch, Seq, Chan ]
            input_tensor = (
                torch.tensor(input_dataframe.values, device=device)
                .view(1, input_length, -1)  #### CHANNELS LAST
                .float()
            )
            if self.cfg.dataset_conf.zscore_scale_windows == "by_open":
                open_col = input_tensor[..., 0]
                mean, std = open_col.mean(), open_col.std()
                input_tensor = (input_tensor - mean) / std
            if not channels_last:
                input_tensor = rearrange(input_tensor, "b s c -> b c s")
            # class_indices :: [Batch, SeqLen]
            class_indices = model.predict(
                input_tensor, threshold=self.prediction_threshold
            )
            for i, class_idx in enumerate(class_indices[0, input_length // 2 :]):
                prediction_step = candle_idx - i
                class_idx = class_idx.item()
                class_name = self.class2idx[class_idx]
                if class_name == "buy":
                    model_output[prediction_step] = 1
                elif class_name == "sell":
                    model_output[prediction_step] = -1

            if pbar:
                pbar.update()
        if pbar:
            pbar.close()

        return model_output


class OptimalSequenceTaggerStrategy(SequenceTaggerStrategy):
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


class BuyAndHold(Strategy):
    name: str = "BuyAndHold"

    def init(self):
        super().init()

    def next(self):
        if not self.position:
            self.buy(size=0.999999)


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
        model=model,
        cfg=cfg,
        go_short=False,
        go_long=True,
    )
    print(stats)
