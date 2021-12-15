import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from backtesting import Backtest
from src.evaluation.backtesting_strategies import RegressionStrategy

from omegaconf import DictConfig
from src.pl_data.dataset import MultiTickerDataset


BACKTEST_METRICS = [
    "Exposure Time [%]",
    "Equity Final [$]",
    "Equity Peak [$]",
    "Return [%]",
    "Buy & Hold Return [%]",
    "Return (Ann.) [%]",
    "Volatility (Ann.) [%]",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Calmar Ratio",
    "Max. Drawdown [%]",
    "Avg. Drawdown [%]",
    "Profit Factor",
    "Expectancy [%]",
    "SQN",
]


def startswith_one_of(string, prefixes):
    return any(string.startswith(prefix) for prefix in prefixes)


class BacktestCallback(pl.Callback):
    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        val_multi_dataset: MultiTickerDataset = trainer.datamodule.val_datasets[0]
        datasets = [
            d
            for d in val_multi_dataset.datasets
            if startswith_one_of(d.name, self.cfg.backtest.tested_tickers)
        ]
        assert len(datasets) > 0
        all_stats = {}
        for dataset in datasets:
            dataframe = dataset.dataframe
            bt = Backtest(
                dataframe,
                RegressionStrategy,
                cash=1000,
                commission=0.002,
                exclusive_orders=True,
            )
            stats = bt.run(
                model=pl_module,
                cfg=self.cfg,
                threshold=self.cfg.backtest.threshold,
                go_short=self.cfg.backtest.go_short,
                go_long=self.cfg.backtest.go_long,
                mult_factor=self.cfg.backtest.scale_model_output,
            )
            stats_df = pd.DataFrame(stats).loc[BACKTEST_METRICS]
            stats_dict = {k: v[0] for k, v in stats_df.T.to_dict().items()}
            all_stats[dataset.name] = stats_dict

            # [k for k in stats_df.index if not str(k).startswith("_")]
            # ]
        mean_stats = pd.DataFrame(all_stats).T.mean()
        trainer.logger.experiment.log(
            {
                f"val/backtest/{metric}": mean_stats[metric]
                for metric in mean_stats.index
            }
        )


class ContinueIfNotOverfittingStopping(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        train_loss, val_loss = logs["train/loss"], logs["val/loss"]
        can_stop = train_loss / val_loss > 1

        if can_stop:
            self._run_early_stopping_check(trainer, pl_module)
