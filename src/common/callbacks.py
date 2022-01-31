import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from backtesting import Backtest
from src.evaluation.backtesting_strategies import SequenceTaggerStrategy

from omegaconf import DictConfig


def startswith_one_of(string, prefixes):
    return any(string.startswith(prefix) for prefix in prefixes)


class ShuffleDatasetIndices(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        print("shuffling train dataset")
        trainer.train_dataloader.dataset.datasets.reset()

    def on_validation_epoch_start(self, trainer, pl_module):
        print("shuffling val dataset")
        trainer.val_dataloaders[0].dataset.reset()


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


def backtest_model(
    dataframe,
    model,
    cfg,
    strategy,
    **strategy_kwargs,
):

    backtest = Backtest(
        dataframe,
        strategy,
        cash=100_000,
        commission=0.002,
        # exclusive_orders=True,
    )
    stats = backtest.run(model=model.cuda(), cfg=cfg, **strategy_kwargs)
    return backtest, stats


class BacktestCallback(pl.Callback):
    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.volatility_std_mult = (
            None  # cfg.dataset_conf.dataset_reader.get("std_mult", None)
        )
        self.price_delta_pct = cfg.dataset_conf.dataset_reader.get(
            "price_delta_pct", None
        )

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        val_multi_dataset = trainer.datamodule.val_datasets[0]
        datasets = val_multi_dataset.datasets
        assert len(datasets) > 0
        all_stats = {}
        dataset = datasets[0]
        dataframe = dataset.dataframe
        # bt = Backtest(
        #     dataframe,
        #     SequenceTaggerStrategy,
        #     cash=100_000,
        #     commission=0.002,
        #     # exclusive_orders=True,
        # )
        # stats = bt.run(
        bt, stats = backtest_model(
            dataframe=dataframe,
            strategy=SequenceTaggerStrategy,
            model=pl_module,
            cfg=self.cfg,
            go_short=True,
            go_long=True,
            position_size_pct=0.5,
            verbose=False,
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
