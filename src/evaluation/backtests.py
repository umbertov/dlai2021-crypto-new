from typing import List
from backtesting.backtesting import Backtest
import hydra
import torch
from einops import rearrange
import os
from pathlib import Path
import plotly.graph_objects as go
from sys import argv

from src.common.plot_utils import plot_ohlcv, plot_multi_lines
from src.common.utils import get_hydra_cfg, get_model, get_datamodule
from src.evaluation.backtesting_strategies import (
    OptimalSequenceTaggerStrategy,
    SequenceTaggerStrategy,
)

# We don't ever need gradient descent here
torch.set_grad_enabled(False)


ENTITY, PROJECT, RUN_ID = "erpi", "dlai-stonks-new", "wandb_run_id"


def load_model_checkpoint(model, checkpoint_path: Path):
    return model.load_from_checkpoint(checkpoint_path=str(checkpoint_path))


def get_cfg_model(checkpoint_path=None, run_dir=None):
    if checkpoint_path is not None and run_dir is not None:
        cfg = get_hydra_cfg(config_path=(f"{run_dir}/files"), config_name="hparams")
        model = load_model_checkpoint(get_model(cfg), checkpoint_path=checkpoint_path)
    else:
        cfg = get_hydra_cfg(overrides=argv[1:])
        model = get_model(cfg)
    return cfg, model


if __name__ == "__main__":
    from src.ui.ui_utils import get_run_dir

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--go-short", default=False, action="store_true")
    parser.add_argument("--run-id", default=RUN_ID, type=str)
    parser.add_argument("--project", default=PROJECT, type=str)
    parser.add_argument(
        "--price-path", default="${oc.env:PROJECT_ROOT}/data/ccxt_ohlcv/BTC-USDT*.csv"
    )
    parser.add_argument("--backtest-length-pct", default=None, type=float)
    parser.add_argument("--backtest-start-pct", default=None, type=float)
    args = parser.parse_args()
    PROJECT, RUN_ID = args.project, args.run_id

    run_dir: Path = get_run_dir(entity=ENTITY, project=PROJECT, run_id=RUN_ID)
    checkpoint_paths: list[Path] = list(run_dir.rglob("checkpoints/*"))

    cfg, model = get_cfg_model(checkpoint_path=checkpoint_paths[0], run_dir=run_dir)
    print("Got Model and config!")

    cfg.dataset_conf.data_path.data_path = args.price_path

    datamodule = get_datamodule(cfg)
    train_dataloader = datamodule.train_dataloader()
    train_dataset = datamodule.train_dataset
    val_dataloader = datamodule.val_dataloader()
    val_dataset = datamodule.val_datasets[0]
    full_dataframe = train_dataset.datasets[0].dataframe

    backtest_start = 0
    backtest_length = 5_000
    position_size = 0.5
    price_delta_pct = None

    if args.backtest_start_pct is not None:
        assert 0.0 < args.backtest_start_pct <= 1.0
        backtest_start = int(len(full_dataframe) * args.backtest_start_pct)
    if args.backtest_length_pct is not None:
        assert 0.0 < args.backtest_length_pct < 1.0
        backtest_length = int(len(full_dataframe) * args.backtest_length_pct)

    def backtest_model(
        backtest_start=backtest_start,
        backtest_length=backtest_length,
        position_size=position_size,
        price_delta_pct=price_delta_pct,
    ):

        backtest = Backtest(
            full_dataframe.iloc[backtest_start : backtest_start + backtest_length],
            SequenceTaggerStrategy,
            cash=1_000_000,
            commission=0.002,
            # exclusive_orders=True,
        )
        stats = backtest.run(
            model=model.cuda(),
            cfg=cfg,
            go_short=args.go_short,
            go_long=True,
            and_predictions=0,
            position_size_pct=position_size,
            price_delta_pct=price_delta_pct,
        )
        return backtest, stats

    backtest, stats = backtest_model(
        price_delta_pct=cfg.dataset_conf.dataset_reader.alpha
    )
    print(stats)
    backtest.plot(results=stats, plot_return=True, plot_equity=False)

    optimal_backtest = Backtest(
        full_dataframe.iloc[backtest_start : backtest_start + backtest_length],
        OptimalSequenceTaggerStrategy,
        cash=1_000_000,
        commission=0.002,
        # exclusive_orders=True,
    )
    optimal_stats = optimal_backtest.run(
        model=model.cuda(),
        cfg=cfg,
        go_short=args.go_short,
        go_long=True,
        and_predictions=0,
        position_size_pct=position_size,
        price_delta_pct=cfg.dataset_conf.dataset_reader.alpha,
    )
    print(optimal_stats)
    optimal_backtest.plot(results=optimal_stats, plot_return=True, plot_equity=False)
