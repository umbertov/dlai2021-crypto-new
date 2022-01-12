from backtesting.backtesting import Backtest
import torch
from pathlib import Path
from sys import argv

from argparse import ArgumentParser

from src.common.utils import get_hydra_cfg, get_model, get_datamodule
from src.evaluation.backtesting_strategies import (
    OptimalSequenceTaggerStrategy,
    SequenceTaggerStrategy,
)
from src.evaluation.common import (
    get_cfg_model,
    PROJECT,
    RUN_ID,
    ENTITY,
    DEVICE,
    move_dict,
)

# We don't ever need gradient descent here
torch.set_grad_enabled(False)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--use-split", default="test")
    parser.add_argument("--close-on-signal", default=False, action="store_true")
    parser.add_argument("--run-id", default=RUN_ID, type=str)
    parser.add_argument("--project", default=PROJECT, type=str)
    parser.add_argument(
        "--price-path", default="${oc.env:PROJECT_ROOT}/data/ccxt_ohlcv/BTC-USDT*.csv"
    )
    parser.add_argument("--backtest-length-pct", default=1.0, type=float)
    parser.add_argument("--backtest-start-pct", default=0.0, type=float)
    parser.add_argument("--position-size-pct", default=0.5, type=float)
    args = parser.parse_args()
    assert args.use_split in ("train", "val", "test")
    assert 0.0 <= args.backtest_start_pct <= 1.0
    assert 0.0 < args.backtest_length_pct <= 1.0
    if args.position_size_pct == 1:
        args.position_size_pct = int(args.position_size_pct)
    return args


if __name__ == "__main__":
    from src.ui.ui_utils import get_run_dir

    ##### CLI ARGS
    args = parse_args()
    PROJECT, RUN_ID = args.project, args.run_id

    ##### LOAD HYDRA CFG, MODEL CHECKPOINT FROM WANDB RUN
    run_dir: Path = get_run_dir(entity=ENTITY, project=PROJECT, run_id=RUN_ID)
    checkpoint_paths: list[Path] = list(run_dir.rglob("checkpoints/*"))
    cfg, model = get_cfg_model(checkpoint_path=checkpoint_paths[0], run_dir=run_dir)
    print("Got Model and config!")
    # override dataset path
    cfg.dataset_conf.data_path.data_path = args.price_path

    # Load data
    datamodule = get_datamodule(
        cfg, "fit" if args.use_split in ("train", "val") else "test"
    )
    if args.use_split == "train":
        dataset = datamodule.train_dataset
    elif args.use_split == "val":
        dataset = datamodule.val_datasets[0]
    elif args.use_split == "test":
        dataset = datamodule.test_datasets[0]
    # We'll use the underlying dataframe, not the Dataset instance itself
    full_dataframe = dataset.datasets[0].dataframe

    # Backtest parameters
    position_size = args.position_size_pct
    price_delta_pct = None
    # turn backtest start/length from percentages into integer number of steps
    backtest_start = int(len(full_dataframe) * args.backtest_start_pct)
    backtest_length = int(len(full_dataframe) * args.backtest_length_pct)

    def backtest_model(
        strategy,
        backtest_start=backtest_start,
        backtest_length=backtest_length,
        **strategy_kwargs,
    ):

        backtest = Backtest(
            full_dataframe.iloc[backtest_start : backtest_start + backtest_length],
            strategy,
            cash=100_000,
            commission=0.002,
            # exclusive_orders=True,
        )
        stats = backtest.run(model=model.cuda(), cfg=cfg, **strategy_kwargs)
        return backtest, stats

    volatility_std_mult = cfg.dataset_conf.dataset_reader.get("std_mult", None)
    price_delta_pct = cfg.dataset_conf.dataset_reader.get("price_delta_pct", None)
    print(f"{price_delta_pct=}")
    print(f"{volatility_std_mult=}")

    ##### BACKTESTING
    ## Long-Only Strategy
    long_only_backtest, long_only_stats = backtest_model(
        SequenceTaggerStrategy,
        name=f"{RUN_ID}.{args.use_split}",
        go_long=True,
        go_short=False,
        close_on_signal=args.close_on_signal,
        position_size_pct=position_size,
        price_delta_pct=price_delta_pct,
        volatility_std_mult=volatility_std_mult,
        trailing_mul=None,
    )
    print("Long only results:\n", long_only_stats)
    ## Short-Only Strategy
    short_only_backtest, short_only_stats = backtest_model(
        SequenceTaggerStrategy,
        name=f"{RUN_ID}.{args.use_split}",
        go_long=False,
        go_short=True,
        close_on_signal=args.close_on_signal,
        position_size_pct=position_size,
        price_delta_pct=price_delta_pct,
        volatility_std_mult=volatility_std_mult,
        trailing_mul=None,
    )
    print("Short only results:\n", short_only_stats)
    ## Long-Short Strategy
    up_down_backtest, up_down_stats = backtest_model(
        SequenceTaggerStrategy,
        name=f"{RUN_ID}.{args.use_split}",
        go_long=True,
        go_short=True,
        close_on_signal=args.close_on_signal,
        position_size_pct=position_size,
        price_delta_pct=price_delta_pct,
        volatility_std_mult=volatility_std_mult,
        trailing_mul=None,
    )
    print("Long+Short results", up_down_stats)
    ## Optimal Long-Only Strategy
    optimal_backtest, optimal_stats = backtest_model(
        OptimalSequenceTaggerStrategy,
        name="OptimalLongOnly",
        go_short=False,
        go_long=True,
        close_on_signal=args.close_on_signal,
        position_size_pct=position_size,
        price_delta_pct=price_delta_pct,
        volatility_std_mult=volatility_std_mult,
    )

    print("Optimal strategy results:\n", optimal_stats)

    ##### PLOT ALL PREVIOUS RESULTS
    long_only_backtest.plot(
        results=long_only_stats, plot_return=True, plot_equity=False, resample="4h"
    )
    short_only_backtest.plot(
        results=short_only_stats, plot_return=True, plot_equity=False, resample="4h"
    )
    up_down_backtest.plot(
        results=up_down_stats, plot_return=True, plot_equity=False, resample="4h"
    )
    optimal_backtest.plot(
        results=optimal_stats, plot_return=True, plot_equity=False, resample="4h"
    )

    import ffn, pandas as pd, matplotlib as plt

    # Visualize and compare equity curves
    long_only_equity = long_only_stats._equity_curve.Equity
    short_only_equity = short_only_stats._equity_curve.Equity
    up_down_equity = up_down_stats._equity_curve.Equity
    data = pd.DataFrame(
        {
            "btc": full_dataframe.Close,
            "long": long_only_equity,
            "short": short_only_equity,
            "longshort": up_down_equity,
        }
    ).rebase()

    data.plot()

    import bt

    def specify_cash(cash):
        def f(target):
            target.temp["cash"] = 0.2
            return True

        return f

    # Simulate a portfolio that runs these strats simultaneously + holds btc
    s = bt.Strategy(
        "strat portfolio",
        [
            specify_cash(0.2),
            bt.algos.RunIfOutOfBounds(0.2),
            bt.algos.SelectAll(),
            bt.algos.WeighSpecified(
                **{"long": 0.1, "short": 0.1, "longshort": 0.1, "btc": 0.1}
            ),
            bt.algos.Rebalance(),
        ],
    )
    portfolio_backtest = bt.Backtest(s, data)
    print("starting bt backtest...")
    portfolio_results = bt.run(portfolio_backtest)
    print("done. plotting bt backtest...")
    portfolio_results.plot()
