from backtesting.backtesting import Backtest
import torch
from pathlib import Path
from sys import argv

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


if __name__ == "__main__":
    from src.ui.ui_utils import get_run_dir

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--use-split", default="val")
    parser.add_argument("--go-short", default=False, action="store_true")
    parser.add_argument("--run-id", default=RUN_ID, type=str)
    parser.add_argument("--project", default=PROJECT, type=str)
    parser.add_argument(
        "--price-path", default="${oc.env:PROJECT_ROOT}/data/ccxt_ohlcv/BTC-USDT*.csv"
    )
    parser.add_argument("--backtest-length-pct", default=None, type=float)
    parser.add_argument("--backtest-start-pct", default=None, type=float)
    args = parser.parse_args()
    assert args.use_split in ("train", "val")
    PROJECT, RUN_ID = args.project, args.run_id

    run_dir: Path = get_run_dir(entity=ENTITY, project=PROJECT, run_id=RUN_ID)
    checkpoint_paths: list[Path] = list(run_dir.rglob("checkpoints/*"))

    cfg, model = get_cfg_model(checkpoint_path=checkpoint_paths[0], run_dir=run_dir)
    print("Got Model and config!")

    cfg.dataset_conf.data_path.data_path = args.price_path

    datamodule = get_datamodule(cfg)
    if args.use_split == "train":
        dataset = datamodule.train_dataset
    elif args.use_split == "val":
        dataset = datamodule.val_datasets[0]
    full_dataframe = dataset.datasets[0].dataframe

    backtest_start = 0
    backtest_length = 5_000
    position_size = 0.5
    price_delta_pct = None

    if args.backtest_start_pct is not None:
        assert 0.0 <= args.backtest_start_pct <= 1.0
        backtest_start = int(len(full_dataframe) * args.backtest_start_pct)
    if args.backtest_length_pct is not None:
        assert 0.0 < args.backtest_length_pct <= 1.0
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
            cash=1_000_000,
            commission=0.002,
            # exclusive_orders=True,
        )
        stats = backtest.run(model=model.cuda(), cfg=cfg, **strategy_kwargs)
        return backtest, stats

    volatility_std_mult = cfg.dataset_conf.dataset_reader.get("std_mult", None)
    price_delta_pct = cfg.dataset_conf.dataset_reader.get("price_delta_pct", None)
    print(f"{price_delta_pct=}")
    print(f"{volatility_std_mult=}")
    long_only_backtest, long_only_stats = backtest_model(
        SequenceTaggerStrategy,
        name=RUN_ID,
        go_long=True,
        go_short=False,
        position_size_pct=position_size,
        price_delta_pct=price_delta_pct,
        volatility_std_mult=volatility_std_mult,
        trailing_mul=None,
    )
    print("Long only results:\n", long_only_stats)
    short_only_backtest, short_only_stats = backtest_model(
        SequenceTaggerStrategy,
        name=RUN_ID,
        go_long=False,
        go_short=True,
        position_size_pct=position_size,
        price_delta_pct=price_delta_pct,
        volatility_std_mult=volatility_std_mult,
        trailing_mul=None,
    )
    print("Short only results:\n", short_only_stats)
    up_down_backtest, up_down_stats = backtest_model(
        SequenceTaggerStrategy,
        name=RUN_ID,
        go_long=True,
        go_short=True,
        position_size_pct=position_size,
        price_delta_pct=price_delta_pct,
        volatility_std_mult=volatility_std_mult,
        trailing_mul=None,
    )
    print("Long+Short results", up_down_stats)
    optimal_backtest, optimal_stats = backtest_model(
        OptimalSequenceTaggerStrategy,
        name="Optimal",
        go_short=args.go_short,
        go_long=True,
        position_size_pct=position_size,
        price_delta_pct=price_delta_pct,
        volatility_std_mult=volatility_std_mult,
    )

    print("Optimal strategy results:\n", optimal_stats)

    long_only_backtest.plot(
        results=long_only_stats, plot_return=True, plot_equity=False
    )
    short_only_backtest.plot(
        results=short_only_stats, plot_return=True, plot_equity=False
    )
    up_down_backtest.plot(results=up_down_stats, plot_return=True, plot_equity=False)

    optimal_backtest.plot(results=optimal_stats, plot_return=True, plot_equity=False)

    import ffn, pandas as pd, matplotlib as plt

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
    print("starting bt backtest")
    portfolio_results = bt.run(portfolio_backtest)
    portfolio_results.plot()
