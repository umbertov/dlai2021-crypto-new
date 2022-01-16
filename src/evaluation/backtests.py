from backtesting.backtesting import Backtest
import torch
from pathlib import Path
from sys import argv, exit

import ffn, pandas as pd, matplotlib as plt
import pandas as pd

import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 200

from argparse import ArgumentParser

from src.common.utils import get_hydra_cfg, get_model, get_datamodule
from src.evaluation.backtesting_strategies import (
    BuyAndHold,
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

BACKTEST_METRICS = [
    "Return [%]",
    "Buy & Hold Return [%]",
    "Sharpe Ratio",
    "Max. Drawdown [%]",
    "Avg. Drawdown [%]",
    "Profit Factor",
    "Win Rate [%]",
]

OUT_DIR = "evaluation/backtest"


def parse_args():
    def none_or_float(value):
        if value == "None":
            return None
        return float(value)

    global PROJECT, RUN_ID, ENTITY
    parser = ArgumentParser()
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--use-split", default="test")
    parser.add_argument("--close-on-signal", default=False, action="store_true")
    parser.add_argument("--run-id", default=RUN_ID, type=str)
    parser.add_argument("--project", default=PROJECT, type=str)
    parser.add_argument("--entity", default=ENTITY, type=str)
    parser.add_argument(
        "--data-path", default="${oc.env:PROJECT_ROOT}/data/ccxt_ohlcv/BTC-USDT*.csv"
    )
    parser.add_argument("--backtest-length-pct", default=1.0, type=float)
    parser.add_argument("--backtest-start-pct", default=0.0, type=float)
    parser.add_argument("--position-size-pct", default=1, type=float)
    parser.add_argument("--show-plots", default=False, action="store_true")
    parser.add_argument("--verbose-stats", default=False, action="store_true")
    parser.add_argument("--optimal", default=False, action="store_true")
    parser.add_argument("--buy-and-hold", default=False, action="store_true")
    parser.add_argument("--prediction-threshold", default=0.7, type=none_or_float)
    args = parser.parse_args()
    assert args.use_split in ("train", "val", "test")
    assert 0.0 <= args.backtest_start_pct <= 1.0
    assert 0.0 < args.backtest_length_pct <= 1.0
    assert args.prediction_threshold is None or 0.0 < args.prediction_threshold < 1.0
    if args.position_size_pct == 1:
        args.position_size_pct = int(args.position_size_pct)
    return args


if __name__ == "__main__":
    from src.ui.ui_utils import get_run_dir

    ##### CLI ARGS
    args = parse_args()
    PROJECT, RUN_ID, ENTITY = args.project, args.run_id, args.entity
    OUT_DIR = args.out_dir
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    ##### LOAD HYDRA CFG, MODEL CHECKPOINT FROM WANDB RUN
    run_dir: Path = get_run_dir(entity=ENTITY, project=PROJECT, run_id=RUN_ID)
    checkpoint_paths: list[Path] = list(run_dir.rglob("checkpoints/*"))
    cfg, model = get_cfg_model(checkpoint_path=checkpoint_paths[0], run_dir=run_dir)
    # override dataset path
    cfg.dataset_conf.data_path.data_path = args.data_path

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

    volatility_std_mult = None  # cfg.dataset_conf.dataset_reader.get("std_mult", None)
    price_delta_pct = cfg.dataset_conf.dataset_reader.get("price_delta_pct", None)
    print(f"{price_delta_pct=}")
    print(f"{volatility_std_mult=}")

    ##### BUY AND HOLD BACKTESTING
    if args.buy_and_hold:
        buy_and_hold_backtest = Backtest(
            full_dataframe.iloc[backtest_start : backtest_start + backtest_length],
            BuyAndHold,
            cash=100_000,
            commission=0.002,
        )
        buy_and_hold_stats = buy_and_hold_backtest.run(
            name=f"BuyAndHold.{RUN_ID}.{args.use_split}",
        )
        buy_and_hold_s = pd.Series(buy_and_hold_stats).loc[BACKTEST_METRICS]
        global_stats = pd.DataFrame({"Buy and Hold": buy_and_hold_s})
        buy_and_hold_backtest.plot(
            filename=f"{OUT_DIR}/{str(buy_and_hold_stats._strategy)}.html",
            results=buy_and_hold_stats,
            plot_return=True,
            plot_equity=False,
            resample="1d",
            open_browser=args.show_plots,
        )
        #### Export it to CSV and print latex table
        global_stats.to_csv(
            f"{OUT_DIR}/backtest_metrics.BuyAndHold.{args.use_split}.csv"
        )
        print("LATEX TABLE:\n")
        print(f"%%%% Buy & Hold {args.use_split}")
        print(global_stats.to_latex(float_format=lambda x: f"{x:0.2f}"), "\n\n")
        global_stats.to_latex(
            f"{OUT_DIR}/latex_table.BuyAndHold.{args.use_split}.latex",
            float_format=lambda x: f"{x:0.2f}",
        )

        exit(0)

    #### OPTIMAL STRATEGY TESTING
    if args.optimal:
        ## Optimal Long-Only Strategy
        optimal_long_backtest, optimal_long_stats = backtest_model(
            OptimalSequenceTaggerStrategy,
            name="OptimalLongOnly",
            go_short=False,
            go_long=True,
            close_on_signal=args.close_on_signal,
            position_size_pct=position_size,
            price_delta_pct=price_delta_pct,
            volatility_std_mult=volatility_std_mult,
        )
        if args.verbose_stats:
            print("Optimal Long Only strategy results:\n\n", optimal_long_stats)
        ## Optimal Short-Only Strategy
        optimal_short_backtest, optimal_short_stats = backtest_model(
            OptimalSequenceTaggerStrategy,
            name="OptimalShortOnly",
            go_short=True,
            go_long=False,
            close_on_signal=args.close_on_signal,
            position_size_pct=position_size,
            price_delta_pct=price_delta_pct,
            volatility_std_mult=volatility_std_mult,
        )
        if args.verbose_stats:
            print("Optimal Short Only strategy results:\n\n", optimal_short_stats)
        ## Optimal Long-Short Strategy
        optimal_updown_backtest, optimal_updown_stats = backtest_model(
            OptimalSequenceTaggerStrategy,
            name="OptimalLongShort",
            go_short=True,
            go_long=True,
            close_on_signal=args.close_on_signal,
            position_size_pct=position_size,
            price_delta_pct=price_delta_pct,
            volatility_std_mult=volatility_std_mult,
        )
        if args.verbose_stats:
            print("Optimal Long+Short strategy results:\n\n", optimal_updown_stats)
        long_s = pd.Series(optimal_long_stats).loc[BACKTEST_METRICS]
        short_s = pd.Series(optimal_short_stats).loc[BACKTEST_METRICS]
        updown_s = pd.Series(optimal_updown_stats).loc[BACKTEST_METRICS]
        global_stats = pd.DataFrame(
            {
                "Long Only": long_s,
                "Short Only": short_s,
                "Up/Down": updown_s,
            }
        )
        #### Export it to CSV and print latex table
        global_stats.to_csv(f"{OUT_DIR}/backtest_metrics.Optimal.{args.use_split}.csv")
        print("LATEX TABLE:\n")
        print(f"%%%% OPTIMAL {args.use_split}")
        print(global_stats.to_latex(float_format=lambda x: f"{x:0.2f}"), "\n\n")
        global_stats.to_latex(
            f"{OUT_DIR}/latex_table.Optimal.{args.use_split}.latex",
            float_format=lambda x: f"{x:0.2f}",
        )

    #### MODEL-BASED BACKTESTING

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
        prediction_threshold=args.prediction_threshold,
    )
    if args.verbose_stats:
        print("Long only results:\n\n", long_only_stats)
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
        prediction_threshold=args.prediction_threshold,
    )
    if args.verbose_stats:
        print("Short only results:\n\n", short_only_stats)
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
        prediction_threshold=args.prediction_threshold,
    )
    if args.verbose_stats:
        print("Long+Short results\n\n", up_down_stats)

    ##### PLOT ALL PREVIOUS RESULTS
    long_only_backtest.plot(
        filename=f"{OUT_DIR}/{str(long_only_stats._strategy)}.html",
        results=long_only_stats,
        plot_return=True,
        plot_equity=False,
        resample="1d",
        open_browser=args.show_plots,
    )
    short_only_backtest.plot(
        filename=f"{OUT_DIR}/{str(short_only_stats._strategy)}.html",
        results=short_only_stats,
        plot_return=True,
        plot_equity=False,
        resample="1d",
        open_browser=args.show_plots,
    )
    up_down_backtest.plot(
        filename=f"{OUT_DIR}/{str(up_down_stats._strategy)}.html",
        results=up_down_stats,
        plot_return=True,
        plot_equity=False,
        resample="1d",
        open_browser=args.show_plots,
    )

    import pandas as pd

    #### Aggregate all relevant stats in a single dataframe
    long_s = pd.Series(long_only_stats).loc[BACKTEST_METRICS]
    short_s = pd.Series(short_only_stats).loc[BACKTEST_METRICS]
    updown_s = pd.Series(up_down_stats).loc[BACKTEST_METRICS]
    global_stats = pd.DataFrame(
        {
            "Long Only": long_s,
            "Short Only": short_s,
            "Up/Down": updown_s,
        }
    )
    #### Export it to CSV and print latex table
    global_stats.to_csv(f"{OUT_DIR}/backtest_metrics.{RUN_ID}.{args.use_split}.csv")
    print("LATEX TABLE:\n")
    print(f"%%%% {RUN_ID} {args.use_split}")
    print(global_stats.to_latex(float_format=lambda x: f"{x:0.2f}"), "\n\n")
    global_stats.to_latex(
        f"{OUT_DIR}/latex_table.{RUN_ID}.{args.use_split}.latex",
        float_format=lambda x: f"{x:0.2f}",
    )
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

    ax = data.plot(figsize=(10, 7))
    if args.show_plots:
        ax.figure.show()
    ax.figure.savefig(f"{OUT_DIR}/equities {RUN_ID}.{args.use_split}.png")
