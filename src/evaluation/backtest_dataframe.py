from backtesting.backtesting import Backtest
import hydra
import torch
from pathlib import Path
from sys import argv, exit

import ffn, pandas as pd, matplotlib as plt
import pandas as pd

import matplotlib as mpl

from src.dataset_readers import Compose, DateRangeCut

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
    "Sharpe Ratio",
    "Max. Drawdown [%]",
    "Win Rate [%]",
]

OUT_DIR = "evaluation/backtest"


def parse_args():
    def none_or(other):
        """Argparse validator"""

        def f(x):
            if x == "None":
                return None
            return other(x)

        return f

    # validators declaration
    none_or_float = none_or(float)
    none_or_datetime = none_or(lambda x: pd.to_datetime(x).date())

    global PROJECT, RUN_ID, ENTITY
    parser = ArgumentParser()
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--start-date", default=None, type=none_or_datetime)
    parser.add_argument("--end-date", default=None, type=none_or_datetime)
    parser.add_argument("--close-on-signal", default=False, action="store_true")
    parser.add_argument("--go-long", default=True, type=bool)
    parser.add_argument("--go-short", default=True, type=bool)
    parser.add_argument("--resample", default="1d", type=str)
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
    parser.add_argument("--prediction-threshold", default=None, type=none_or_float)
    parser.add_argument("--price-delta-pct", default=None, type=none_or_float)
    parser.add_argument("--max-entries", default=1, type=int)

    args = parser.parse_args()
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

    dataset_reader = hydra.utils.instantiate(cfg.dataset_conf.dataset_reader)
    dataframe: pd.DataFrame = dataset_reader(args.data_path)
    if args.start_date and args.end_date:
        dataframe = DateRangeCut(args.start_date, args.end_date)(dataframe)

    # Backtest parameters
    position_size = args.position_size_pct
    price_delta_pct = args.price_delta_pct
    # turn backtest start/length from percentages into integer number of steps
    backtest_start = int(len(dataframe) * args.backtest_start_pct)
    backtest_length = int(len(dataframe) * args.backtest_length_pct)

    def backtest_model(
        strategy,
        backtest_start=backtest_start,
        backtest_length=backtest_length,
        **strategy_kwargs,
    ):

        backtest = Backtest(
            dataframe.iloc[backtest_start : backtest_start + backtest_length],
            strategy,
            cash=100_000 * args.max_entries,
            commission=0.002,
            # exclusive_orders=True,
        )
        stats = backtest.run(model=model.cuda(), cfg=cfg, **strategy_kwargs)
        return backtest, stats

    volatility_std_mult = None  # cfg.dataset_conf.dataset_reader.get("std_mult", None)
    if price_delta_pct is None:
        price_delta_pct = cfg.dataset_conf.dataset_reader.get("price_delta_pct", None)
    print(f"{price_delta_pct=}")
    print(f"{volatility_std_mult=}")

    ##### BUY AND HOLD BACKTESTING
    if args.buy_and_hold:
        buy_and_hold_backtest = Backtest(
            dataframe.iloc[backtest_start : backtest_start + backtest_length],
            BuyAndHold,
            cash=100_000,
            commission=0.002,
        )
        buy_and_hold_stats = buy_and_hold_backtest.run(
            name=f"BuyAndHold.{RUN_ID}.{args.start_date}-{args.end_date}",
        )
        buy_and_hold_s = pd.Series(buy_and_hold_stats).loc[BACKTEST_METRICS]
        global_stats = pd.DataFrame({"Buy and Hold": buy_and_hold_s})
        buy_and_hold_backtest.plot(
            filename=f"{OUT_DIR}/{str(buy_and_hold_stats._strategy)}.html",
            results=buy_and_hold_stats,
            plot_return=True,
            plot_equity=False,
            resample=args.resample,
            open_browser=args.show_plots,
        )
        #### Export it to CSV and print latex table
        global_stats.to_csv(
            f"{OUT_DIR}/backtest_metrics.BuyAndHold.{args.start_date}-{args.end_date}.csv"
        )
        print("LATEX TABLE:\n")
        print(f"%%%% Buy & Hold {args.start_date}-{args.end_date}")
        print(global_stats.to_latex(float_format=lambda x: f"{x:0.2f}"), "\n\n")
        global_stats.to_latex(
            f"{OUT_DIR}/latex_table.BuyAndHold.{args.start_date}-{args.end_date}.latex",
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
            max_entries=args.max_entries,
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
            max_entries=args.max_entries,
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
            max_entries=args.max_entries,
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
        global_stats.to_csv(
            f"{OUT_DIR}/backtest_metrics.Optimal.{args.start_date}-{args.end_date}.csv"
        )
        print("LATEX TABLE:\n")
        print(f"%%%% OPTIMAL {args.start_date}-{args.end_date}")
        print(global_stats.to_latex(float_format=lambda x: f"{x:0.2f}"), "\n\n")
        global_stats.to_latex(
            f"{OUT_DIR}/latex_table.Optimal.{args.start_date}-{args.end_date}.latex",
            float_format=lambda x: f"{x:0.2f}",
        )

    #### MODEL-BASED BACKTESTING

    up_down_backtest, up_down_stats = backtest_model(
        SequenceTaggerStrategy,
        name=f"{RUN_ID}.{args.start_date}-{args.end_date}",
        go_long=args.go_long,
        go_short=args.go_short,
        close_on_signal=args.close_on_signal,
        position_size_pct=position_size,
        price_delta_pct=price_delta_pct,
        volatility_std_mult=volatility_std_mult,
        trailing_mul=None,
        prediction_threshold=args.prediction_threshold,
        max_entries=args.max_entries,
    )
    if args.verbose_stats:
        print("Long+Short results\n\n", up_down_stats)

    ##### PLOT ALL PREVIOUS RESULTS
    up_down_backtest.plot(
        filename=f"{OUT_DIR}/{str(up_down_stats._strategy)}.html",
        results=up_down_stats,
        plot_return=True,
        plot_equity=False,
        resample=args.resample,
        open_browser=args.show_plots,
    )

    import pandas as pd

    #### Aggregate all relevant stats in a single dataframe
    updown_s = pd.Series(up_down_stats).loc[BACKTEST_METRICS]
    global_stats = pd.DataFrame(
        {
            # "Long Only": long_s,
            # "Short Only": short_s,
            "Up/Down": updown_s,
        }
    )
    #### Export it to CSV and print latex table
    global_stats.to_csv(
        f"{OUT_DIR}/backtest_metrics.{RUN_ID}.{args.start_date}-{args.end_date}.csv"
    )
    print("LATEX TABLE:\n")
    print(f"%%%% {RUN_ID} {args.start_date}-{args.end_date}")
    print(global_stats.to_latex(float_format=lambda x: f"{x:0.2f}"), "\n\n")
    global_stats.to_latex(
        f"{OUT_DIR}/latex_table.{RUN_ID}.{args.start_date}-{args.end_date}.latex",
        float_format=lambda x: f"{x:0.2f}",
    )
    # Visualize and compare equity curves
    # long_only_equity = long_only_stats._equity_curve.Equity
    # short_only_equity = short_only_stats._equity_curve.Equity
    up_down_equity = up_down_stats._equity_curve.Equity
    data = pd.DataFrame(
        {
            "btc": dataframe.Close.resample(args.resample).mean(),
            # "long": long_only_equity.resample(args.resample).mean(),
            # "short": short_only_equity.resample(args.resample).mean(),
            "longshort": up_down_equity.resample(args.resample).mean(),
        }
    ).rebase()

    ax = data.plot(figsize=(10, 7))
    if args.show_plots:
        ax.figure.show()
    ax.figure.savefig(
        f"{OUT_DIR}/equities {RUN_ID}.{args.start_date}-{args.end_date}.png"
    )
