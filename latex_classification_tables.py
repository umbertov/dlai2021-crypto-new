import pandas as pd
from pathlib import Path
from collections import defaultdict

MODES = ["train", "val", "test"]

METRICS_PATH = Path("./evaluation/classification_metrics/")

## # dictionary str -> str -> float
## #            run_id -> metric name -> metric value
## METRICS_DICT = defaultdict(dict)
## for csv_name in METRICS_PATH.rglob("*.csv"):
##     run_id, mode, _ = csv_name.name.split(".")
##     df = pd.read_csv(csv_name)
##     df.index = df.metric
##     METRICS_DICT[run_id][mode] = df.ALL.to_dict()

METRICS_DICT = {mode: pd.DataFrame() for mode in MODES}
for csv_name in METRICS_PATH.rglob("*.csv"):
    run_id, mode, _ = csv_name.name.split(".")
    df = pd.read_csv(csv_name)
    df.index = df.metric.apply(str.title)
    METRICS_DICT[mode][run_id] = df.ALL

METRICS_DICT = {
    k: df[~df.index.str.contains("Masked")] for k, df in METRICS_DICT.items()
}

for mode in METRICS_DICT:
    print(mode)
    df = METRICS_DICT[mode]
    print(df.to_latex(float_format=lambda x: f"{x*100:.1f} %"))
