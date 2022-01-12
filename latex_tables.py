import pandas as pd
from pathlib import Path
from collections import defaultdict

MODES = ["train", "val", "test"]

METRICS_PATH = Path("./evaluation")

# dictionary str -> str -> float
#            run_id -> metric name -> metric value
METRICS_DICT = defaultdict(dict)

for csv_name in METRICS_PATH.rglob("*.csv"):
    run_id, mode, _ = csv_name.name.split(".")
    df = pd.read_csv(csv_name)
    df.index = df.metric
    METRICS_DICT[run_id][mode] = df.ALL.to_dict()

for run_id, mode2metrics in METRICS_DICT.items():
    print(run_id)
    for mode in MODES:
        for metric_name, metric_val in mode2metrics[mode].items():
            if "masked" in metric_name:
                continue
            print(f"    & {metric_val*100:.1f}\\% % {mode} {metric_name}")
    print("\\\\")

    print("%" * 10)
