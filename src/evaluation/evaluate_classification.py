from argparse import ArgumentParser
from pathlib import Path
from omegaconf.dictconfig import DictConfig
import torch
from src.ui.ui_utils import get_run_dir
from src.common.utils import (
    compute_classification_metrics,
    get_datamodule,
)
from src.lightning_modules.sequence_tagging import TimeSeriesClassifier

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
    global PROJECT, RUN_ID
    parser = ArgumentParser()
    parser.add_argument("--use-split", default="test")
    parser.add_argument("--data-path", default=None, type=str)
    parser.add_argument("--run-id", default=RUN_ID, type=str)
    parser.add_argument("--project", default=PROJECT, type=str)
    parser.add_argument("--backtest-length-pct", default=None, type=float)
    parser.add_argument("--backtest-start-pct", default=None, type=float)
    args = parser.parse_args()
    assert args.use_split in ("train", "val", "test")
    return args


def compute_metrics_on_dataset(dataset, model):
    dataset.reset()
    megabatch = dataloader.collate_fn(list(dataset))
    megabatch = move_dict(megabatch, "cuda")
    all_predictions = model.predict(**megabatch).squeeze()
    all_targets = megabatch["categorical_targets"].squeeze()

    metrics = compute_classification_metrics(
        all_predictions.view(-1),
        all_targets.view(-1),
        num_classes=cfg.dataset_conf.n_classes,
    )

    # Now compute metrics just for {down, up} classes
    mask = all_targets != 1  # ignore the "neutral" class

    true_positives = all_predictions == all_targets
    masked_accuracy = true_positives[mask].sum() / true_positives[mask].numel()

    # mask pred and target tensors
    masked_predictions, masked_targets = all_predictions[mask], all_targets[mask]
    # problem: we still have "neutral" predictions in masked_predictions!
    # solution: use a null label and specify ignore_index later
    masked_predictions[masked_predictions == 1] = 3
    # adjust the class index for "up" class in both preds and targets
    masked_predictions[masked_predictions == 2] = 1
    masked_predictions[masked_predictions == 3] = 2
    masked_targets[masked_targets == 2] = 1
    # finally compute metrics
    masked_metrics = compute_classification_metrics(
        masked_predictions.view(-1),
        masked_targets.view(-1),
        num_classes=3,
        ignore_index=2,
    )
    # change the name of metric keys so they don't clash with the others we are returning
    masked_metrics = {f"{k}_masked": v[[0, 1]] for k, v in masked_metrics.items()}
    return {**metrics, **masked_metrics}


DATA_ROOT = "${oc.env:PROJECT_ROOT}/data"

if __name__ == "__main__":

    args = parse_args()
    PROJECT, RUN_ID = args.project, args.run_id

    run_dir: Path = get_run_dir(entity=ENTITY, project=PROJECT, run_id=RUN_ID)
    checkpoint_paths: list[Path] = list(run_dir.rglob("checkpoints/*"))

    # Get Hydra cfg and load checkpoint from specific wandb run
    cfg, model = get_cfg_model(checkpoint_path=checkpoint_paths[0], run_dir=run_dir)
    # Optionally specify alternative data path
    if args.data_path is not None:
        cfg.dataset_conf.data_path.data_path = f"{DATA_ROOT}/{args.data_path}"

    # obtain the data for evaluation
    datamodule = get_datamodule(
        cfg, "fit" if args.use_split in ("train", "val") else "test"
    )
    # decide which split to use
    if args.use_split == "train":
        dataset = datamodule.train_dataset
        dataloader = datamodule.train_dataloader()
    elif args.use_split == "val":
        dataset = datamodule.val_datasets[0]
        dataloader = datamodule.val_dataloader()[0]
    elif args.use_split == "test":
        dataset = datamodule.test_datasets[0]
        dataloader = datamodule.test_dataloader()[0]
    else:
        raise Exception("wrong args")

    tickers2datasets = {d.name: d for d in dataset.datasets}

    global_metrics = compute_metrics_on_dataset(dataset, model)

    metrics = {"ALL": global_metrics}
    # for ticker, dataset in tickers2datasets.items():
    #     metrics[ticker] = compute_metrics_on_dataset(dataset, model)

    for dataset_name in metrics:
        print("[ ", dataset_name[:8], "]")
        print(
            *[
                f'  - {name.split("/")[1]}: {val}'
                for name, val in metrics[dataset_name].items()
            ],
            sep="\n",
        )
        print("\n")
