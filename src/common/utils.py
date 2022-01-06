import os
from pathlib import Path
from typing import Dict, List, Optional

import dotenv
import pytorch_lightning as pl
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.global_hydra import GlobalHydra

from torch.nn import functional as F
from torchmetrics import functional as M


def notnone(x):
    return x is not None


def try_or_default(f, default=None, exception=Exception):
    def apply(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except exception:
            return None

    return apply


def get_hydra_cfg(config_path=None, overrides=[], config_name="default"):
    GlobalHydra.instance().clear()
    if config_path is None:
        config_path = str(PROJECT_ROOT / "conf")
    hydra.initialize_config_dir(config_dir=config_path)
    cfg = hydra.compose(config_name=config_name, overrides=overrides)
    return cfg


# def get_hydra_cfg(config_name: str = "default", overrides=[]) -> DictConfig:
#     """
#     Instantiate and return the hydra config -- streamlit and jupyter compatible
#
#     Args:
#         config_name: .yaml configuration name, without the extension
#
#     Returns:
#         The desired omegaconf.DictConfig
#     """
#     register_custom_resolvers()
#     GlobalHydra.instance().clear()
#     hydra.experimental.initialize_config_dir(config_dir=config_path)
#     return compose(config_name='default, overrides=overrides)


def get_datasets(cfg: DictConfig):
    train_dataset, val_dataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets, _recursive_=True
    )
    return train_dataset, val_dataset


def get_model(cfg):
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )
    return model


def get_datamodule(cfg):
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)
    datamodule.setup()
    return datamodule


# def get_hydra_cfg(config_name: str = "default", overrides=[]) -> DictConfig:
#     """
#     Instantiate and return the hydra config -- streamlit and jupyter compatible
#
#     Args:
#         config_name: .yaml configuration name, without the extension
#
#     Returns:
#         The desired omegaconf.DictConfig
#     """
#     GlobalHydra.instance().clear()
#     hydra.experimental.initialize_config_dir(config_dir=str(PROJECT_ROOT / "conf"))
#     return compose(config_name=config_name, overrides=overrides)


def register_custom_resolvers():
    if not OmegaConf.has_resolver("length"):
        OmegaConf.register_new_resolver("length", lambda x: len(x))
    if not OmegaConf.has_resolver("lengthMinusOne"):
        OmegaConf.register_new_resolver("lengthMinusOne", lambda x: len(x) - 1)
    if not OmegaConf.has_resolver("last"):
        OmegaConf.register_new_resolver("last", lambda x: x[-1])


register_custom_resolvers()


def wandb_to_hydra_conf(cfg: DictConfig) -> DictConfig:
    """
    Takes a DictConfig object obtained by the config.yaml file
    stored in the wandb run dir.
    This is because wandb stores configs in flat structures,
    with keys like:
        top_key/nested_key1/.../nested_keyN => {'value': value, 'desc':None }
    rather than:
        top_key.nested_key1.[...].nested_keyN => value
    """
    cfg = dict(cfg)
    items = [(k, v) for k, v in cfg.items() if "/" in k]
    for slashed_key, value in items:
        keys = slashed_key.split("/")
        c = cfg
        for new_key in keys[:-1]:
            if not new_key in c:
                c[new_key] = {}
            c = c[new_key]
        c[keys[-1]] = value["value"]

    for k, _ in items:
        del cfg[k]
    for k in (k for k in list(cfg.keys()) if "_wandb" in k):
        del cfg[k]

    return OmegaConf.create(cfg)


def get_env(env_name: str, default: Optional[str] = None) -> str:
    """
    Safely read an environment variable.
    Raises errors if it is not defined or it is empty.

    :param env_name: the name of the environment variable
    :param default: the default (optional) value for the environment variable

    :return: the value of the environment variable
    """
    if env_name not in os.environ:
        if default is None:
            raise KeyError(f"{env_name} not defined and no default value is present!")
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            raise ValueError(
                f"{env_name} has yet to be configured and no default value is present!"
            )
        return default

    return env_value


def load_envs(env_file: Optional[str] = None) -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


STATS_KEY: str = "stats"


# Adapted from https://github.com/hobogalaxy/lightning-hydra-template/blob/6bf03035107e12568e3e576e82f83da0f91d6a11/src/utils/template_utils.py#L125
def log_hyperparameters(
    cfg: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionally saves:
        - sizes of train, val, test dataset
        - number of trainable model parameters
    Args:
        cfg (DictConfig): [description]
        model (pl.LightningModule): [description]
        trainer (pl.Trainer): [description]
    """
    hparams = OmegaConf.to_container(cfg, resolve=True)

    # save number of model parameters
    hparams[f"{STATS_KEY}/params_total"] = sum(p.numel() for p in model.parameters())
    hparams[f"{STATS_KEY}/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams[f"{STATS_KEY}/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # (this is just a trick to prevent trainer from logging hparams of model, since we already did that above)
    trainer.logger.log_hyperparams = lambda params: None


# Load environment variables
load_envs()

# Set the cwd to the project root
PROJECT_ROOT: Path = Path(get_env("PROJECT_ROOT"))
assert (
    PROJECT_ROOT.exists()
), "You must configure the PROJECT_ROOT environment variable in a .env file!"

os.chdir(PROJECT_ROOT)


def compute_classification_metrics(preds, targets, num_classes):
    return {
        "metrics/accuracy": M.accuracy(
            preds, targets, num_classes=2, average="macro", ignore_index=1
        ),
        "metrics/precision": M.precision(
            preds, targets, num_classes=num_classes, average="macro"
        ),
        "metrics/recall": M.recall(
            preds, targets, num_classes=num_classes, average="macro"
        ),
        "metrics/f1": M.f1(preds, targets, num_classes=num_classes, average="macro"),
    }


def compute_confusion_matrix(predictions, targets, num_classes):
    return M.confusion_matrix(
        predictions.view(-1),
        targets.view(-1),
        num_classes=num_classes,
    )
