import os
from pathlib import Path
from typing import List

import hydra
import omegaconf
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import WandbLogger

from src.common.utils import get_env, log_hyperparameters, PROJECT_ROOT
from src.common.callbacks import BacktestCallback, ShuffleDatasetIndices


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info(f"Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        hydra.utils.log.info(f"Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info(f"Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                save_last=True,
                verbose=cfg.train.model_checkpoints.verbose,
                auto_insert_metric_name=True,
            )
        )

    if (
        "stochastic_weight_averaging" in cfg.train
        and cfg.train.stochastic_weight_averaging.active
    ):
        hydra.utils.log.info(f"Adding callback <StochasticWeightAveraging>")
        callbacks.append(
            StochasticWeightAveraging(
                swa_epoch_start=cfg.train.stochastic_weight_averaging.swa_epoch_start
            )
        )

    if not cfg.train.pl_trainer.fast_dev_run:
        callbacks.append(
            pl.callbacks.RichProgressBar(
                refresh_rate_per_second=cfg.logging.progress_bar_refresh_rate
            )
        )
    callbacks.append(pl.callbacks.RichModelSummary(max_depth=2))

    callbacks.append(ShuffleDatasetIndices())

    callbacks.append(BacktestCallback(cfg))

    return callbacks


def run(cfg: DictConfig) -> None:
    """
    Generic train loop

    :param cfg: run configuration, defined by Hydra in /conf
    """
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run=}>. "
            f"Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.pl_trainer.gpus = 0
        cfg.data.datamodule.num_workers.train = 0
        cfg.data.datamodule.num_workers.val = 0
        cfg.data.datamodule.num_workers.test = 0

        # Switch wandb mode to offline to prevent online logging
        cfg.logging.wandb.mode = "offline"

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)
    hydra.utils.log.info(f"Hydra Run Dir is: {hydra_dir.absolute()}")

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    hydra.utils.log.info(
        f"Train Period:\n {OmegaConf.to_yaml(cfg.dataset_conf.train_period)}"
    )
    hydra.utils.log.info(
        f"Val Period:\n {OmegaConf.to_yaml(cfg.dataset_conf.val_period)}"
    )

    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )

    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    # Logger instantiation/configuration
    wandb_logger = None
    if "wandb" in cfg.logging:
        hydra.utils.log.info(f"Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            **wandb_config,
            tags=cfg.core.tags,
        )
        hydra.utils.log.info(f"W&B is now watching <{cfg.logging.wandb_watch.log}>!")
        wandb_logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )

    # Store the YaML config separately into the wandb dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (Path(wandb_logger.experiment.dir) / "hparams.yaml").write_text(yaml_conf)

    hydra.utils.log.info(f"Instantiating the Trainer")

    default_root_dir = (
        wandb_logger.experiment.dir if wandb_logger is not None else hydra_dir
    )
    if "profiler" in cfg.train.pl_trainer:
        profiler = pl.profiler.AdvancedProfiler(dirpath="/tmp", filename="profiler.out")
    else:
        profiler = None
    # The Lightning core, the Trainer
    trainer = pl.Trainer(
        default_root_dir=default_root_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        val_check_interval=cfg.logging.val_check_interval,
        profiler=profiler,
        **cfg.train.pl_trainer,
    )
    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    try:
        hydra.utils.log.info(f"Starting training!")
        trainer.fit(model=model, datamodule=datamodule)
    except KeyboardInterrupt:
        hydra.utils.log.info("trainer.fit() interrupted")
        pass

    try:
        hydra.utils.log.info(f"Starting testing with LAST model!")
        testresult_last = trainer.test(model=model, datamodule=datamodule)
        print(testresult_last)
        hydra.utils.log.info(f"Starting testing with BEST model!")
        testresult_best = trainer.test(datamodule=datamodule, ckpt_path="best")
        print(testresult_best)
    except KeyboardInterrupt:
        wandb_logger.experiment.finish()
        pass

    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb_logger.experiment.finish()
        wandb_logger.save()


if __name__ == "__main__":
    config_name = get_env("CONFIG_NAME", "default")

    @hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name=config_name)
    def main(cfg: omegaconf.DictConfig):
        print(OmegaConf.to_yaml(cfg))
        run(cfg)

    main()
