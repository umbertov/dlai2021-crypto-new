if __name__ == "__main__":
    import sys
    from src.common.utils import get_hydra_cfg, get_datamodule, get_model

    cfg = get_hydra_cfg(overrides=sys.argv[1:])

    model = get_model(cfg)
    datamodule = get_datamodule(cfg)

    batch = next(iter(datamodule.train_dataloader()))
    model_out = model(**batch)
