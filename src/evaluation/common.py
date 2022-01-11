from sys import argv

from src.common.utils import get_hydra_cfg, get_model


DEVICE = "cuda"

ENTITY, PROJECT, RUN_ID = "erpi", "dlai-stonks-new", "wandb_run_id"


def move_dict(d: dict, device=DEVICE) -> dict:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in d.items()}


def load_model_checkpoint(model, checkpoint_path):
    return model.load_from_checkpoint(checkpoint_path=str(checkpoint_path))


def get_cfg_model(checkpoint_path=None, run_dir=None):
    if checkpoint_path is not None and run_dir is not None:
        if not "files" in str(run_dir):
            run_dir = f"{run_dir}/files"
        cfg = get_hydra_cfg(config_path=str(run_dir), config_name="hparams")
        model = load_model_checkpoint(get_model(cfg), checkpoint_path=checkpoint_path)
    else:
        cfg = get_hydra_cfg(overrides=argv[1:])
        model = get_model(cfg)
    print("Got Model and config!")
    return cfg, model.to(DEVICE).eval()
