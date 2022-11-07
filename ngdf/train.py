# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import pytorch_lightning.utilities.seed as seed_utils
import wandb
import yaml
from omegaconf import OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from ngdf.dataset import GraspDataModule
from ngdf.networks import Decoder

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")


def init_model(cfg):
    model = Decoder(**cfg.net)
    logging.info("Model loaded.")
    return model


@hydra.main(config_path="config", config_name="train")
def main(cfg):
    if cfg.experiment_yml is not None:
        exp_cfg = OmegaConf.load(cfg.experiment_yml)
        cfg = OmegaConf.merge(cfg, exp_cfg)

    with open(".hydra/command.txt", "w") as f:
        command = "python " + " ".join(sys.argv)
        f.write(command)
    seed_utils.seed_everything(cfg.seed, workers=True)

    dm = GraspDataModule(cfg)
    dm.setup()
    logging.info("Data loaded.")
    model = init_model(cfg)

    # Logging
    logging.info(f"Load path: {cfg.load_path}")

    loggers = [
        pl_loggers.CSVLogger(save_dir=cfg.csv_logs),
        pl_loggers.TensorBoardLogger(save_dir=cfg.tb_logs, default_hp_metric=False),
    ]
    if cfg.net.wandb:
        if not os.path.exists("run_id.yaml"):
            run_id = wandb.util.generate_id()
            with open("run_id.yaml", "w") as f:
                yaml.dump({"id": run_id}, f)
        else:
            with open("run_id.yaml", "r") as f:
                run_id = yaml.load(f, Loader=yaml.FullLoader)["id"]
        loggers.append(
            pl_loggers.WandbLogger(
                id=run_id,
                project=cfg.wdb_project,
                entity=cfg.wdb_entity,
                name=str(Path(os.getcwd()).name),
                config=cfg,
                resume="allow",
            )
        )
    checkpoint_callback = ModelCheckpoint(
        monitor="loss/val",
        save_last=True,
        save_top_k=-1,
        every_n_epochs=cfg.every_n_epochs,
    )
    trainer = pl.Trainer(
        gpus=cfg.gpu,
        logger=loggers,
        max_epochs=cfg.epochs,
        log_every_n_steps=cfg.log_every_n_steps,
        val_check_interval=cfg.val_check_interval,
        check_val_every_n_epoch=cfg.every_n_epochs,
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=cfg.load_path,
        profiler="simple",
        num_sanity_val_steps=cfg.num_sanity_val_steps,
        progress_bar_refresh_rate=cfg.progress_bar_refresh_rate,
    )

    logging.info(f"Process ID {os.getpid()}")
    trainer.fit(model, dm)
    if cfg.net.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
