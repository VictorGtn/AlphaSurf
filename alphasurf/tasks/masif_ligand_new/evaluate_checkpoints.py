"""Evaluate MaSIF-Ligand checkpoints without rerunning training."""

import argparse

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from alphasurf.tasks.masif_ligand_new.datamodule import MasifLigandDataModule
from alphasurf.tasks.masif_ligand_new.pl_model import MasifLigandModule


torch.set_num_threads(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    pl.seed_everything(cfg.seed, workers=True)
    datamodule = MasifLigandDataModule(cfg)
    model = MasifLigandModule(cfg)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[cfg.device],
        logger=False,
        limit_test_batches=cfg.train.limit_test_batches,
    )

    for checkpoint in args.checkpoint:
        print(f"Testing {checkpoint}")
        trainer.test(model, ckpt_path=checkpoint, datamodule=datamodule)


if __name__ == "__main__":
    main()
