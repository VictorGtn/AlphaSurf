#!/usr/bin/env python3
"""
Test inference.py without a real checkpoint.

Composes the Hydra config, creates a model with random weights, saves a
temporary checkpoint, then runs both 'embed' and 'interact' on a sample PDB.

Usage:
    python tests/manual/test_inference.py [--pdb path/to/protein.pdb]
"""

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.append(str(REPO_ROOT))

import torch
from alphasurf.tasks.pinder_pair.pl_model import PinderPairModule
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONF_DIR = str(REPO_ROOT / "alphasurf" / "tasks" / "pinder_pair" / "conf")
PINDER_PDB_DIR = str(REPO_ROOT / "data" / "pinder-pair" / "pdb")
SAMPLE_PDB_R = os.path.join(PINDER_PDB_DIR, "12as__A1_P00963--12as__B1_P00963_R.pdb")
SAMPLE_PDB_L = os.path.join(PINDER_PDB_DIR, "12as__A1_P00963--12as__B1_P00963_L.pdb")


def build_dummy_ckpt():
    """Compose Hydra config and save a checkpoint with random weights."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(config_name="config")
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    model = PinderPairModule(cfg)
    tmpdir = tempfile.mkdtemp(prefix="pinder_test_")
    ckpt_path = os.path.join(tmpdir, "dummy.ckpt")
    import pytorch_lightning as pl

    torch.save(
        {
            "pytorch-lightning_version": pl.__version__,
            "hyper_parameters": {"cfg": cfg},
            "state_dict": model.state_dict(),
            "global_step": 0,
            "epoch": 0,
        },
        ckpt_path,
    )
    print(f"[test] Saved dummy checkpoint to {ckpt_path}")
    return ckpt_path, cfg


def test_embed(ckpt_path, pdb_path):
    import argparse

    from alphasurf.tasks.pinder_pair.inference import cmd_embed

    args = argparse.Namespace(
        ckpt=ckpt_path,
        pdb=pdb_path,
        data_dir=None,
        output=os.path.join(os.path.dirname(ckpt_path), "test_embed.pt"),
    )
    print("\n[test] === EMBED MODE ===")
    cmd_embed(args)
    assert os.path.exists(args.output), f"Embed output missing: {args.output}"
    print(f"[test] embed OK -> {args.output}")


def test_interact(ckpt_path, pdb_r, pdb_l):
    import argparse

    from alphasurf.tasks.pinder_pair.inference import cmd_interact

    args = argparse.Namespace(
        ckpt=ckpt_path,
        pdb_r=pdb_r,
        pdb_l=pdb_l,
        data_dir=None,
        output=os.path.join(os.path.dirname(ckpt_path), "test_interact.pt"),
    )
    print("\n[test] === INTERACT MODE ===")
    cmd_interact(args)
    assert os.path.exists(args.output), f"Interact output missing: {args.output}"
    print(f"[test] interact OK -> {args.output}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test inference.py with random weights"
    )
    parser.add_argument("--pdb", default=None, help="PDB for embed test")
    parser.add_argument("--pdb-r", default=None, help="Receptor PDB for interact test")
    parser.add_argument("--pdb-l", default=None, help="Ligand PDB for interact test")
    args = parser.parse_args()

    pdb = args.pdb or SAMPLE_PDB_R
    pdb_r = args.pdb_r or SAMPLE_PDB_R
    pdb_l = args.pdb_l or SAMPLE_PDB_L

    ckpt_path, cfg = build_dummy_ckpt()

    test_embed(ckpt_path, pdb)
    test_interact(ckpt_path, pdb_r, pdb_l)

    print("\n[test] ALL TESTS PASSED")


if __name__ == "__main__":
    main()
