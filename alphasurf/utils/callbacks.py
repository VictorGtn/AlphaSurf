from pathlib import Path

import torch
import wandb

# 3p
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger


def add_wandb_logger(loggers, projectname, runname):
    # init logger
    wand_id = wandb.util.generate_id()
    tb_logger = loggers[-1]
    tags = []
    Path(tb_logger.log_dir).absolute().mkdir(parents=True, exist_ok=True)
    wandb_logger = WandbLogger(
        project=projectname,
        name=runname,
        tags=tags,
        version=Path(tb_logger.log_dir).stem,
        id=wand_id,
        save_dir=tb_logger.log_dir,
        log_model=False,
    )
    loggers += [wandb_logger]


class CommandLoggerCallback(Callback):
    def __init__(self, command):
        self.command = command

    def setup(self, trainer, pl_module, stage):
        tensorboard = pl_module.loggers[0].experiment
        tensorboard.add_text("Command", self.command)


class GPUMemoryCallback(Callback):
    """Track peak GPU memory usage per batch with component-level breakdown"""

    def __init__(self):
        self.peak_memory_percent = []
        self.component_memory = {
            "after_encoder": [],
            "after_pooling": [],
            "after_topnet": [],
            "surface_encoder": [],
            "graph_encoder": [],
            "total_peak": [],
        }
        self.total_gpu_memory = None
        self.hooks = []
        self.current_batch_memory = {}

    def _get_memory_mb(self):
        """Get current allocated memory in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2  # MB
        return 0

    def _get_memory_percent(self, memory_mb):
        """Convert MB to percentage of total GPU memory"""
        if self.total_gpu_memory is not None:
            memory_gb = memory_mb / 1024
            return (memory_gb / self.total_gpu_memory) * 100
        return 0

    def _register_hooks(self, model):
        """Register forward hooks to track memory at different stages"""

        def make_hook(stage_name):
            def hook(module, input, output):
                if torch.cuda.is_available():
                    memory_mb = self._get_memory_mb()
                    # Keep track of peak memory at this stage
                    if stage_name not in self.current_batch_memory:
                        self.current_batch_memory[stage_name] = memory_mb
                    else:
                        self.current_batch_memory[stage_name] = max(
                            self.current_batch_memory[stage_name], memory_mb
                        )

            return hook

        # Hook into the encoder (tracks memory after surface+graph encoding)
        if hasattr(model, "encoder"):
            hook = model.encoder.register_forward_hook(make_hook("after_encoder"))
            self.hooks.append(hook)

            # Hook into individual encoder blocks to track surface and graph separately
            if hasattr(model.encoder, "blocks"):
                for i, block in enumerate(model.encoder.blocks):
                    # Hook into surface encoder if it exists
                    if (
                        hasattr(block, "surface_encoder")
                        and block.surface_encoder is not None
                    ):
                        if block.surface_encoder != "None":
                            hook = block.surface_encoder.register_forward_hook(
                                make_hook(f"surface_encoder_block_{i}")
                            )
                            self.hooks.append(hook)

                    # Hook into graph encoder if it exists
                    if (
                        hasattr(block, "graph_encoder")
                        and block.graph_encoder is not None
                    ):
                        if block.graph_encoder != "None":
                            hook = block.graph_encoder.register_forward_hook(
                                make_hook(f"graph_encoder_block_{i}")
                            )
                            self.hooks.append(hook)

        # Hook into top_net (tracks memory after head network)
        if hasattr(model, "top_net"):
            hook = model.top_net.register_forward_hook(make_hook("after_topnet"))
            self.hooks.append(hook)

    def on_train_start(self, trainer, pl_module):
        """Get total GPU memory capacity and register hooks"""
        if torch.cuda.is_available():
            self.total_gpu_memory = (
                torch.cuda.get_device_properties(0).total_memory / 1024**3
            )  # GB
            # Register hooks on the model
            self._register_hooks(pl_module.model)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Reset memory tracking at start of batch"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            self.current_batch_memory = {}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log peak memory usage after each batch with component breakdown"""
        if torch.cuda.is_available() and self.total_gpu_memory is not None:
            # Get peak memory for this batch (max during the batch)
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2  # MB
            peak_percent = self._get_memory_percent(peak_memory_mb)

            self.peak_memory_percent.append(peak_percent)
            self.component_memory["total_peak"].append(peak_percent)

            # Track component memory (if available)
            if "after_encoder" in self.current_batch_memory:
                enc_percent = self._get_memory_percent(
                    self.current_batch_memory["after_encoder"]
                )
                self.component_memory["after_encoder"].append(enc_percent)
                pl_module.log(
                    "gpu_memory/after_encoder_percent",
                    enc_percent,
                    on_step=True,
                    on_epoch=False,
                )

            # Track surface encoder memory (max across all blocks)
            surface_encoder_keys = [
                k
                for k in self.current_batch_memory.keys()
                if k.startswith("surface_encoder_block_")
            ]
            if surface_encoder_keys:
                max_surface_mb = max(
                    [self.current_batch_memory[k] for k in surface_encoder_keys]
                )
                surface_percent = self._get_memory_percent(max_surface_mb)
                self.component_memory["surface_encoder"].append(surface_percent)
                pl_module.log(
                    "gpu_memory/surface_encoder_percent",
                    surface_percent,
                    on_step=True,
                    on_epoch=False,
                )

            # Track graph encoder memory (max across all blocks)
            graph_encoder_keys = [
                k
                for k in self.current_batch_memory.keys()
                if k.startswith("graph_encoder_block_")
            ]
            if graph_encoder_keys:
                max_graph_mb = max(
                    [self.current_batch_memory[k] for k in graph_encoder_keys]
                )
                graph_percent = self._get_memory_percent(max_graph_mb)
                self.component_memory["graph_encoder"].append(graph_percent)
                pl_module.log(
                    "gpu_memory/graph_encoder_percent",
                    graph_percent,
                    on_step=True,
                    on_epoch=False,
                )

            if "after_topnet" in self.current_batch_memory:
                topnet_percent = self._get_memory_percent(
                    self.current_batch_memory["after_topnet"]
                )
                self.component_memory["after_topnet"].append(topnet_percent)
                pl_module.log(
                    "gpu_memory/after_topnet_percent",
                    topnet_percent,
                    on_step=True,
                    on_epoch=False,
                )

            # Estimate pooling memory (between encoder and topnet)
            if (
                "after_encoder" in self.current_batch_memory
                and "after_topnet" in self.current_batch_memory
            ):
                # Pooling happens between encoder and topnet, so we track the memory after encoder
                # as a proxy for pooling (since pooling is usually lightweight)
                pool_percent = self._get_memory_percent(
                    self.current_batch_memory["after_encoder"]
                )
                self.component_memory["after_pooling"].append(pool_percent)
                pl_module.log(
                    "gpu_memory/after_pooling_percent",
                    pool_percent,
                    on_step=True,
                    on_epoch=False,
                )

            # Log to console (every 10 batches to avoid spam)
            if batch_idx % 10 == 0:
                print(
                    f"Batch {batch_idx}: Peak GPU Memory: {peak_percent:.1f}%", end=""
                )
                surface_encoder_keys = [
                    k
                    for k in self.current_batch_memory.keys()
                    if k.startswith("surface_encoder_block_")
                ]
                if surface_encoder_keys:
                    max_surface_mb = max(
                        [self.current_batch_memory[k] for k in surface_encoder_keys]
                    )
                    surface_percent = self._get_memory_percent(max_surface_mb)
                    print(f" | Surface Encoder: {surface_percent:.1f}%", end="")
                graph_encoder_keys = [
                    k
                    for k in self.current_batch_memory.keys()
                    if k.startswith("graph_encoder_block_")
                ]
                if graph_encoder_keys:
                    max_graph_mb = max(
                        [self.current_batch_memory[k] for k in graph_encoder_keys]
                    )
                    graph_percent = self._get_memory_percent(max_graph_mb)
                    print(f" | Graph Encoder: {graph_percent:.1f}%", end="")
                if "after_encoder" in self.current_batch_memory:
                    enc_percent = self._get_memory_percent(
                        self.current_batch_memory["after_encoder"]
                    )
                    print(f" | After Encoder: {enc_percent:.1f}%", end="")
                if "after_topnet" in self.current_batch_memory:
                    topnet_percent = self._get_memory_percent(
                        self.current_batch_memory["after_topnet"]
                    )
                    print(f" | After TopNet: {topnet_percent:.1f}%", end="")
                print()

            # Log total peak to tensorboard/wandb
            pl_module.log(
                "gpu_memory/peak_percent", peak_percent, on_step=True, on_epoch=False
            )

    def on_train_epoch_end(self, trainer, pl_module):
        """Log summary statistics at end of epoch"""
        if self.peak_memory_percent and self.total_gpu_memory is not None:
            mean_percent = sum(self.peak_memory_percent) / len(self.peak_memory_percent)
            max_percent = max(self.peak_memory_percent)

            # print("\n=== GPU Memory Summary ===")
            # print(f"Mean Peak GPU Memory per batch: {mean_percent:.1f}%")
            # print(f"Max Peak GPU Memory per batch: {max_percent:.1f}%")

            # Component breakdown
            if self.component_memory["surface_encoder"]:
                mean_surf = sum(self.component_memory["surface_encoder"]) / len(
                    self.component_memory["surface_encoder"]
                )
                max_surf = max(self.component_memory["surface_encoder"])
                # print(f"Surface Encoder - Mean: {mean_surf:.1f}%, Max: {max_surf:.1f}%")
                pl_module.log(
                    "gpu_memory/mean_surface_encoder_percent", mean_surf, on_epoch=True
                )
                pl_module.log(
                    "gpu_memory/max_surface_encoder_percent", max_surf, on_epoch=True
                )

            if self.component_memory["graph_encoder"]:
                mean_graph = sum(self.component_memory["graph_encoder"]) / len(
                    self.component_memory["graph_encoder"]
                )
                max_graph = max(self.component_memory["graph_encoder"])
                # print(f"Graph Encoder - Mean: {mean_graph:.1f}%, Max: {max_graph:.1f}%")
                pl_module.log(
                    "gpu_memory/mean_graph_encoder_percent", mean_graph, on_epoch=True
                )
                pl_module.log(
                    "gpu_memory/max_graph_encoder_percent", max_graph, on_epoch=True
                )

            if self.component_memory["after_encoder"]:
                mean_enc = sum(self.component_memory["after_encoder"]) / len(
                    self.component_memory["after_encoder"]
                )
                max_enc = max(self.component_memory["after_encoder"])
                # print(f"After Encoder (total) - Mean: {mean_enc:.1f}%, Max: {max_enc:.1f}%")
                pl_module.log(
                    "gpu_memory/mean_after_encoder_percent", mean_enc, on_epoch=True
                )
                pl_module.log(
                    "gpu_memory/max_after_encoder_percent", max_enc, on_epoch=True
                )

            if self.component_memory["after_topnet"]:
                mean_topnet = sum(self.component_memory["after_topnet"]) / len(
                    self.component_memory["after_topnet"]
                )
                max_topnet = max(self.component_memory["after_topnet"])
                # print(f"After TopNet - Mean: {mean_topnet:.1f}%, Max: {max_topnet:.1f}%")
                pl_module.log(
                    "gpu_memory/mean_after_topnet_percent", mean_topnet, on_epoch=True
                )
                pl_module.log(
                    "gpu_memory/max_after_topnet_percent", max_topnet, on_epoch=True
                )

            # print(f"Total batches: {len(self.peak_memory_percent)}")
            # print("=" * 30 + "\n")

            # Log to tensorboard/wandb
            pl_module.log("gpu_memory/mean_peak_percent", mean_percent, on_epoch=True)
            pl_module.log("gpu_memory/max_peak_percent", max_percent, on_epoch=True)

            # Reset for next epoch
            self.peak_memory_percent = []
            for key in self.component_memory:
                self.component_memory[key] = []

    def on_train_end(self, trainer, pl_module):
        """Clean up hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
