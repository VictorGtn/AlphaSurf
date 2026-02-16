import time

import numpy as np
import torch

# Import pytorch_lightning for callback base class
try:
    import pytorch_lightning as pl

    _PL_AVAILABLE = True
except ImportError:
    _PL_AVAILABLE = False

# Global configuration for timing collection
# 'enabled': Master switch
# 'global': Coarse-grained timing (separate forward and backward passes)
# 'internal': Fine-grained timing (specific operations via time_operation)
# 'total': Combined Forward + Backward pass timing (minimal synchronization)
TIMING_FLAGS = {"enabled": False, "global": False, "internal": False, "total": False}

# Global state for timing
_backward_start_time = None
_backward_metadata = None
_forward_start_time = None
_forward_metadata = None
_total_start_time = None
_total_metadata = None


def enable_timing(category="total"):
    """
    Enable timing collection for a single category, or disable with 'none'.

    Args:
        category (str): Timing category to enable.
            Options: 'global', 'internal', 'total', 'none'.
            'global': Separate Forward and Backward stats.
            'internal': Detailed manual blocks.
            'total': Combined Forward+Backward stats (one sync).
            'none': Disable all timing (same as disable_timing()).
            Defaults to 'total'.
    """
    global TIMING_FLAGS

    # Handle 'none' as a way to disable timing via config
    if category == "none":
        disable_timing()
        return

    TIMING_FLAGS["enabled"] = True

    # Reset all specific flags
    TIMING_FLAGS["global"] = False
    TIMING_FLAGS["internal"] = False
    TIMING_FLAGS["total"] = False

    valid_categories = {"global", "internal", "total"}

    if category not in valid_categories:
        raise ValueError(
            f"Invalid timing category: {category}. Must be one of {valid_categories} or 'none'"
        )

    TIMING_FLAGS[category] = True


def disable_timing():
    """Disable all timing collection."""
    global TIMING_FLAGS
    TIMING_FLAGS["enabled"] = False
    TIMING_FLAGS["global"] = False
    TIMING_FLAGS["internal"] = False
    TIMING_FLAGS["total"] = False


def is_timing_enabled(category=None):
    """
    Check if timing is enabled, optionally for a specific category.

    Args:
        category (str, optional): 'global', 'internal', or 'total'.
                                  If None, checks master switch.
    """
    if not TIMING_FLAGS["enabled"]:
        return False
    if category is None:
        return True
    return TIMING_FLAGS.get(category, False)


class TimingStats:
    """Collect timing statistics for different operations."""

    def __init__(self):
        self.timings = {}
        self.reset()

    def reset(self):
        """Reset all timing statistics."""
        self.timings = {
            "curvatures_keops": [],
            "curvatures_torch": [],
            "dmasif_conv_keops": [],
            "dmasif_conv_sparse": [],
            "dmasif_conv_dense": [],
            "forward_pass": [],
            "backward_pass": [],
            "train_step": [],  # Combined Forward + Backward
        }

    def add_timing(self, operation, duration, metadata=None):
        """Add a timing measurement for an operation."""
        if not TIMING_FLAGS["enabled"]:
            return
        if operation not in self.timings:
            self.timings[operation] = []
        self.timings[operation].append((duration, metadata))

    def get_stats(self, operation):
        """Get mean and std dev for an operation, including metadata stats."""
        if operation not in self.timings or not self.timings[operation]:
            return None, None, 0, None, None

        durations = [t[0] for t in self.timings[operation]]
        mean_time = np.mean(durations)
        std_time = np.std(durations)
        count = len(durations)

        # Extract points information from metadata if available
        points_list = []
        for duration, metadata in self.timings[operation]:
            if metadata and "points" in metadata:
                points_list.append(metadata["points"])

        if points_list:
            mean_points = np.mean(points_list)
            std_points = np.std(points_list)
        else:
            mean_points = None
            std_points = None

        return mean_time, std_time, count, mean_points, std_points

    def print_epoch_stats(self):
        """Print timing statistics for the current epoch."""
        print("\n=== Timing Statistics ===")
        for operation in sorted(self.timings.keys()):
            mean_time, std_time, count, mean_points, std_points = self.get_stats(
                operation
            )
            if count > 0:
                operation_name = operation.replace("_", " ").title()
                stats_str = f"{operation_name}: {mean_time:.6f} ± {std_time:.6f}s ({count} calls)"
                if mean_points is not None:
                    stats_str += f" | {mean_points:.0f} ± {std_points:.0f} points"
                print(stats_str)
        print("=" * 25)


# Global timing stats instance
timing_stats = TimingStats()


def reset_timing_stats():
    """Reset timing statistics for a new epoch."""
    timing_stats.reset()


def print_timing_stats():
    """Print current timing statistics."""
    if TIMING_FLAGS["enabled"]:
        timing_stats.print_epoch_stats()


def time_operation(operation_name, metadata=None):
    """
    Context manager for timing operations with GPU synchronization.
    This is considered 'internal' timing.
    """

    class TimingContext:
        def __init__(self, op_name, meta):
            self.operation_name = op_name
            self.metadata = meta
            self.start_time = None

        def __enter__(self):
            # Only run if both master enabled and internal timing enabled
            if is_timing_enabled("internal"):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.start_time = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time is not None and is_timing_enabled("internal"):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                duration = time.perf_counter() - self.start_time
                timing_stats.add_timing(self.operation_name, duration, self.metadata)

    return TimingContext(operation_name, metadata)


def start_forward_timing(metadata=None):
    """
    Start timing the forward pass (Global).
    """
    global _forward_start_time, _forward_metadata
    if is_timing_enabled("global"):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _forward_start_time = time.perf_counter()
        _forward_metadata = metadata


def end_forward_timing():
    """End timing the forward pass (Global)."""
    global _forward_start_time, _forward_metadata
    if is_timing_enabled("global") and _forward_start_time is not None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.perf_counter() - _forward_start_time
        timing_stats.add_timing("forward_pass", duration, _forward_metadata)
        _forward_start_time = None
        _forward_metadata = None


def start_backward_timing(metadata=None):
    """
    Start timing the backward pass (Global or Internal).
    """
    global _backward_start_time, _backward_metadata
    if is_timing_enabled("global") or is_timing_enabled("internal"):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _backward_start_time = time.perf_counter()
        _backward_metadata = metadata


def end_backward_timing():
    """End timing the backward pass (Global or Internal)."""
    global _backward_start_time, _backward_metadata
    if (
        is_timing_enabled("global") or is_timing_enabled("internal")
    ) and _backward_start_time is not None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.perf_counter() - _backward_start_time
        timing_stats.add_timing("backward_pass", duration, _backward_metadata)
        _backward_start_time = None
        _backward_metadata = None


def start_total_timing(metadata=None):
    """
    Start timing the combined train step (Total: Forward + Backward).
    """
    global _total_start_time, _total_metadata
    if is_timing_enabled("total"):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _total_start_time = time.perf_counter()
        _total_metadata = metadata


def end_total_timing():
    """End timing the combined train step (Total: Forward + Backward)."""
    global _total_start_time, _total_metadata
    if is_timing_enabled("total") and _total_start_time is not None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.perf_counter() - _total_start_time
        timing_stats.add_timing("train_step", duration, _total_metadata)
        _total_start_time = None
        _total_metadata = None


class TimingStatsCallback(pl.callbacks.Callback if _PL_AVAILABLE else object):
    """
    PyTorch Lightning callback for collecting and printing timing statistics.
    Integrates forward, backward, and total timing with PyTorch Lightning's training loop.
    """

    def on_train_epoch_start(self, trainer, pl_module):
        """Reset timing statistics at the start of each training epoch."""
        reset_timing_stats()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Start timing the forward/total pass at the beginning of the batch."""
        metadata = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "batch_idx": batch_idx,
        }
        start_forward_timing(metadata)
        start_total_timing(metadata)

    def on_before_backward(self, trainer, pl_module, loss):
        """
        End timing the forward pass and start timing the backward pass.
        Note: on_before_backward is called after forward/loss computation but before backward.
        """
        # Forward pass ends here (conceptually, or at least the part we want to time including loss)
        end_forward_timing()

        metadata = {"epoch": trainer.current_epoch, "global_step": trainer.global_step}
        start_backward_timing(metadata)

    def on_after_backward(self, trainer, pl_module):
        """End timing the backward and total pass."""
        end_backward_timing()
        end_total_timing()

    def on_train_epoch_end(self, trainer, pl_module):
        """Print timing statistics at the end of each training epoch."""
        print_timing_stats()


def create_timing_callback():
    """
    Create a PyTorch Lightning callback for timing statistics.

    Returns:
        TimingStatsCallback: Configured callback for timing collection
    """
    return TimingStatsCallback()
