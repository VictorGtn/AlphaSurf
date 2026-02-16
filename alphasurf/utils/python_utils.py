import os
import errno
from pathlib import Path
import time
from torch.utils.data import DataLoader
from multiprocessing import Pool
from tqdm import tqdm


def collate_wrapper(x):
    return x[0]


def makedirs_path(in_path):
    path = Path(in_path)
    path.parent.mkdir(parents=True, exist_ok=True)


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


def do_all(dataset, num_workers=4, prefetch_factor=100, max_sys=None):
    """
    Given a pytorch dataset, uses pytorch multiprocessing system for easy parallelization.
    :param dataset:
    :param num_workers:
    :param prefetch_factor:
    :return:
    """
    dataloader = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=1,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_wrapper,
    )
    total_success = 0
    t0 = time.time()
    for i, success in enumerate(dataloader):
        if max_sys is not None and i > max_sys:
            break
        total_success += int(success)
        if not i % 10:
            print(
                f"Processed {i + 1}/{len(dataloader)}, in {time.time() - t0:.3f}s, with {total_success} successes"
            )
    print(
        f"Processed {len(dataloader)}, in {time.time() - t0:.3f}s, with {total_success} successes"
    )


def _process_dataset_item(args):
    """Helper for multiprocessing: call dataset's __getitem__."""
    dataset, i = args
    try:
        return dataset[i]
    except Exception as e:
        import traceback

        print(f"Error processing item {i}: {e}\n{traceback.format_exc()}")
        return 0  # Return 0 for failure


def do_all_simple(dataset, num_workers=4, max_sys=None):
    """
    Alternative to do_all using standard Python multiprocessing instead of PyTorch DataLoader.
    Use this when dataset contains subprocess calls (e.g., alpha_complex) to avoid segfaults.

    :param dataset: Dataset with __len__ and __getitem__
    :param num_workers: Number of parallel workers (0 for sequential)
    :param max_sys: Maximum number of items to process
    :return:
    """
    n_items = len(dataset) if max_sys is None else min(len(dataset), max_sys)
    total_success = 0
    t0 = time.time()

    if num_workers == 0:
        # Sequential processing
        for i in tqdm(range(n_items), desc="Processing"):
            success = _process_dataset_item((dataset, i))
            total_success += int(success)
            if not i % 10:
                print(
                    f"Processed {i + 1}/{n_items}, in {time.time() - t0:.3f}s, with {total_success} successes"
                )
    else:
        # Parallel processing with standard multiprocessing
        from multiprocessing import TimeoutError as MPTimeoutError

        with Pool(processes=num_workers) as pool:
            results = []
            args_list = [(dataset, i) for i in range(n_items)]
            # Use apply_async with timeout instead of imap for better timeout control
            async_results = [
                pool.apply_async(_process_dataset_item, (arg,)) for arg in args_list
            ]

            for i, async_result in enumerate(tqdm(async_results, desc="Processing")):
                try:
                    # Wait
                    success = async_result.get(timeout=30)
                except MPTimeoutError:
                    print(f"⏱️ Timeout on item {i} after 30 seconds - skipping")
                    success = 0
                except Exception as e:
                    print(f"❌ Error on item {i}: {e}")
                    success = 0

                total_success += int(success)
                results.append(success)
                if len(results) % 10 == 0:
                    print(
                        f"Processed {len(results)}/{n_items}, in {time.time() - t0:.3f}s, with {total_success} successes"
                    )

    print(
        f"Processed {n_items}, in {time.time() - t0:.3f}s, with {total_success} successes"
    )
