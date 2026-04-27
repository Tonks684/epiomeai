"""Print the structure and summary statistics of the HDF5 data file."""
import argparse

import h5py
import numpy as np

from src.data import TASK_GROUPS, task_summary


def main() -> None:
    """Print the structure, shapes, and task summaries of the HDF5 data file."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', required=True)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  HDF5 file: {args.h5}")
    print('='*60)

    with h5py.File(args.h5, 'r') as f:
        for name, obj in f.items():
            if isinstance(obj, h5py.Dataset):
                arr = obj[:]
                print(f"\nDataset : {name}")
                print(f"  shape : {arr.shape}  dtype: {arr.dtype}")
                if arr.dtype.kind == 'f':
                    print(f"  range : [{arr.min():.4f}, {arr.max():.4f}]  mean: {arr.mean():.4f}")
                    print(f"  NaNs  : {np.isnan(arr).sum()}")

    print(f"\n{'='*60}")
    print("  Task summaries")
    print('='*60)
    for task in ('task1', 'task2'):
        s = task_summary(args.h5, task)
        print(f"\n  {task}")
        g = TASK_GROUPS[task]
        print(f"  Labels : 0 = {g['label_names'][0]}, 1 = {g['label_names'][1]}")
        print(f"  n(0)   : {s['n_negative']}")
        print(f"  n(1)   : {s['n_positive']}")
        print(f"  total  : {s['n_total']}")
        print(f"  +rate  : {s['positive_rate']:.1%}")

    print()


if __name__ == '__main__':
    main()
