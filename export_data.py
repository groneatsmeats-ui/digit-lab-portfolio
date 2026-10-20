"""Repack the Keras MNIST training split for Digit Lab. Requires NumPy."""
import gzip
from pathlib import Path
import sys
import numpy as np

source = np.load(sys.argv[1])
assert source["x_train"].shape == (60000, 28, 28)
assert source["y_train"].shape == (60000,)
indices = np.random.default_rng(42).permutation(60000)
raw = source["x_train"][indices].astype("uint8").tobytes()
raw += source["y_train"][indices].astype("uint8").tobytes()
target = Path(__file__).resolve().parent / "mnist-training.bin.gz"
target.write_bytes(gzip.compress(raw, mtime=0))
print(f"Wrote {target} ({target.stat().st_size:,} bytes)")

