# MNIST training data

MNIST by Yann LeCun and Corinna Cortes, derived from NIST datasets.
Source: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
Documentation and license notice: https://keras.io/api/datasets/mnist/

License: Creative Commons Attribution-ShareAlike 3.0
https://creativecommons.org/licenses/by-sa/3.0/

This redistributed file remains under CC BY-SA 3.0. Changes: the 60,000
training examples were permuted with NumPy `default_rng(42)` and packed
into a gzip-compressed binary file. No pixel values or labels were modified.
The official test set is excluded.

Uncompressed layout:

- First 47,040,000 bytes: 60,000 row-major 28×28 uint8 grayscale images.
- Last 60,000 bytes: uint8 class labels.
- First 50,000 examples: training; final 10,000: validation.

Regenerate with `python export_data.py path/to/mnist.npz`.

