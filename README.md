# Digit Lab — Portfolio Edition

A separate copy of [Digit Lab](https://github.com/groneatsmeats-ui/digit-lab), themed to match Lucas Erikson's personal website. The original repository is unchanged.

Live project: https://groneatsmeats-ui.github.io/digit-lab-portfolio/

The presentation uses the portfolio's Manrope and DM Sans fonts, light gray surfaces, and blue accents. Training, the MNIST dataset, the WebAssembly backend, and handwriting recognition are retained from the original.

Theme styles are in `portfolio-theme.css`. Serve this directory over HTTP; no build step or package installation is needed. Enable GitHub Pages from the main branch root.

---

# Digit Lab

A classroom neural-network playground for UConn AI. Start with two hidden
neurons, add layers, change activation functions, and test the resulting model
on MNIST and your own handwriting.

## Run

Open the GitHub Pages site in a current Chrome, Edge, Firefox, or Safari.
No Python, account, API key, or backend is required for students.

To preview a checkout locally:

```sh
python -m http.server 8080
```

Then visit http://localhost:8080. Opening `index.html` as a `file://` URL
will not work because workers and data fetching need HTTP.

## Classroom activity

1. Train the initial two-neuron network.
2. Change a hidden layer's neuron count, add a layer, or change its activation.
3. Train again and compare validation accuracy.
4. Draw a digit in the sketchpad and click **Guess digit**.
5. Explain one experiment and export the results as CSV.

Each browser has its own model and history. Drawings never leave the browser.
Reloading discards the model; train again to enable sketchpad predictions.
The last 30 results persist in browser local storage when available.

## What runs

- TensorFlow.js 4.22.0 with the single-threaded WebAssembly backend.
- Training runs in a Web Worker so controls and page scrolling stay responsive.
- Fixed 50,000/10,000 training/validation split from MNIST's original 60,000
  training images. The official 10,000-image test set is not bundled or scored.
- Five epochs; batch size 32; Adam at 0.001; 20% dropout after the last
  hidden layer; categorical cross-entropy and ten softmax outputs.
- Dense layers use seeded Glorot initialization. Data is pre-shuffled with
  NumPy `default_rng(42)` and traversed in fixed order each epoch.
- 1–6 hidden layers, 1–256 neurons each; ReLU, sigmoid, or tanh.
- The sketchpad crops ink, scales proportionally to fit 20×20, then centers
  the ink's center of mass inside 28×28. Predictions use the last completed
  model, even if the architecture controls have since changed.

The initial download includes about 10 MB of compressed digits and the
locally bundled browser runtime. Training time depends on device and model size.
Use laptops for the workshop; larger architectures need more memory and time.
Browser numerical results and timing differ from the earlier Python version.
Do not compare their scores as if they were identical experiments.

## Hosting on GitHub Pages

In repository Settings → Pages, choose **Deploy from a branch**, then
**main** and **/(root)**. No build workflow is needed. Relative asset URLs work
on a project site such as `/digit-lab/`.

## Files

- `index.html`: interface, network controls, history, sketchpad preprocessing.
- `browser-engine.js`: worker communication and local history.
- `trainer.js`: real training and inference.
- `mnist-training.bin.gz`: fixed split, pixels then labels; see data README.
- Bundled runtime files: pinned TensorFlow.js and WebAssembly runtime plus license.
- `export_data.py`: reproducible conversion from Keras MNIST.

## Attribution

Inspired by the TensorFlow Authors'
[beginner quickstart](https://www.tensorflow.org/tutorials/quickstart/beginner).
TensorFlow.js is Apache-2.0 licensed; see `LICENSE-tensorflow.txt`.
MNIST is credited and licensed separately in `DATA.md`.

