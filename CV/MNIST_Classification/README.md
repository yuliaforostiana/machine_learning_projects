# Handwritten Digit Classification with a CNN (PyTorch)

## 📌 Overview

This project builds and evaluates a convolutional neural network (CNN) for handwritten digit classification on the MNIST dataset, implementing the well-known TinyVGG architecture (as featured on CNN Explainer) from scratch in PyTorch. Beyond training a single model, the project benchmarks CPU vs. GPU training time, performs a detailed error analysis (confusion matrix, misclassified examples), and runs a set of targeted experiments on `nn.Conv2d` hyperparameters to build intuition for how kernel size, stride, padding, and channel count each affect a convolutional layer's output.

## 🎯 Problem Statement

Classifying handwritten digits is a foundational computer vision task, but building a CNN that works well requires understanding not just how to define the architecture, but *why* it works — how convolution and pooling layers transform an image, how architectural choices affect output dimensions, and where a trained model's mistakes come from (genuine ambiguity in the data vs. real modeling gaps). This project treats each of these as something to investigate empirically, not just implement.

## 🎯 Goal

- Implement `TinyVGG` (the CNN architecture from CNN Explainer) from scratch in PyTorch and train it on MNIST.
- Quantify the practical training speedup from GPU vs. CPU on identical architecture and epoch count.
- Evaluate model quality with a confusion matrix and targeted misclassification analysis, distinguishing genuine data ambiguity from real model errors.
- Build hands-on intuition for `nn.Conv2d` hyperparameters (`kernel_size`, `stride`, `padding`, `out_channels`) by directly observing their effect on output tensor shape.

## 🔍 Approach

### 1. Foundational Exploration
- Surveyed real-world computer vision applications across manufacturing (defect detection), agriculture (crop monitoring via drone imagery), defense, and logistics, to ground the technical work in practical context.
- Used the interactive [CNN Explainer](https://poloclub.github.io/cnn-explainer/) tool to visualize how a custom uploaded image is transformed layer-by-layer through a CNN, drawing conclusions about filter specialization, max pooling behavior, and the model's tendency to misclassify out-of-distribution inputs based on superficial visual similarity (e.g., a red sports car being classified similarly to a red bell pepper due to shared color/texture).

### 2. Data Preparation
- Loaded MNIST via `torchvision.datasets.MNIST`, inspected dataset size, image shape, and class labels, and visualized a random sample of training images.
- Wrapped both train and test sets in `DataLoader` objects (`batch_size=32`) and confirmed the resulting batch shapes.

### 3. Model Architecture
- Implemented `MNISTModelV2` (TinyVGG): two convolutional blocks (each with two `Conv2d` + `ReLU` layers followed by `MaxPool2d`), followed by a flattening classifier head.

### 4. Training & Hardware Benchmarking
- Trained the identical architecture for 5 epochs on both CPU and GPU (`cuda`), tracking wall-clock training time for each, using `CrossEntropyLoss` and SGD optimization.
- Evaluated the trained model on the test set (loss and accuracy).

### 5. Prediction Visualization & Error Analysis
- Visualized model predictions against ground truth on a sample of test images, color-coding correct (green) vs. incorrect (red) predictions.
- Built a full confusion matrix (`torchmetrics.ConfusionMatrix` + `mlxtend.plot_confusion_matrix`) across all 10 digit classes.
- Isolated and visualized 9 specific misclassified examples, explicitly assessing for each whether the error reflected genuine ambiguity in the handwriting or a real model shortcoming.

### 6. Convolutional Layer Hyperparameter Experiments
- Passed a random `[1, 3, 64, 64]` tensor through multiple `nn.Conv2d` configurations, systematically varying `kernel_size`, `stride`, `padding` (`"same"` vs. `"valid"`), and `out_channels`, and recorded the resulting output shape for each.

## 📊 Results

### Training performance
- **GPU training was ~3.2x faster** than CPU for the identical architecture and epoch count.
- Training accuracy improved steadily across all 5 epochs, suggesting the model had not yet fully converged and could likely improve further with more epochs.

### Classification quality
- The confusion matrix showed strong diagonal dominance (most predictions correct), with **digits 0 and 1 classified most reliably**, and **digit 5 the most frequently misclassified**.
- The most common confusions were **7 vs. 2/3**, **9 vs. 3**, and **5 vs. 3**.
- Error analysis of specific misclassified examples found a **mixed picture**: some errors (e.g., 6 vs. 1, 5 vs. 7) reflected genuinely ambiguous handwriting that would challenge a human reader too, while others (e.g., 8 vs. 6, 2 vs. 7, 8 vs. 0) involved clearly-formed digits the model still got wrong — indicating real, addressable modeling gaps alongside unavoidable data ambiguity.

### Convolutional hyperparameter effects (on a `[1,3,64,64]` input)
| Change | Effect on output shape |
|---|---|
| `kernel_size` 3→5 with `stride` 1→2 | Output shrank from 62×62 to 30×30 — larger kernel + larger stride both reduce spatial resolution |
| `stride` 1→3 (kernel_size fixed at 3) | Output shrank from 62×62 to 21×21 — larger stride alone reduces output resolution |
| `padding` `"same"`→`"valid"` | Output shrank from 64×64 to 62×62 — `"valid"` doesn't zero-pad the borders, `"same"` does |
| `out_channels` 10→64 | Spatial output size unchanged, but the number of learned filters (and model parameters) increases |

## 🛠️ Tech Stack

- **Language:** Python
- **Deep learning:** PyTorch (`torch.nn`, `nn.Conv2d`, `nn.MaxPool2d`, `nn.CrossEntropyLoss`, `torch.optim.SGD`, `DataLoader`), `torchvision` (`datasets.MNIST`, `ToTensor`)
- **Evaluation:** `torchmetrics` (`ConfusionMatrix`), `mlxtend` (`plot_confusion_matrix`)
- **Data handling & visualization:** `matplotlib`, `tqdm` (training progress), custom `helper_functions` (`accuracy_fn`, `print_train_time`, `eval_model`)

## 📁 Repository Structure

├── cnn_mnist_classification.ipynb   # Full architecture, training, and analysis notebook

├── helper_functions.py               # Shared training/eval utility functions

└── README.md

## 🚀 How to Run

```bash
pip install torch torchvision matplotlib tqdm torchmetrics "mlxtend==0.23.4" "scikit-learn==1.6.1"
jupyter notebook cnn_mnist_classification.ipynb
```

> Note: MNIST is downloaded automatically via `torchvision.datasets.MNIST(download=True)` on first run — no manual dataset download required. A CUDA-capable GPU is optional but needed to reproduce the CPU vs. GPU training time comparison.

## 📈 Next Steps

- Train for additional epochs given the steadily improving accuracy trend observed, to check whether performance continues to improve or plateaus.
- Investigate the specific confusion pairs (7/2/3, 9/3, 5/3) further — e.g., with targeted data augmentation or additional convolutional capacity — to address the modeling-error subset identified in the error analysis.
- Extend the architecture and hyperparameter experiments to a more complex dataset (e.g., FashionMNIST or CIFAR-10) to see whether the same TinyVGG configuration generalizes beyond simple digit classification.
