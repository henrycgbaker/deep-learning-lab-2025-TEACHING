# GRAD-E1394: Deep Learning Labs 2025

Graduate-level lab materials covering deep learning fundamentals through modern architectures. Each lab builds practical skills with hands-on PyTorch implementations.

## Getting Started

| Notebook | Description |
|----------|-------------|
| [Python Foundations](lab_1_python_foundations.ipynb) | **Start here** — type hints, NumPy, PyTorch tensors, autograd basics |

## Labs

### Neural Network Fundamentals

| Lab | Notebook | Topics |
|-----|----------|--------|
| 2 | [Intro to FFNNs](lab_2_intro_to_ffnns.ipynb) | Feed-forward networks, backpropagation, MNIST |
| 3 | [TensorBoard & Tuning](lab_3_tensorboard_vanishing_gradients_hyperparams.ipynb) | Vanishing gradients, hyperparameter tuning |

### Convolutional Networks

| Lab | Notebook | Topics |
|-----|----------|--------|
| 4 | [Intro to CNNs](lab_4_introduction_to_cnns.ipynb) | Convolutions, pooling, image classification |
| 5 | [Advanced CNNs](lab_5_advanced_cnns.ipynb) | ResNet, transfer learning, class imbalance |

### Sequence Models

| Lab | Notebook | Topics |
|-----|----------|--------|
| 6 | [RNN Foundations](lab_6_rnn_foundations.ipynb) | Vanilla RNNs, BPTT, language models |
| 7 | [Advanced RNNs](lab_7_rnn_adv.ipynb) | LSTM, GRU, gating mechanisms |
| 8 | [Seq2Seq & Attention](lab_8_seq2seq_attention.ipynb) | Encoder-decoder, attention, translation |

### Transformers & Production

| Lab | Notebook | Topics |
|-----|----------|--------|
| 9 | [Transformers](lab_9_transformers.ipynb) | Self-attention, positional encoding |
| 10 | [MLOps & Fine-tuning](lab_10_mlops_finetuning_evaluation.ipynb) | HuggingFace, fine-tuning, evaluation |

## Problem Sets

Theory problem sets with solutions are available in [`problem_sets/`](problem_sets/).

## Prerequisites

- Python programming (functions, classes, list comprehensions)
- Linear algebra (matrices, vectors, dot products)
- Calculus (derivatives, chain rule)
- Basic probability and statistics

## Frameworks

- **PyTorch** — primary deep learning framework
- **HuggingFace Transformers** — pre-trained models (Labs 9-10)
- **TensorBoard** — training visualisation

## Companion Notes

Unofficial companion notes are available covering theoretical foundations and lecture content:

### Deep Learning Notes
📄 **[Download PDF](henry_notes/deep_learning/main.pdf)** — Comprehensive deep learning notes (weeks 1-9)

Covers: DNN fundamentals, backpropagation, CNNs, RNNs/LSTMs, attention mechanisms, transformers, and LLMs in practice.

### Prerequisite Notes

Also included are notes from prerequisite courses for additional context:

| Subject | PDF | Topics |
|---------|-----|--------|
| Machine Learning | [Download](henry_notes/machine_learning/main.pdf) | Training, generalisation, kernels, trees, neural networks |
| Maths for Data Science | [Download](henry_notes/maths_for_data_science/main.pdf) | Probability, calculus, random variables, statistics |

LaTeX source files are available in [`henry_notes/`](henry_notes/) for reference or modification.
