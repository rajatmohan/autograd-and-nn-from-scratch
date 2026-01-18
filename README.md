# From Scratch Neural Network & Autograd Engine

This repository contains a **from-scratch implementation of an automatic differentiation engine and neural network framework**, written purely in Python without using deep learning libraries such as PyTorch or TensorFlow.

The goal of this project is **educational**: to deeply understand how modern neural networks work internally — from computation graphs and backpropagation to optimizers and training loops.

---

## Inspiration

This work is inspired by:

> **“[The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)”**  
> by **[Andrej Karpathy](https://karpathy.ai)**  

Karpathy’s explanation motivated me to go beyond usage and **rebuild the core ideas from first principles**, extending them into a more complete, modular neural-network pipeline.

---

## What This Project Implements

### 1. Custom Tensor & Autograd Engine
- Scalar-based `MyTensor` object
- Dynamic computation graph
- Automatic differentiation using reverse-mode autodiff
- Supported operations:
  - Addition, subtraction, multiplication, division
  - Power
  - Negation
  - `exp`, `tanh`, `sigmoid`
- Topological sorting for backpropagation

---

### 2. Neural Network Building Blocks
- Linear (fully connected) layers
- Activation functions
- Sequential model abstraction
- Parameter collection & management

This allows building **general neural networks**, not just hard-coded MLPs.

---

### 3. Loss Functions
- Binary Cross-Entropy (BCE)
- Mean Squared Error

---

### 4. Optimizers
- Stochastic Gradient Descent (SGD)
- Adam Optimizer

---

### 5. Training Pipeline
- Mini-batch gradient descent
- Epoch-wise training loop
- Loss tracking
- Accuracy evaluation

---

### 6. Real-World Dataset Experiment
- **Breast Cancer Wisconsin Dataset**
- Binary classification task
- Achieved:
  - ~97–98% test accuracy using single hidden layer of 16 neurons

---


