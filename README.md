# Python AI Journey — Foundations: Data Visualization

## Overview

**Python AI Journey** is a structured, step-by-step learning repository designed to build strong foundations in data analysis, visualization, and machine learning using Python.

This first module focuses on **core Python data tools** and demonstrates how raw demographic data can be transformed into meaningful visual insights using `pandas` and `matplotlib`.

The goal of this repository is to **learn one tool at a time**, beginning with visualization and gradually progressing toward full AI/ML systems and neural networks.

This project acts as the starting point of the journey:
➡ Data handling
➡ Visualization
➡ Machine learning fundamentals
➡ Deep learning

---

## What This Module Teaches

This module introduces:

* Structuring data using **Pandas DataFrames**
* Visualizing distributions and relationships using **Matplotlib**
* Understanding how demographic data is explored before building AI models
* Writing clean, modular Python scripts for data work
* Preparing the environment for future AI/ML integration

---

## Visualizations Included

Using a sample demographic dataset:

* **Histogram** → Age distribution
* **Pie Chart** → Gender distribution
* **Bar Graph** → Average income comparison
* **Scatter Plot** → Age vs Income relationship
* **Line Graph** → Income trend by age

These visualizations demonstrate how different chart types reveal different patterns in the same dataset.

---

## External Python Libraries Used

This module introduces the essential libraries used throughout the AI journey.

### Data & Math

```python
import numpy as np
import pandas as pd
```

### Visualization

```python
import matplotlib.pyplot as plt
```

### Machine Learning Foundations (introduced for future modules)

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
```

### Deep Learning (used in later stages)

```python
import torch
import torch.nn as nn
import torch.optim as optim
```

---

## Tech Stack

* **Language**: Python 3.10+ (recommended: Python 3.14)
* **Data Processing**: Pandas, NumPy
* **Visualization**: Matplotlib
* **Machine Learning**: Scikit-learn (intro), PyTorch (future modules)

---

## Setup & Run Instructions

### 1. Clone the repository

```bash
git clone https://github.com/AyanMajumdar100/python-tools.git
cd python-ai-journey
```

---

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
```

---

### 3. Activate environment

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

---

### 4. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn torch torchvision
```

---

### 5. Run the visualization script

```bash
python 01_data_visualization.py
```

---

## Learning Philosophy

This repository follows a **progressive AI learning path**:

### Stage 1 — Python Tools

Learn core libraries individually:

* NumPy → numerical computation
* Pandas → data manipulation
* Matplotlib → visualization

### Stage 2 — Data Understanding

Explore patterns before modeling:

* distributions
* correlations
* feature relationships

### Stage 3 — Machine Learning

Introduce predictive models:

* regression
* preprocessing
* model evaluation

### Stage 4 — Deep Learning

Build neural networks using PyTorch:

* tensors
* training loops
* optimization
* model visualization

Each directory represents a milestone in the journey.

---

## Project Structure

```text
python-tools/
│
├── 01_data_visualization.py
├── 01_data_visualization_GUI.py
│
├── data/
│   └── (Optional) data.csv
│
├── Screenshots/
│   ├── PyTorch NN Mockup.jpg
│   └── ....
│
├── Notes.txt
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Future Expansions

Planned additions to the repository:

* Loading real datasets from CSV
* Interactive GUI dashboards
* Data preprocessing pipelines
* Model evaluation visualizations
* Neural network training modules
* End-to-end AI applications

---

## License

This repository is intended for educational and learning purposes.
See the `LICENSE` file for details.

---

## Author Note

This repository documents a hands-on journey into Artificial Intelligence — starting from basic Python tools and gradually building toward real AI systems.

Built for learning, experimentation, and growth in AI
Welcome to the journey 🚀