# Mood Prediction from Smartphone Sensor Data

Predicting user mood states from smartphone sensor data using machine learning, completed as part of the Data Mining Techniques course at VU Amsterdam.

**Authors:** Kushnava Singha and collaborators

## Overview

This project tackles mood classification using the `dataset_mood_smartphone.csv` dataset, which contains time-series sensor readings (accelerometer, GPS, app usage, call/SMS logs, screen activity) alongside self-reported mood labels.

## Methods

- **Random Forest Classifier** — baseline model with grid search hyperparameter tuning
- **Recurrent Neural Network (RNN)** — PyTorch-based sequential classifier for capturing temporal patterns
- **Custom KNN Imputer** — Annoy-based approximate nearest neighbor imputer for handling missing sensor data efficiently

## Project Structure

| File | Description |
|------|-------------|
| `main.py` | End-to-end pipeline: data loading, preprocessing, model training and evaluation |
| `rnn_classifier.py` | PyTorch RNN model definition |
| `annoyimputer.py` | Custom KNN imputer using the Annoy library |
| `plots.py` | Visualization utilities |
| `regression.ipynb` | Regression analysis notebook |
| `Exploratory_Analysis.ipynb` | Exploratory data analysis |

## How to Run

```bash
pip install numpy pandas scikit-learn torch annoy matplotlib tqdm
python main.py
```

Jupyter notebooks can be run independently for exploration and regression analysis.

## Requirements

- Python 3.8+
- PyTorch
- scikit-learn
- Annoy
- pandas, NumPy, Matplotlib
