# Reproducibility Code for DCR (Synthetic Experiments)

This repository contains the implementation to reproduce the synthetic data experiments presented in the paper.

## Overview

The code compares **DCR (Distribution-Free Calibration for Ranking)** and its stochastic approximation **MDCR** on a controlled synthetic dataset using a RankNet base model.

## File Structure

- `main.py`: Entry point. Handles data generation, model training, and the main evaluation loop.
- `methods.py`: Implementation of the DCR and MDCR algorithms (threshold calculation, set construction).
- `model.py`: PyTorch implementation of the RankNet architecture.

## Requirements

- Python >= 3.8
- numpy
- pandas
- scipy
- torch
- scikit-learn
- tqdm

Install dependencies via:
```bash
pip install -r requirements.txt