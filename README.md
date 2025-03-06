# MLACAD Project

A deep learning project for CAD (Computer-Aided Design) sequence modeling and generation using mla-enhanced transformer-based architecture.

## Overview

This project implements a novel approach to understanding and generating CAD modeling sequences using deep learning techniques. It focuses on learning the underlying patterns in CAD operations to assist in automated design processes.

## Key Features

- Transformer-based architecture for CAD sequence modeling
- Support for various CAD operations and command sequences
- Data augmentation for robust training
- Configurable model architectures (base, large, and deep variants)
- Mixed precision training support
- Gradient accumulation for memory efficiency

## Technical Details

- **Framework**: PyTorch
- **Architecture**: Modified Transformer
- **Training**: 
  - Mixed precision (FP16)
  - Gradient accumulation
  - Configurable batch sizes and model dimensions
  - Multiple training configurations (standard, large batch, progressive)

## Model Configurations

Three main model variants are provided:
- **Base Model**: Lightweight version (192 dim, 4 layers)
- **Large Model**: Enhanced capacity (256 dim, 6 layers)
- **Deep Model**: Deeper architecture (192 dim, 8 layers)

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA capable GPU
- h5py
- numpy
- tqdm

