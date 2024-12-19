# CIFAR10 Image Classification

This project implements a CNN model for CIFAR10 image classification using PyTorch. The model architecture follows specific requirements including the use of depthwise separable convolutions, dilated convolutions, and no max pooling layers.

## Project Structure

├── Models.py # Contains all model architectures
├── train.py # Training script with data loading and training loops
├── data/ # Directory for MNIST dataset (auto-downloaded)
├── models/ # Directory for saved model checkpoints
└── images/ # Directory for saved screenshots

## Model Architecture

The model consists of 4 convolution blocks (C1, C2, C3, C4) with the following features:
- C1: Standard convolutions with stride=2 reduction
- C2: Dilated convolutions (dilation=4) with stride=2 reduction
- C3: Depthwise separable convolutions with stride=2 reduction
- C4: Standard convolutions with stride=2 reduction
- Global Average Pooling followed by FC layer
- Final Receptive Field: 55 (requirement met, > 44)
- No MaxPooling (uses strided convolutions instead)
- Less than 200k parameters

## Data Augmentation

Using Albumentations library with:
- Horizontal Flip (p=0.5)
- ShiftScaleRotate
  - Shift limit: ±0.1
  - Scale limit: ±0.1
  - Rotate limit: ±15 degrees
  - Probability: 0.5
- CoarseDropout
  - max_holes = 1
  - max_height/width = 16px
  - min_holes = 1
  - min_height/width = 16px
  - fill_value = dataset mean
  - Probability: 0.5
- Normalization
  - Mean: (0.4914, 0.4822, 0.4465)
  - Std: (0.2470, 0.2435, 0.2616)

## Training Details

- **Optimizer**: SGD with momentum (0.9) and weight decay (5e-4)
- **Learning Rate**: OneCycleLR scheduler
  - Initial LR: 0.01
  - Max LR: 0.1
- **Batch Size**: 128
- **Epochs**: 20
- **Device**: Automatically uses CUDA if available, else CPU

## Metrics Tracking

The training process tracks:
- Training Loss
- Training Accuracy
- Test Loss
- Test Accuracy

Plots are generated at the end of training showing the progression of these metrics.

## Usage

1. Clone the repository
2. Run train.py to start training:

The CIFAR10 dataset will be automatically downloaded to the `data/` directory on first run.

## Requirements

- PyTorch
- torchvision
- tqdm
- matplotlib
- numpy

