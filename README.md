# 🔮 Style_Transfer

# Neural Style Transfer Implementation

This repository contains an implementation of Neural Style Transfer using PyTorch, based on the paper "A Neural Algorithm of Artistic Style".

## Overview

Neural Style Transfer is a technique that combines the content of one image with the style of another image. The implementation uses a pre-trained VGG19 network to extract features and optimize a generated image to match both content and style characteristics.

<img width="856" alt="Screenshot 2025-05-06 at 1 10 40 am" src="https://github.com/user-attachments/assets/8649564a-e08d-4020-a4c0-86c6bc125dd7" />


## 🔑 Key Components

### Model Architecture
- Uses pre-trained VGG19 as the feature extractor
- Extracts features from specific layers:
  - Content features: conv4_2 layer
  - Style features: conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 layers

### Image Representation
- Content images are processed through VGG19 to obtain deep feature representations
- Style images are processed to create Gram matrices, which capture the style information
- Input images are preprocessed:
  - Resized to 512x512
  - Normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Loss Functions

<img width="891" alt="Screenshot 2025-05-06 at 1 17 51 am" src="https://github.com/user-attachments/assets/24fcd297-1d9d-4ea4-b348-7397d9565f2e" />

#### Content Loss
- Measures the difference between content features of the content image and generated image
- Uses Mean Squared Error (MSE) between feature representations
- Extracted from conv4_2 layer of VGG19

#### Style Loss
- Uses **Gram matrices** to capture style information
- Gram matrix calculation:
  
   <img width="726" alt="Screenshot 2025-05-06 at 1 14 04 am" src="https://github.com/user-attachments/assets/eab035c6-21de-4f87-82aa-b4033cfb6a59" />

  - Reshapes feature maps to (batch_size, channels, height*width)
  - Computes matrix multiplication of features and its transpose
  - It consists of the correlations between the different filter responses, where the expectation is taken over the spatial extent of the feature maps.
- Measures MSE between Gram matrices of style image and generated image
- Applied to multiple layers (conv1_1 through conv5_1)

#### Correlation between Content Loss and Style Loss with respect to the number of convolution layers

<img width="850" alt="Screenshot 2025-05-06 at 1 19 02 am" src="https://github.com/user-attachments/assets/540d9d3b-aec4-46c2-9215-a0cdc0ccd67e" />

- As the input passes through deeper convolutional layers, the spatial resolution decreases due to smaller effective receptive fields, causing the content image to become more blurred.
- However, at the same time, the number of channels increases, allowing the Gram matrix to capture style features more distinctly from the style image.

### Optimization
- Optimizer: LBFGS (Limited-memory BFGS) is the best for image synthesis according to paper
- Learning rate: 1.0
- Loss weights:
  - Content loss weight (α): 1
  - Style loss weight (β): 1e6
- Important: The optimization process **updates the generated image (x) directly**, not the model parameters
- The VGG19 model remains frozen during the entire process

### Training Process
- Runs for 1000 epochs
- Saves generated images every 100 epochs
- Uses tqdm for progress tracking
- Implements closure function for LBFGS optimizer
- Tracks and displays:
  - Total loss
  - Content loss
  - Style loss

## Key Implementation Details

### Gram Matrix
The Gram matrix is crucial for style representation as it:
- Captures the correlation between different feature maps
- Is invariant to the spatial arrangement of features
- Preserves style information while being independent of content
- Formula: G = (F * F^T) / (C * H * W)
  where F is the feature map, C is channels, H is height, and W is width

### Image Processing
- Pre-processing:
  - Resize to 512x512
  - Convert to tensor
  - Normalize using ImageNet statistics
- Post-processing:
  - Denormalize using ImageNet statistics
  - Clip values to [0, 1]
  - Convert to uint8 format
  - Save as JPEG

## Usage

1. Place your content and style images in the `images` directory
2. Run the training script:
```python
python train.py
```
3. Generated images will be saved in the output directory with format: `{alpha}_{beta}_{learning_rate}/generated_{epoch}.jpg`

## Requirements
- PyTorch
- torchvision
- PIL
- numpy
- tqdm

## Notes
- The implementation focuses on optimizing the generated image directly rather than training a model
- The style transfer process is computationally intensive and may take significant time depending on your hardware
- Results can be tuned by adjusting the content and style loss weights (α and β)
