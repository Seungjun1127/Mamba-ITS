# Mamba-ITS: Next-Generation Time Series-Vision Model with SSM

## Project Overview

**Mamba-ITS** inherits the visualization strategy of ViTST and reconstructs the structural advantages of Swin Transformer (local window, hierarchical patch merging, inter-window interaction, etc.) with SSM (Mamba)-based modules. This next-generation time series-vision model targets the PAM (Patient Activity Monitoring) dataset, aiming to overcome the limitations of existing Transformer-based models by combining the intuitiveness of time series → image conversion with the theoretical strengths of SSM.

## Key Design Principles

### 1. Local + Global SSM Fusion
- Replaces Swin's window self-attention:
  - **Local SSM**: Processes 7×7 patches as 1D flattened sequences with SSM
  - **Global SSM**: Applies SSM to the entire sequence
- Merges both paths in parallel to capture both local and global patterns

### 2. Hierarchical Patch Merging
- Inherits Swin's patch merging:
  - Groups 2×2 neighboring patches → halves resolution, doubles channels
  - Connects independent SSM blocks at each stage

### 3. Explicit 2D Positional Encoding
- Adds (row, col) dual embeddings to each token
  - Clearly reflects 2D spatial information
  - Can experiment with line embedding and other variants

### 4. Inter-Window Interaction Support
- Implements Swin's shifted window or
  - Intermediate global pooling & broadcast
  - Designs auxiliary paths for information flow between distant tokens

## Goals & Research Significance

1. **Maintain ViTST's Visualization Strategy**  
   - Inherits the intuitive interpretability of converting irregular time series into "line graph images"

2. **Replace Transformer Self-Attention with Mamba (SSM) Blocks**  
   - Secures theoretical strengths such as frequency response and dynamic modeling
   - Enhances scalability with linear time and memory complexity

3. **Reconstruct Swin's Structural Advantages for SSM**  
   - Window-wise SSM, shifted SSM, hierarchical patch merging, etc.

4. **Secure Mathematical & Signal Processing Interpretability**  
   - From "Why visualize as images?" to "What filters (kernels) capture patterns?"
   - Strengthens theoretical justification and interpretability

## Dataset

- **PAM (Patient Activity Monitoring)**
  - Converts time series data into line graph images as input
  - Provides data preprocessing, visualization pipeline, and augmentation

## Model Architecture

```
Time Series Input
   ↓
Visualization (Line Graph Image)
   ↓
Patch Embedding
   ↓
[Local SSM + Global SSM] × N
   ↓
Patch Merging (Resolution ↓, Channels ↑)
   ↓
[SSM Block] × M
   ↓
Shifted SSM, Global Pooling, etc.
   ↓
Classification Head
```

## Installation & Usage

### 1. Environment Setup

```bash
git clone https://github.com/Seungjun1127/Mamba-ITS.git
cd Mamba-ITS
# (Virtual environment recommended)
pip install -r requirements.txt
```

### 2. Data Preparation

- Download and preprocess the PAM dataset
- Place images and metadata in the `data/` folder

### 3. Training/Evaluation

```bash
TBD
```

## References

- [Time Series as Images: Vision Transformer for Irregularly Sampled Time Series](https://arxiv.org/abs/2303.12799)
- [MambaVision: A Hybrid Mamba-Transformer Vision Backbone](https://arxiv.org/abs/2407.08083)

## Contribution & Contact

- Pull requests and issues are welcome
- For research collaboration and inquiries: tmdwns1127@kaist.ac.kr

## License

- MIT License

### ✨ The starting point for next-generation time series-vision model research that captures both mathematical interpretability and practical performance!
