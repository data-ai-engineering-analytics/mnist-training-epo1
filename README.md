# MNIST CNN Training Project

A PyTorch-based Convolutional Neural Network implementation for MNIST digit classification with comprehensive training reports and evaluation metrics.

## 🏗️ Model Architecture

The optimized CNN model consists of 4 convolutional layers followed by 1 fully connected layer:

```
Input (1, 28, 28) 
    ↓
Conv2d(1→8, kernel=3) + ReLU + MaxPool2d(2)
    ↓
Conv2d(8→16, kernel=3) + ReLU + MaxPool2d(2)
    ↓
Conv2d(16→22, kernel=3) + ReLU
    ↓
Conv2d(22→56, kernel=3) + ReLU + MaxPool2d(2)
    ↓
Flatten → Linear(56×4×4 → 10) + LogSoftmax
    ↓
Output (10 classes)
```

### Architecture Details:
- **Total Parameters**: 24,552
- **Trainable Parameters**: 24,552
- **Input Size**: 28×28 grayscale images
- **Output**: 10 classes (digits 0-9)
- **Activation**: ReLU for hidden layers, LogSoftmax for output
- **Key Optimization**: Removed the second fully connected layer to reduce parameters while maintaining performance

## 🚀 Quick Start

### Prerequisites
- Python ≥ 3.12
- UV package manager

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd mnist-training-epo1

# Install dependencies using UV
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Training
```bash
# Run the training notebook
jupyter notebook mnist_cnn_training.ipynb
```

## 📊 Training Configuration

| Parameter | Value |
|-----------|-------|
| **Dataset** | MNIST |
| **Batch Size** | 512 |
| **Epochs** | 20 |
| **Optimizer** | SGD (lr=0.001, momentum=0.9) |
| **Scheduler** | StepLR (step_size=15, gamma=0.1) |
| **Loss Function** | CrossEntropyLoss |
| **Device** | MPS (Apple Silicon) / CUDA / CPU |

## 📈 Training Reports

Comprehensive HTML reports are automatically generated for each training run:

### Latest Training Reports:
- [CNN Model_MNIST_20250919_162301.html](reports/CNN%20Model_MNIST_20250919_162301.html) - **95.21% Accuracy** (24,552 parameters)
- [CNN Model_MNIST_20250919_151823.html](reports/CNN%20Model_MNIST_20250919_151823.html) - 95.07% Accuracy
- [CNN Model_MNIST_20250919_130247.html](reports/CNN%20Model_MNIST_20250919_130247.html)
- [CNN Model_MNIST_20250919_125504.html](reports/CNN%20Model_MNIST_20250919_125504.html)

### Report Features:
- 📊 **Training Metrics**: Loss and accuracy curves
- 🔧 **Model Configuration**: Architecture and hyperparameters
- 📈 **Epoch-by-Epoch Results**: Detailed training history
- 🎯 **Final Performance**: Test accuracy and loss
- 📱 **Responsive Design**: Mobile-friendly HTML reports

## 🎯 Performance Metrics

### Latest Results:
- **Final Test Accuracy**: 95.21%
- **Final Test Loss**: 0.1559
- **Training Epochs**: 1 (single epoch run)
- **Model Parameters**: 24,552

> **Achievement**: Successfully achieved >95% accuracy with <25,000 parameters in a single epoch through careful architecture optimization and hyperparameter tuning.

## 📊 Incremental Optimization Journey

The following table shows the step-by-step improvements made to achieve >95% test accuracy with <25,000 parameters:

| Commit | Change Description | Batch Size | Test Accuracy | Parameters | Key Insight |
|--------|-------------------|------------|---------------|------------|-------------|
| `4e218cb` | Reduced batch size from 512 to 256 | 256 | 76.74% | ~593K | Smaller batch sizes improve learning |
| `a27c374` | Further reduced batch size to 128 | 128 | 87.48% | ~593K | Continued batch size optimization |
| `68d1bb1` | Reduced batch size to 64 | 64 | 91.83% | ~593K | Significant accuracy improvement |
| `3ae9ac8` | Reduced batch size to 32 | 32 | 94.40% | ~593K | Approaching target accuracy |
| `72452f7` | Reduced batch size to 16 | 16 | 94.79% | ~593K | Peak accuracy with original architecture |
| `0ccde0f` | Increased batch size to 32, changed layer channels | 32 | 92.36% | ~593K | Architecture changes needed |
| `b224f37` | **Optimized architecture**: Changed I/O channels and removed last FC layer | 32 | **95.07%** | **24,552** | **Target achieved!** |
| `fb087de` | Added shuffle=True for better results | 32 | **95.21%** | **24,552** | Final optimization |

### Key Optimizations Applied:
1. **Batch Size Tuning**: Systematically reduced from 512 → 32 for optimal learning
2. **Architecture Optimization**: 
   - Reduced channel counts: 1→8→16→22→56 (vs original 1→32→64→128→256)
   - Removed second fully connected layer (FC2)
   - Direct connection from FC1 to output layer
3. **Parameter Reduction**: From 593,200 → 24,552 parameters (96% reduction)
4. **Data Shuffling**: Enabled shuffle=True for better generalization

## 🧠 Model Architecture Details

### Net() Model Output
```
Net(
  (conv1): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1))
  (conv3): Conv2d(16, 22, kernel_size=(3, 3), stride=(1, 1))
  (conv4): Conv2d(22, 56, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=896, out_features=10, bias=True)
)
```

### Model Summary (torchsummary)
```
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
├─Conv2d: 1-1                            80
├─Conv2d: 1-2                            1,168
├─Conv2d: 1-3                            3,190
├─Conv2d: 1-4                            11,144
├─Linear: 1-5                            8,970
=================================================================
Total params: 24,552
Trainable params: 24,552
Non-trainable params: 0
=================================================================
```

## 📊 Training Results

### Training and Test Accuracy Output
```
Epoch 1
Train: Loss=0.0975 Batch_id=1874 Accuracy=77.43: 100%|██████████| 1875/1875 [00:07<00:00, 236.31it/s]
Test set: Average loss: 0.1559, Accuracy: 57124/60000 (95.21%)
```

**Key Performance Metrics:**
- **Training Accuracy**: 77.43% (final batch)
- **Test Accuracy**: 95.21% (57,124/60,000 correct predictions)
- **Test Loss**: 0.1559
- **Training Speed**: 236.31 iterations/second on MPS

## 🛠️ Project Structure

```
mnist-training-epo1/
├── mnist_cnn_training.ipynb    # Main training notebook
├── reports/                    # Generated HTML reports
│   ├── CNN Model_MNIST_*.html
│   └── ...
├── pyproject.toml             # Project dependencies
├── requirements.txt           # Alternative dependency list
├── uv.lock                   # UV lock file
└── README.md                 # This file
```

## 🔧 Dependencies

Core dependencies managed via `pyproject.toml`:
- **PyTorch** ≥ 2.8.0 (with MPS support)
- **TorchVision** ≥ 0.23.0
- **Matplotlib** ≥ 3.10.6
- **Pandas** ≥ 2.3.2
- **TQDM** ≥ 4.67.1
- **IPyKernel** ≥ 6.30.1

## 🖥️ Device Support

The project automatically detects and uses the best available device:
- **MPS** (Apple Silicon GPU) - Primary choice for Mac users
- **CUDA** (NVIDIA GPU) - For NVIDIA GPU systems
- **CPU** - Fallback option

## 📝 Usage Example

```python
# Initialize the model
model = Net().to(device)

# Set up training
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Train the model
for epoch in range(num_epochs):
    train(model, device, train_loader, optimizer, criterion)
    test(model, device, test_loader, criterion)
    scheduler.step()
```

## 📊 Data Augmentation

The training pipeline includes data augmentation:
- **Random Center Crop** (22×22) with 10% probability
- **Random Rotation** (-15° to +15°)
- **Normalization** using MNIST statistics

## 🎨 Visualization

Training reports include:
- Loss curves (training vs test)
- Accuracy progression
- Model architecture summary
- Performance metrics dashboard

## 📄 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the training notebook to generate reports
5. Submit a pull request

---

**Generated with ❤️ using PyTorch and UV**