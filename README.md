# MNIST CNN Training Project

A PyTorch-based Convolutional Neural Network implementation for MNIST digit classification with comprehensive training reports and evaluation metrics.

## 🏗️ Model Architecture

The CNN model consists of 4 convolutional layers followed by 2 fully connected layers:

```
Input (1, 28, 28) 
    ↓
Conv2d(1→32, kernel=3) + ReLU + MaxPool2d(2)
    ↓
Conv2d(32→64, kernel=3) + ReLU + MaxPool2d(2)
    ↓
Conv2d(64→128, kernel=3) + ReLU
    ↓
Conv2d(128→256, kernel=3) + ReLU + MaxPool2d(2)
    ↓
Flatten → Linear(256×4×4 → 50) + ReLU
    ↓
Linear(50 → 10) + LogSoftmax
    ↓
Output (10 classes)
```

### Architecture Details:
- **Total Parameters**: 593,200
- **Trainable Parameters**: 593,200
- **Input Size**: 28×28 grayscale images
- **Output**: 10 classes (digits 0-9)
- **Activation**: ReLU for hidden layers, LogSoftmax for output

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
- [CNN Model_MNIST_20250919_130247.html](reports/CNN%20Model_MNIST_20250919_130247.html) - Most recent run
- [CNN Model_MNIST_20250919_125504.html](reports/CNN%20Model_MNIST_20250919_125504.html)
- [CNN Model_MNIST_20250919_125115.html](reports/CNN%20Model_MNIST_20250919_125115.html)
- [CNN Model_MNIST_20250919_124230.html](reports/CNN%20Model_MNIST_20250919_124230.html)

### Report Features:
- 📊 **Training Metrics**: Loss and accuracy curves
- 🔧 **Model Configuration**: Architecture and hyperparameters
- 📈 **Epoch-by-Epoch Results**: Detailed training history
- 🎯 **Final Performance**: Test accuracy and loss
- 📱 **Responsive Design**: Mobile-friendly HTML reports

## 🎯 Performance Metrics

### Latest Results:
- **Final Test Accuracy**: 40.35%
- **Final Test Loss**: 2.2249
- **Training Epochs**: 1 (single epoch run)
- **Model Parameters**: 593,200

> **Note**: The current results show a single epoch run. For optimal performance, run the full 20-epoch training cycle.

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