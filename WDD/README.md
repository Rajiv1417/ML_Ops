# Welding Defect Detection 🔬

An interactive 180-minute workshop on AI model optimization for edge deployment.

## Quick Start

### Prerequisites
- **Python**: 3.11 or 3.12
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 500MB available
- **OS**: Windows, macOS, or Linux

### Installation

1. **Extract the bundle** to your laptop
   ```bash
   cd Welding_Defect_Detection
   ```

2. **Run the setup script** (automatic - recommended)
   
   **Windows:**
   ```bash
   setup.bat
   ```
   
   OR **Manual installation:**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**: The app will open automatically at `http://localhost:8501`

---

## Workshop Structure

### 🎓 Phase 1: Training (10-20 minutes)

In this phase, you will:
- **Generate** synthetic weld defect data (~1 minute)
- **Configure** training parameters (epochs, batch size, learning rate)
- **Train** a MobileNetV2 model to classify weld defects
- **Monitor** loss curves and accuracy metrics in real-time
- **Save** checkpoints for later optimization

**Learning Goals:**
- Understand neural network training
- Recognize overfitting vs underfitting
- Learn about validation and testing

### 🔧 Phase 2: Optimization (160+ minutes)

In this phase, you will:
- **Load** your trained model
- **Select** a deployment profile for your weld inspection scenario
- **Optimize** the model using three techniques:
  - **Compression**: Reduce model complexity
  - **Pruning**: Remove unnecessary network connections
  - **Quantization**: Use fewer bits for weights
- **Monitor** trade-offs between accuracy and model size
- **Deploy** when constraints are satisfied

**Learning Goals:**
- Understand model compression techniques
- Learn about edge AI constraints
- Make engineering trade-off decisions
- Deploy models for resource-constrained environments

---

## What's Included

```
Welding_Defect_Detection/
├── app.py                        # Main Streamlit application
├── config.py                     # Configuration file
├── requirements.txt              # Python dependencies
│
├── data/
│   └── generator.py             # Synthetic data generation
│
├── trainer/
│   ├── model_builder.py         # MobileNetV2 implementation
│   ├── dataset.py               # Data loading & preprocessing
│   └── trainer.py               # Training engine
│
├── simulator/                    # (Coming: optimization engine)
├── ui/                          # (Coming: optimization UI)
├── checkpoints/                 # Saved model checkpoints
└── reports/                     # Generated reports
```

---

## Features

✅ **Offline-First**: No internet required after installation  
✅ **Portable**: Runs on any laptop with Python 3.11+  
✅ **CPU-Optimized**: No GPU needed (trains in ~5-10 minutes)  
✅ **Synthetic Data**: Reproducible, no licensing issues  
✅ **Interactive UI**: No coding required  
✅ **Educational**: Learn ML optimization concepts hands-on  

---

## Synthetic Dataset

The data generator creates **800 synthetic weld inspection images**:

**Defect Types:**
- Cracks
- Porosity
- Undercut
- Spatter
- Incomplete Fusion

**Split:**
- 70% Training (560 images)
- 15% Validation (120 images)
- 15% Test (120 images)

Generated images are realistic and support the educational goals while keeping the bundle portable.

---

## Model Architecture

**MobileNetV2** - A lightweight CNN optimized for mobile and edge devices

- **Base Size**: ~14 MB
- **Trainable Parameters**: ~3.5 million
- **Input**: 224×224 RGB images
- **Output**: 2 classes (Normal / Defect)

Designed to be:
- ✅ Fast to train on CPU
- ✅ Compressible for optimization learning
- ✅ Representative of real edge AI models

---

## System Requirements Details

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| Python | 3.11 | 3.12 |
| RAM | 4 GB | 8+ GB |
| Storage | 500 MB | 1+ GB |
| Processor | Any CPU | Modern multi-core |
| GPU | Not required | Not required |
| Internet | Not required | Not required |

**Training Time**: ~5-10 minutes for 10 epochs on typical laptop CPU

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
```bash
pip install -r requirements.txt
```

### "Port 8501 already in use"
```bash
streamlit run app.py --server.port 8502
```

### Slow training on Windows
- This is normal on CPU. Training 10 epochs takes ~5-10 minutes.
- Consider reducing epochs or batch size.

### Dataset generation fails
- Check disk space (need ~100 MB)
- Try: `python -c "from data.generator import generate_dataset; generate_dataset(force_regenerate=True)"`

---

## Educational Goals

By the end of the workshop, you will understand:

1. **Model Compression**: How to reduce model size without losing accuracy
2. **Pruning**: Removing unnecessary network connections
3. **Quantization**: Using fewer bits for weights and activations
4. **Compression vs Inference**: Trade-offs between model size and speed
5. **Sparsity**: How to measure and optimize sparse models
6. **Accuracy vs Model Size**: The fundamental trade-off in edge AI
7. **Latency & Throughput**: Real-time constraints for edge devices
8. **Deep Compression**: Combining all techniques for maximum efficiency
9. **Deployment Constraints**: Hardware limitations in the wild
10. **Engineering Trade-offs**: Making real-world optimization decisions
11. **Edge AI Deployment**: Practical considerations for resource-constrained devices

---

## Deployment Profiles

Students can optimize for different deployment scenarios:

### 🔶 Weld Vision
- Real-time weld inspection on assembly line
- Max Size: 20 MB
- Max Latency: 100 ms
- Min Accuracy: 92%

### 🔵 Battery Inspection
- Battery defect detection in QC
- Max Size: 50 MB
- Max Latency: 200 ms
- Min Accuracy: 95%

### 🟡 Paint Inspection
- Paint defect detection for car bodies
- Max Size: 100 MB
- Max Latency: 500 ms
- Min Accuracy: 90%

---

## Advanced Usage

### Generate your own dataset
```python
from data.generator import generate_dataset

# Force regenerate with different seed
generate_dataset(num_defects=500, num_normal=500, force_regenerate=True)
```

### Train via command line
```python
from trainer.trainer import WeldDefectTrainer
from trainer.dataset import create_dataloaders
from config import DATASET_DIR

trainer = WeldDefectTrainer()
train_loader, val_loader, _ = create_dataloaders(DATASET_DIR, batch_size=16)
results = trainer.train(train_loader, val_loader, num_epochs=10)
```

---

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Verify Python version: `python --version`
3. Check dependencies: `pip list | grep torch`
4. Review the specification: See `Welding_Defect_Detection_Master_Project_Specification.md` (in parent directory)

---

## License

Educational use only. Tata Motors workshop material.

---

**Happy Optimizing! 🚀**
