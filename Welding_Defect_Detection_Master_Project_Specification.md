# Welding Defect Detection
# Complete Engineering Project Specification (Version 1.0)

> This document is intended to serve as the master specification for GitHub Copilot Agent, Cursor, Claude Code, or a software development team.

---

# 1. Executive Summary

## Objective

Build a production-quality Streamlit application that teaches AI model optimization concepts through an interactive simulation based on an automotive manufacturing use case.

The application will be used in a 180-minute instructor-led workshop. Participants should not write Python code. They interact only with the graphical interface. The extended duration allows for deeper exploration of optimization techniques and more hands-on practice.

## Deployment Model

- **Single standalone bundle**: All code, model architecture, training data, dependencies bundled together
- **Completely offline**: No internet required after initial distribution
- **Portable**: Runs on any Windows/Mac/Linux laptop with Python 3.11+
- **CPU-only**: No GPU/CUDA required; optimized for typical student laptops (8GB RAM)
- **Two-phase workflow**: 
  1. Training phase (5-10 min): Students train a base model on weld inspection data
  2. Optimization phase (remaining time): Students optimize the trained model

---

# 2. Educational Goals

Participants should understand:

- Model Compression
- Compression vs Inference
- CNN Pruning
- Evaluation of Pruning
- Deep Compression
- Quantization
- Sparsity
- Accuracy vs Model Size
- Latency
- Throughput
- Engineering trade-offs for Edge AI deployment

---

# 3. Workshop Scenario

The user is an AI Performance Engineer at Tata Motors.

**Phase 1 (Training - 10 min):**
- Students train a baseline weld inspection CNN on a small dataset
- They can start/stop/resume training via UI
- Goal: Understand model training, loss curves, accuracy progression
- Outcome: A trained model checkpoint

**Phase 2 (Optimization - 170 min):**
- The trained model cannot be deployed because it exceeds hardware limits
- Participants must optimize (compress, prune, quantize) until it satisfies deployment constraints
- Participants make engineering trade-off decisions
- Outcome: Production-ready optimized model

---

# 4. Personas

### Student

- No coding required
- Uses sliders and buttons
- Reads metrics and graphs
- Makes optimization decisions

### Instructor

- Explains engineering concepts
- Demonstrates optimization
- Reviews deployment decisions

---

# 5. Functional Modules

1. Streamlit Application
2. **Training Module** (NEW)
3. Optimization Engine
4. Dashboard
5. Charts
6. Deployment Profiles
7. Deployment Scoring
8. Recommendation Engine
9. Report Generator

---

# 6. Project Folder Structure

```text
Welding_Defect_Detection/
│
├── app.py                          # Main Streamlit entry point
├── config.py
├── requirements.txt
├── README.md
│
├── trainer/                        # NEW: Training module
│   ├── trainer.py                  # Training logic, start/stop/resume
│   ├── dataset.py                  # Weld inspection dataset loader
│   ├── model_builder.py            # MobileNetV2 architecture
│   └── __init__.py
│
├── simulator/
│   ├── optimizer.py
│   ├── metrics.py
│   ├── profiles.py
│   ├── scorer.py
│   └── __init__.py
│
├── ui/
│   ├── dashboard.py
│   ├── sidebar.py
│   ├── charts.py
│   ├── status.py
│   ├── styles.py
│   ├── training_ui.py              # NEW: Training phase UI
│   └── __init__.py
│
├── data/
│   ├── weld_defects/               # Training dataset (images + labels)
│   └── baseline_model/             # Pre-trained checkpoint (optional fallback)
│
├── assets/
├── checkpoints/                    # Student-saved training checkpoints
├── reports/
└── tests/
```

---

# 7. Screen Design

## Tab 1: Training Phase

**Title**: "Train Your Weld Inspection Model"

**Left Panel (Controls)**:
- Epochs slider (1-10)
- Batch size selector (8, 16, 32)
- Learning rate slider
- Start Training button
- Pause/Resume button
- Stop button
- Load Model dropdown (to resume from checkpoint)

**Right Panel (Visualizations)**:
- Real-time loss curve (training + validation)
- Accuracy progression
- Estimated time remaining
- Current epoch / batch progress bar
- Status text ("Training...", "Paused", "Complete")

**Bottom**:
- "Save Checkpoint" button
- List of saved checkpoints

---

## Tab 2: Optimization Phase (Dashboard)

**Sidebar Controls**:
- Model selection (Load from checkpoint)
- Deployment profile selector
- Optimization sliders (Compression, Pruning, Quantization)
- Optimize button
- Reset button

**Main Display - KPI Cards**:
- Accuracy
- Model Size
- Latency
- Throughput
- Sparsity
- Deployment Score
- Deployment Status

**Charts**:
- Accuracy vs Compression
- Latency vs Compression
- Model Size vs Compression
- Deployment Score gauge

---

# 8. Sidebar Controls

Deployment Profile

Compression Slider

Pruning Slider

Quantization Dropdown

Optimize Button

Reset Button

---

# 9. Optimization Engine

Input

- Compression
- Pruning
- Quantization
- Deployment

Output

- Accuracy
- Model Size
- Latency
- Throughput
- Sparsity
- Deployment Score
- Recommendation

The engine must be UI-independent.

---

# 10. Deployment Profiles

Weld Inspection Vision

Battery Inspection

Paint Inspection

Each profile has independent deployment constraints.

---

# 11. Dashboard Behaviour

Changing any optimization parameter must immediately update:

- KPI cards
- Graphs
- Deployment status
- Recommendation

---

# 12. Charts

Use Plotly.

**Training Phase**:
- Training Loss vs Epoch
- Validation Accuracy vs Epoch
- Learning Rate schedule (informational)
- Batch processing timeline

**Optimization Phase**:
- Accuracy vs Compression
- Latency vs Compression
- Model Size vs Compression
- Throughput vs Compression
- Gauge for Deployment Score

---

# 13. Engineering Rules

Compression should exhibit diminishing returns.

Pruning should preserve accuracy initially and degrade after a threshold.

Quantization should influence size, latency, and accuracy.

Deep Compression combines all three techniques.

---

# 14. Code Standards

- Python 3.11+
- Type hints
- Dataclasses
- Modular architecture
- SOLID principles where practical
- PEP8
- Docstrings
- Separation of concerns
- CPU-only (no CUDA/GPU code)
- Thread-safe for long-running training

---

# 15. Build Roadmap

Sprint 1
- config.py
- requirements.txt
- data/ folder with weld defect dataset

Sprint 2
- model_builder.py (MobileNetV2)
- dataset.py (data loader)
- trainer.py (training logic with start/stop/resume)

Sprint 3
- app.py (two-tab structure)
- training_ui.py

Sprint 4
- optimizer.py

Sprint 5
- dashboard.py
- charts.py (both training + optimization)

Sprint 6
- profiles.py
- scorer.py
- Optimization scoring

Sprint 7
- report generation
- checkpoint management

Sprint 8
- polish
- performance optimization on CPU

---

# 16. Testing

- Unit tests for optimizer
- Unit tests for trainer (start/stop/resume logic)
- Integration tests for UI updates
- CPU performance benchmarks
- Smoke test: `streamlit run app.py`
- Training smoke test: Verify training runs for 1 epoch without errors
- Bundle portability test: Extract bundle on clean machine, verify it runs

---

# 17. Model & Technical Decisions

**Model Architecture**: MobileNetV2
- Reason: Lightweight (14MB base), CPU-trainable, industry-relevant, good for optimization teaching
- Input: 224x224 RGB images
- Output: Binary classification (defect/no-defect) or multi-class (defect type)
- Parameters: ~3.5M (compressible to 0.5-1M with optimization)

**Dataset**: Weld Defect Inspection
- **Source**: Procedurally generated synthetic data (reproducible, portable, no licensing)
- **Size**: 800 labeled images (400 defect, 400 normal) generated on first run
- **Generation**: ~1 minute to generate; cached in `data/weld_defects/`
- **Defect Types**: Cracks, porosity, undercut, spatter, incomplete fusion
- **Split**: 70% train, 15% val, 15% test
- **Preprocessing**: Resize to 224x224, normalize to [0,1]
- **Storage**: Bundled as generator code only (~2MB) → expands to ~50MB after generation

**Training on CPU**:
- Target: 5-10 min training on typical laptop
- Strategy: Use smaller dataset, fewer epochs (5-10), larger batch size
- Framework: PyTorch (CPU-optimized)

---

## 17. Stretch Goals

- PDF report with optimization journey
- CSV export of metrics
- Hardware profile comparison (edge vs cloud)
- Animated training loss visualization
- Model architecture visualization
- Export optimized model to ONNX

---

# 18. Prompt Pack for GitHub Copilot

For every file:

1. Generate one file only.
2. Produce complete, runnable code.
3. No TODO placeholders.
4. Keep UI separate from business logic.
5. Use Plotly for charts.
6. Use Streamlit best practices.
7. Add docstrings and comments.
8. Ensure imports are correct.
9. Write maintainable production-quality code.
10. Wait for review before generating the next file.

---

# 18. System Requirements

**Minimum (Student Laptops)**:
- OS: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- Python: 3.11 or 3.12
- RAM: 4GB (8GB recommended)
- Storage: 1GB (for code + dataset + checkpoints)
- Processor: Any CPU (training will be slow on older CPUs, but functional)
- GPU: Not required
- Internet: Not required (fully offline)

**Bundle Contents**:
- Python environment file (requirements.txt)
- All source code
- Pre-packaged weld defect dataset (~100-200MB)
- Pre-built checkpoint (optional fallback if training fails)
- README with setup instructions

---

# 19. Acceptance Criteria

The project is complete when:

- The application launches with `streamlit run app.py`
- **Training phase works**: Students can train for 5-10 min, save checkpoint, see loss curves
- **Training is controllable**: Start/Stop/Pause/Resume all work correctly
- **Optimization phase works**: All controls are interactive
- **Bundle is portable**: Extract on any laptop with Python 3.11+, runs without internet
- **CPU-only**: No GPU code; training completes in 10 min on typical laptop CPU
- The optimization engine updates all metrics
- Charts update automatically (both training and optimization)
- Deployment profiles work correctly
- Participants can complete the workshop without writing code
- The UI resembles a professional engineering dashboard with two clear phases
