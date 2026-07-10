# Welding Defect Detection - Architecture & Design

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Application                       │
│                        (app.py)                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────┐    ┌──────────────────────────┐   │
│  │   Phase 1: Training      │    │  Phase 2: Optimization   │   │
│  ├─────────────────────────┤    ├──────────────────────────┤   │
│  │ • Dataset generation    │    │ • Model loading          │   │
│  │ • Model creation        │    │ • Compression slider     │   │
│  │ • Training loop         │    │ • Pruning slider         │   │
│  │ • Real-time metrics     │    │ • Quantization selector  │   │
│  │ • Checkpoint save/load  │    │ • Real-time scoring      │   │
│  │ • Loss curve display    │    │ • Trade-off visualization│   │
│  └─────────────────────────┘    └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
┌──────────────────────────┐        ┌──────────────────────────┐
│   Training Module         │        │ Optimization Engine      │
│   (trainer/)              │        │ (simulator/)             │
├──────────────────────────┤        ├──────────────────────────┤
│ • model_builder.py       │        │ • optimizer.py           │
│ • dataset.py             │        │ • metrics.py             │
│ • trainer.py             │        │ • scorer.py              │
│                          │        │ • profiles.py            │
└──────────────────────────┘        └──────────────────────────┘
         │                                    │
         ▼                                    ▼
┌──────────────────────────┐        ┌──────────────────────────┐
│   Data Layer              │        │ UI Components            │
│   (data/)                │        │ (ui/)                    │
├──────────────────────────┤        ├──────────────────────────┤
│ • generator.py           │        │ • dashboard.py           │
│ • weld_defects/          │        │ • charts.py              │
│   (generated images)     │        │                          │
└──────────────────────────┘        └──────────────────────────┘
```

---

## Module Dependencies

### Training Pipeline (Phase 1)
```
config.py
    ↓
data/generator.py ──→ data/weld_defects/ (800 synthetic images)
    ↓
trainer/model_builder.py ──→ MobileNetV2 model
    ↓
trainer/dataset.py ──→ PyTorch DataLoaders
    ↓
trainer/trainer.py ──→ Training engine with checkpoints
    ↓
app.py (Phase 1 UI)
```

### Optimization Pipeline (Phase 2)
```
config.py
    ↓
trainer/trainer.py ──→ Load trained checkpoint
    ↓
simulator/optimizer.py
    ├── simulator/metrics.py (metric calculations)
    └── OptimizationEngine (state management)
    ↓
simulator/scorer.py
    ├── simulator/profiles.py (deployment constraints)
    └── DeploymentScorer (constraint validation)
    ↓
ui/dashboard.py ──→ Display controls & KPI cards
    ↓
ui/charts.py ──→ Plotly visualizations
    ↓
app.py (Phase 2 UI) ──→ Streamlit interface
```

---

## Data Flow

### Phase 1: Training

```
Student Input
    ↓
┌─ Epochs slider (1-20)
├─ Batch size selector (8-64)
└─ Learning rate selector
    ↓
trainer.train()
    ├─ load_dataset(train, val)
    ├─ create_model()
    ├─ train_epoch() × epochs
    │   ├─ forward pass
    │   ├─ backward pass
    │   ├─ update weights
    │   └─ track metrics
    ├─ validate_epoch()
    │   ├─ forward pass
    │   └─ compute loss/accuracy
    └─ save_checkpoint()
    ↓
Display
    ├─ Loss curves (training + validation)
    ├─ Accuracy progression
    ├─ Current epoch / time remaining
    ├─ Checkpoint list
    └─ Model info
```

### Phase 2: Optimization

```
Student Input
    ├─ Deployment profile selection
    ├─ Compression slider (0-100%)
    ├─ Pruning slider (0-95%)
    └─ Quantization bits (4/8/16/32)
    ↓
OptimizationEngine.apply_optimization()
    ├─ set_compression(value)
    ├─ set_pruning(value)
    └─ set_quantization_bits(bits)
    ↓
MetricsCalculator.calculate_metrics()
    ├─ accuracy = f(compression, pruning, quantization)
    ├─ size_mb = f(compression, pruning, quantization)
    ├─ latency_ms = f(compression, pruning, quantization)
    ├─ throughput_fps = 1000 / latency_ms
    └─ sparsity = pruning
    ↓
DeploymentScorer.score()
    ├─ check_hard_constraints()
    │   ├─ accuracy ≥ min_accuracy
    │   ├─ size ≤ max_size
    │   └─ latency ≤ max_latency
    ├─ check_soft_constraints()
    │   ├─ preferred_accuracy
    │   └─ preferred_throughput
    ├─ calculate_scores()
    │   ├─ accuracy_score (0-100)
    │   ├─ size_score (0-100)
    │   ├─ latency_score (0-100)
    │   └─ throughput_score (0-100)
    └─ generate_recommendations()
    ↓
Display
    ├─ KPI cards (with deltas)
    ├─ Sparsity metrics
    ├─ Constraint status (green/red)
    ├─ Deployment gauge
    ├─ Progress bars
    ├─ Trade-off charts
    ├─ Recommendations
    └─ Deploy button (if constraints met)
```

---

## Configuration Hierarchy

```
config.py
├── TrainingConfig
│   ├── model_name = "mobilenetv2"
│   ├── num_classes = 2
│   ├── input_size = 224
│   ├── num_epochs = 10 (default)
│   ├── batch_size = 16 (default)
│   ├── learning_rate = 0.001 (default)
│   └── checkpoint_dir = CHECKPOINTS_DIR
│
├── DataGenerationConfig
│   ├── num_defect_images = 400
│   ├── num_normal_images = 400
│   ├── image_size = 224
│   └── defect_types = [crack, porosity, undercut, spatter, incomplete_fusion]
│
├── OptimizationConfig
│   ├── compression_range = (0.0, 1.0)
│   ├── pruning_range = (0.0, 0.95)
│   └── quantization_bits = [4, 8, 16, 32]
│
└── DeploymentProfile
    ├── Weld Vision (max 20MB, 100ms, 92% min)
    ├── Battery Inspection (max 50MB, 200ms, 95% min)
    ├── Paint Inspection (max 100MB, 500ms, 90% min)
    └── Edge Extreme (max 5MB, 500ms, 85% min)
```

---

## State Management (Streamlit Session State)

```
st.session_state
├── trainer: WeldDefectTrainer
│   ├── model
│   ├── training_state
│   │   ├── current_epoch
│   │   ├── is_training
│   │   ├── train_losses []
│   │   ├── val_losses []
│   │   ├── train_accuracies []
│   │   └── val_accuracies []
│   └── checkpoints []
│
├── dataset_ready: bool
│
├── optimizer: OptimizationEngine
│   ├── current_state
│   │   ├── compression
│   │   ├── pruning
│   │   └── quantization_bits
│   ├── current_metrics
│   │   ├── accuracy
│   │   ├── model_size_mb
│   │   ├── latency_ms
│   │   ├── throughput_fps
│   │   ├── sparsity
│   │   └── compression_ratio
│   └── history
│       ├── compressions []
│       ├── prunings []
│       ├── quantizations []
│       ├── accuracies []
│       ├── sizes_mb []
│       └── latencies_ms []
│
├── scorer: DeploymentScorer
│
└── deployment_profile: DeploymentProfile
```

---

## Key Algorithms

### 1. Accuracy Impact Calculation

```
accuracy = baseline_accuracy - accuracy_loss

accuracy_loss = compression_loss + pruning_loss + quantization_loss

compression_loss = compression² × 0.25  # Quadratic decay

pruning_loss = {
    if pruning < 0.3:
        (pruning / 0.3) × 0.05
    else:
        0.05 + ((pruning - 0.3) / 0.7) × 0.15
}

quantization_loss = (32 - bits) / 32 × 0.1

Final accuracy = max(0.5, baseline - accuracy_loss)  # Floor at 50%
```

### 2. Model Size Calculation

```
compression_factor = (1 - compression × 0.8)
pruning_factor = (1 - pruning × 0.7)
quantization_factor = bits / 32

size = baseline_size × compression_factor × pruning_factor × quantization_factor
size = max(0.5, size)  # Floor at 0.5MB
```

### 3. Latency Calculation

```
compression_speedup = 1 - (compression × 0.5)
pruning_speedup = 1 - (pruning × 0.6)
quantization_speedup = (bits / 32)^0.8

latency = baseline_latency × compression_speedup × pruning_speedup × quantization_speedup
latency = max(5.0, latency)  # Floor at 5ms
```

### 4. Deployment Score

```
if all_hard_constraints_met:
    total_score = weighted_average(accuracy, size, latency, throughput)
else:
    total_score = weighted_average(...) × (1 - failures × 0.3)

is_deployable = accuracy ≥ min AND size ≤ max AND latency ≤ max
```

---

## UI Component Hierarchy

```
Streamlit App (app.py)
├── Phase 1 Tab
│   ├── Training controls section
│   │   ├── Dataset section
│   │   │   ├── Dataset generation
│   │   │   └── Dataset stats
│   │   ├── Configuration section
│   │   │   ├── Epochs slider
│   │   │   ├── Batch size selector
│   │   │   └── Learning rate selector
│   │   ├── Training execution
│   │   │   ├── Start Training button
│   │   │   └── Progress display
│   │   ├── Checkpoint management
│   │   │   ├── Load checkpoint dropdown
│   │   │   └── Load button
│   │   └── Model info display
│   │       ├── Model type
│   │       ├── Model size
│   │       ├── Parameters count
│   │       └── Output classes
│   │
│   └── (from dashboard.py)
│
├── Phase 2 Tab
│   ├── Deployment profile selection
│   │   ├── Profile dropdown
│   │   ├── Profile details expander
│   │   └── Constraint display
│   ├── Optimization controls (from dashboard.py)
│   │   ├── Compression slider
│   │   ├── Pruning slider
│   │   └── Quantization selector
│   ├── KPI cards (from dashboard.py)
│   │   ├── Accuracy card
│   │   ├── Size card
│   │   ├── Latency card
│   │   └── Throughput card
│   ├── Sparsity metrics (from dashboard.py)
│   ├── Optimization summary (from dashboard.py)
│   ├── Constraint progress (from dashboard.py)
│   ├── Deployment gauge (from charts.py)
│   ├── Constraint status (from dashboard.py)
│   ├── Deployment readiness (from dashboard.py)
│   ├── Trade-off visualizations (from charts.py)
│   │   ├── Metric evolution
│   │   └── Constraint radar
│   ├── Recommendations (from dashboard.py)
│   └── Action buttons
│       ├── Save Optimization
│       ├── Reset to Baseline
│       └── Ready to Deploy
│
└── Sidebar
    ├── Workshop progress tracker
    └── Tips for each phase
```

---

## Performance Characteristics

### Training Phase
- **Dataset Generation**: ~1 minute (one-time)
- **Training Per Epoch**: ~30-60 seconds on CPU
- **10 Epochs**: ~5-10 minutes
- **Memory Usage**: ~2-3 GB
- **Disk Usage**: ~50MB (data) + ~14MB per checkpoint

### Optimization Phase
- **Metric Calculation**: <1 millisecond
- **Constraint Scoring**: <1 millisecond
- **UI Update**: Instant (Streamlit reactive)
- **Memory**: Constant (no training)
- **All operations**: Real-time, no noticeable latency

---

## Error Handling Strategy

```
Training Phase
├── Dataset generation failures
│   └── Graceful fallback with error message
├── Model creation failures
│   └── Detailed error reporting
├── Training interruption (stop/pause)
│   └── Checkpoint saved before stopping
└── Checkpoint load failures
    └── User-friendly error + list available checkpoints

Optimization Phase
├── Invalid constraint values
│   └── Clamp to valid range
├── Profile selection errors
│   └── Default to first profile
└── Metric calculation edge cases
    └── Bounds checking and floor/ceiling
```

---

## Testing Strategy

1. **Unit Tests** (optional, future)
   - Metric calculations
   - Constraint validation
   - Optimizer state management

2. **Integration Tests** (optional, future)
   - Training → checkpoint → load → optimize
   - Full workflow from dataset to deployment

3. **Manual Testing** (current)
   - `demo.py` script tests all components
   - Streamlit UI tested through browser

4. **Smoke Tests**
   ```bash
   streamlit run app.py  # Full app test
   python demo.py        # Optimization engine test
   ```

---

## Extensibility Points

Designed for easy future enhancements:

1. **New Optimization Techniques**
   - Add to `OptimizationEngine.apply_optimization()`
   - Update metric calculations in `metrics.py`

2. **New Deployment Profiles**
   - Add to `PROFILES` in `profiles.py`
   - Automatically available in UI

3. **New Visualizations**
   - Add to `charts.py`
   - Display in Phase 2 tab

4. **Model Architecture Changes**
   - Create new class in `model_builder.py`
   - Update `trainer.py` to support

5. **Report Generation**
   - Create `ui/reporter.py`
   - Hook into Phase 2 export buttons

---

## Code Quality

- ✅ **Type Hints**: Throughout (Python 3.11+)
- ✅ **Docstrings**: All modules and functions
- ✅ **Error Handling**: Try-catch with user feedback
- ✅ **Modularity**: Clear separation of concerns
- ✅ **Testability**: Independent, mockable components
- ✅ **Maintainability**: Clear naming, logical structure
- ✅ **Performance**: Optimized for CPU inference

---

## Security & Safety

- ✅ **No External Dependencies**: All data bundled
- ✅ **No User Uploads**: Only local data
- ✅ **No Networking**: Fully offline
- ✅ **Safe Defaults**: Conservative initial values
- ✅ **Bounds Checking**: All user inputs validated
- ✅ **Error Recovery**: Graceful failure handling

---

This architecture provides a solid foundation for an interactive, educational AI optimization workshop that is both robust and user-friendly.
