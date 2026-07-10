# Delete Trained Models Feature

## Overview

You can now delete trained model checkpoints (both baseline and optimized) from all three phases of the application.

## Where to Delete Models

### Phase 1: Training
- **Location**: "Step 4: Manage Trained Models"
- **Controls**: 
  - ✅ Load Selected Model (load the model)
  - ℹ️ Info (show model details)
  - 🗑️ Delete (remove the model)

**What happens when you delete:**
- Model checkpoint file is removed from `checkpoints/` directory
- If it was the currently loaded baseline model, Phase 2 state is cleared
- Session state is updated
- You can retrain to create a new model

### Phase 2: Optimization & Retraining
- **Location**: "Save Optimized Model" section
- **Controls**:
  - 💾 Save Optimized Model (save the optimized version)
  - 🗑️ Delete (remove the optimized checkpoint)

**What happens when you delete:**
- Optimized model checkpoint is removed
- You can regenerate by re-optimizing and saving

### Phase 3: Real-time Inference
- **Location**: "Step 1: Load Your Model Checkpoint"
- **Controls**:
  - ✅ Load Model (load for inference)
  - ℹ️ Info (show model details)
  - 🗑️ Delete (remove the model)

**What happens when you delete:**
- Model checkpoint is removed from `checkpoints/` directory
- You can still select another model for inference
- Inference engine automatically loads the new selected model

## Delete Confirmation

All delete operations show a **confirmation dialog**:
- ⚠️ Warning message with model name
- "This action cannot be undone" caption
- Two buttons:
  - 🗑️ Yes, Delete - Confirms and deletes
  - ❌ Cancel - Aborts the deletion

## Available Models to Delete

Models appear as friendly names in the dropdown:
- **Baseline models**: `📦 Baseline: ModelName (dataset) - N epochs`
- **Optimized models**: `⚡ Optimized: ModelName (dataset) - N epochs - {compression}c_{pruning}p_{quantization}q`

Examples:
- `📦 Baseline: MobileNetV2 (opensourcedata01) - 5 epochs`
- `⚡ Optimized: MobileNetV2 (opensourcedata01) - 5 epochs - 0.3c_0.2p_8q`

## Technical Details

### Files Affected
- **Location**: `checkpoints/` directory
- **File extension**: `.pt` (PyTorch checkpoint format)
- **Naming convention**: `{model}_{dataset}_{epochs}ep[_{optimization_params}].pt`

### Session State Management
When a model is deleted:
1. File is removed from disk using `Path.unlink()`
2. If it was the active baseline: `st.session_state.baseline_training_info` is cleared
3. Training status is reset if necessary
4. `st.rerun()` refreshes the UI

### Error Handling
- If deletion fails: error message is displayed
- If file already deleted: information message shown
- Exception details are logged to console

## Use Cases

### Clean Up Old Models
When you've trained multiple versions, delete the ones you don't need to keep the checkpoints directory organized.

### Re-train with Different Parameters
Delete the old baseline model, then retrain with new parameters (epochs, learning rate, etc.).

### Free Up Disk Space
Each model takes ~20-30 MB. Delete unused models to reclaim space.

### Remove Failed Experiments
If a training or optimization didn't produce good results, delete it and try again.

## Notes

- Deletion is **permanent** - there is no undo or recovery
- Always confirm before deleting important models
- Phase 1 reset button also clears all model state without deleting files
- You need at least one baseline model to access Phase 2 and Phase 3
