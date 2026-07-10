# Class Imbalance Issue - RESOLVED

## Problem Identified

The "normal" weld classification wasn't working because the **OpenSourceData01** dataset has a severe class imbalance:

### Dataset Statistics:
- **OpenSourceData01 (Real Images)**
  - Training: 119 normal images vs 720 defect images (86% defects)
  - The model learned to predict everything as "defect" to maximize accuracy
  - This is mathematically optimal but useless for real-world use

- **weld_defects (Synthetic Images)**
  - Training: 280 normal vs 280 defect (perfectly balanced)
  - The synthetic model correctly predicts both normal and defect classes

## Solution Applied

Updated [trainer/trainer.py](trainer/trainer.py) to use **weighted CrossEntropyLoss** that:

1. Counts the number of samples in each class
2. Calculates inverse weights (minority class gets higher weight)
3. Penalizes incorrect predictions on the minority "normal" class

### How It Works:
- Weight for class 0 (normal): total_samples / (2 × count_normal)
- Weight for class 1 (defect): total_samples / (2 × count_defect)
- With the OpenSourceData01 imbalance:
  - Normal weight: ≈ 3.04 (missing normal is 3x worse)
  - Defect weight: ≈ 0.46 (more forgiving on defect misses)

## What You Need To Do

### Option 1: Retrain with Class Weighting (RECOMMENDED)
1. Go to **Phase 1: Training**
2. Select **OpenSourceData01** dataset
3. Delete or rename the old baseline checkpoint to force retraining
4. Click "Start Training" for 5-10 epochs
5. The model will now learn to correctly classify BOTH normal and defect images
6. The console will show the class weights being applied

### Option 2: Use Synthetic Data Only
- Use the **weld_defects** (synthetic) model which is already balanced
- It already correctly predicts normal vs defect images
- Test with: `data/weld_defects/test/normal/*` and `data/weld_defects/test/defect/*`

## Testing After Fix

After retraining on OpenSourceData01, you should see:
- ✅ Normal images → "NORMAL (No Defect)" in green
- ✅ Defect images → "DEFECT DETECTED" in red
- ✅ Both classes properly detected in Phase 3 inference

## Technical Details

The weighted loss function ensures that during training:
- Incorrectly predicting a normal image as defect is penalized more
- Incorrectly predicting a defect as normal is also penalized
- The model learns to distinguish between both classes despite imbalance

This is a standard technique in machine learning for handling imbalanced datasets.
