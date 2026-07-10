# OpenSourceData01 - Dataset Compatibility Issue & Solution

## Problem Summary

The **OpenSourceData01 dataset has a flat structure** that is **incompatible** with the Welding Defect Detection trainer.

### What's Wrong?

**Current Structure (INCOMPATIBLE):**
```
data/OpenSourceData01/
├── train/          ← All images mixed here (no class folders)
├── valid/          ← All images mixed here
└── test/           ← All images mixed here
```

**Required Structure (HIERARCHICAL):**
```
data/OpenSourceData01/
├── train/
│   ├── normal/     ← Good welds (label 0)
│   └── defect/     ← Bad welds (label 1)
├── valid/
│   ├── normal/
│   └── defect/
└── test/
    ├── normal/
    └── defect/
```

### Why Does It Fail?

When the trainer tries to load data, it looks for `train/normal/` and `train/defect/` subdirectories:
- Finds **0 images** because they don't exist
- Returns error: "ValueError: num_samples = 0"
- Training cannot proceed

## Solution: Restructure the Dataset

### Option 1: Automatic Conversion (Recommended)

**Run the conversion script:**
```bash
python data/restructure_flat_dataset.py --dataset OpenSourceData01
```

**What it does:**
- Scans all images in train/, valid/, test/ folders
- Classifies images by filename keywords:
  - **Normal keywords:** good, carbon, stick, tig, mig, ok, perfect, excellent, normal
  - **Defect keywords:** bad, crack, spatter, slag, porosity, defect, poor, not, overlap
- Creates normal/ and defect/ subdirectories
- Copies images to appropriate folders
- Shows classification summary

**Custom keywords:**
```bash
python data/restructure_flat_dataset.py --dataset OpenSourceData01 \
  --normal-keywords "good,perfect,ok" \
  --defect-keywords "bad,crack,defect"
```

### Option 2: Manual Classification

If automatic classification doesn't work well:

1. Create folder structure:
   ```
   train/normal/    ← Move good weld images here
   train/defect/    ← Move bad weld images here
   valid/normal/    ← Split validation good welds
   valid/defect/    ← Split validation bad welds
   test/normal/     ← Split test good welds
   test/defect/     ← Split test bad welds
   ```

2. Move images based on visual inspection
3. Keep splits: ~70% train, ~15% valid, ~15% test

## Verification

After conversion, verify the structure:

```bash
# Linux/Mac
find data/OpenSourceData01 -type d | head -20

# Windows PowerShell
Get-ChildItem -Path data/OpenSourceData01 -Recurse -Directory | Select-Object -First 20
```

You should see:
- OpenSourceData01/train/normal/  (with .jpg files)
- OpenSourceData01/train/defect/  (with .jpg files)
- OpenSourceData01/valid/normal/  (with .jpg files)
- OpenSourceData01/valid/defect/  (with .jpg files)
- OpenSourceData01/test/normal/   (with .jpg files)
- OpenSourceData01/test/defect/   (with .jpg files)

## Try Again

After conversion:

1. **Restart the Streamlit app** or refresh the page
2. **Select OpenSourceData01** from the dataset dropdown
3. The app will now show: "Dataset structure is valid!"
4. **Start training** - it should now find images and proceed

## Troubleshooting

### Script won't run
- Ensure you're in the project root directory (`Welding_Defect_Detection/`)
- Check that `python` is in your PATH
- Try: `python -m data.restructure_flat_dataset --dataset OpenSourceData01`

### Images not classified correctly
- Check the console output - it shows which images went where
- Use `--dry-run` flag to preview without making changes:
  ```bash
  python data/restructure_flat_dataset.py --dataset OpenSourceData01 --dry-run
  ```
- Try custom keywords if classification is poor

### Still getting validation errors
- Delete the normal/defect folders that were created
- Check file permissions (ensure Python can read files)
- Check that split directories (train/, valid/, test/) are read-only
- Try manual classification

## Dataset Information

- **Source:** https://www.kaggle.com/code/jayeshm0103/weld-defect-detection-and-classification
- **Total Images:** ~600+ images
- **Split:** ~train/valid/test (already split, but needs class organization)
- **Image Format:** JPG, PNG
- **Classes:** Good welds vs. Bad welds (binary classification)

## Why This Fix is Needed

The Welding Defect Detection system is designed for:
- **Education:** Teaching how model optimization works
- **Clear structure:** Binary classification (normal/defect) for weld inspection
- **Reproducibility:** Standardized dataset format

The hierarchical structure (normal/defect subdirectories) is essential for:
- Loading class labels correctly
- Balancing training data
- Supporting different optimization techniques
- Academic rigor and clarity
