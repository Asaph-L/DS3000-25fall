# Plant Disease Detection - Model Usage Guide

## Overview

This project contains two main notebooks:
1. **FirstDraft.ipynb** - Complete training pipeline (train models from scratch)
2. **ModelInference.ipynb** - Model loading and inference (reuse trained models)

---

## 📁 File Structure

```
DS3000-25fall/
├── FirstDraft.ipynb              # Training notebook
├── ModelInference.ipynb          # Inference notebook
├── resnet50_complete.pth         # Saved ResNet50 model (generated after training)
├── efficientnet_b0_complete.pth  # Saved EfficientNet model (generated after training)
├── data/
│   ├── archive/Dataset/          # Original dataset
│   └── clean/                    # Cleaned dataset (generated)
└── *.html                        # Exported reports (optional)
```

---

## 🚀 Workflow

### Option 1: First Time Training

**Use Case**: You don't have trained models yet, or want to retrain from scratch.

1. **Open `FirstDraft.ipynb`**
2. **Run all cells sequentially** (Sections 1-6):
   - Section 1: Data import
   - Section 2: Data visualization
   - Section 3: Data cleaning (generates `data/clean/`)
   - Section 4: Model architecture definitions
   - Section 5: **Training** (generates `.pth` files)
   - Section 6: Performance comparison & evaluation
   - Section 7: Export to HTML report

3. **Output files**:
   - `resnet50_complete.pth` (~100 MB)
   - `efficientnet_b0_complete.pth` (~80 MB)
   - `PlantDisease_Training_Report_YYYYMMDD_HHMMSS.html`

**Training Time**: ~30-60 minutes (with GPU) or 2-4 hours (CPU only)

---

### Option 2: Load Pre-trained Models (Fast)

**Use Case**: You already have trained models and want to:
- Make predictions on new images
- Re-evaluate on the dataset
- Generate reports without retraining

1. **Open `ModelInference.ipynb`**
2. **Run all cells sequentially** (Sections 1-11):
   - Sections 1-4: Setup and load models
   - Section 5-6: Inference functions
   - Section 6: Single image prediction example
   - Section 7: Full dataset evaluation
   - Section 8: Confusion matrices
   - Section 9: Classification reports
   - Section 10: Training history visualization
   - Section 11: Export to HTML report

3. **Output files**:
   - `ModelInference_Report_YYYYMMDD_HHMMSS.html`

**Loading Time**: ~5-10 seconds (instant model loading)

---

## 📊 What's Saved in `.pth` Files?

Each `*_complete.pth` checkpoint contains:

```python
{
    'model_state_dict': <trained weights>,
    'history': {
        'train_loss': [...],
        'train_acc': [...],
        'val_loss': [...],
        'val_acc': [...]
    },
    'num_classes': 23,
    'class_to_idx': {'Apple___Apple_scab': 0, ...},
    'model_name': 'resnet50' or 'efficientnet_b0'
}
```

This means you can reload the **exact model** with all training history and class mappings!

---

## 🔧 Requirements

Install dependencies:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn plotly pillow tqdm ipython
pip install nbconvert  # For HTML export
```

---

## 💡 Usage Examples

### Example 1: Train Once, Use Many Times

```bash
# Day 1: Train models (run FirstDraft.ipynb)
# → Generates resnet50_complete.pth and efficientnet_b0_complete.pth

# Day 2+: Load models instantly (run ModelInference.ipynb)
# → No need to retrain!
```

### Example 2: Make Predictions on New Images

In `ModelInference.ipynb`, modify Section 6:

```python
# Your custom image
my_image = "path/to/my/plant_image.jpg"

# Get predictions
resnet_preds, img = predict_single_image(resnet_model, my_image, idx_to_class)
print(f"Prediction: {resnet_preds[0]['class']} ({resnet_preds[0]['probability']*100:.1f}%)")
```

### Example 3: Compare Models on Specific Classes

In `ModelInference.ipynb`, after Section 7:

```python
# Filter results for a specific class
target_class = "Tomato_healthy"
class_idx = list(idx_to_class.values()).index(target_class)

# Get predictions for this class
mask = (resnet_eval['labels'] == class_idx)
class_acc = 100 * np.mean(resnet_eval['predictions'][mask] == resnet_eval['labels'][mask])
print(f"ResNet50 accuracy on {target_class}: {class_acc:.2f}%")
```

---

## 📈 HTML Export

Both notebooks include HTML export functionality (last section).

**To generate reports**:

1. Run all cells in the notebook
2. Execute the final "Export Notebook to HTML" cell
3. Find the generated `.html` file in the same directory

**HTML reports include**:
- All code cells
- All outputs (tables, plots, metrics)
- Embedded images
- Shareable with anyone (no Python/Jupyter required)

**Manual export** (if the cell fails):
- In Jupyter: `File > Download as > HTML`
- Command line: `jupyter nbconvert --to html FirstDraft.ipynb`

---

## 🎯 Quick Reference

| Task | Notebook | Section | Time |
|------|----------|---------|------|
| Train models from scratch | FirstDraft.ipynb | 1-6 | 30-60 min |
| Load trained models | ModelInference.ipynb | 1-4 | <10 sec |
| Predict single image | ModelInference.ipynb | 6 | <1 sec |
| Evaluate full dataset | ModelInference.ipynb | 7 | 2-5 min |
| Generate HTML report | Either notebook | Last section | <1 min |
| Cross-validation | FirstDraft.ipynb | 6.4 (set RUN_CV=True) | 2-4 hours |

---

## ⚠️ Important Notes

1. **Model files must exist**: `ModelInference.ipynb` requires the `.pth` files generated by `FirstDraft.ipynb`

2. **Class mapping consistency**: The saved models remember the class order from training. Don't modify `data/clean/` structure after training.

3. **Device compatibility**: Models saved on GPU can be loaded on CPU (and vice versa) thanks to `map_location=device`

4. **File size**: Each checkpoint is ~100-200 MB. Don't commit to GitHub unless using Git LFS.

5. **Version control**: Save different model versions with timestamps:
   ```python
   save_model_with_metadata(model, history, f'resnet50_v2_{timestamp}', ...)
   ```

---

## 🐛 Troubleshooting

### "FileNotFoundError: resnet50_complete.pth"
→ Run `FirstDraft.ipynb` Section 5 first to train and save models

### "CUDA out of memory"
→ Reduce `BATCH_SIZE` in the configuration cell (try 32 or 16)

### "nbconvert not found"
→ Install with: `pip install nbconvert`

### "Predictions are wrong"
→ Check that `data/clean/` structure matches the training data

---

## 📝 Summary

- **FirstDraft.ipynb**: Complete pipeline (train, evaluate, save)
- **ModelInference.ipynb**: Fast inference (load, predict, analyze)
- Both notebooks export to HTML for easy sharing
- Models are saved with full metadata for reproducibility

**Best Practice**: Train once with `FirstDraft.ipynb`, then use `ModelInference.ipynb` for all future inference tasks!

