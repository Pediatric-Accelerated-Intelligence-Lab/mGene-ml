
# Facial Image Analysis Pipeline

This repository contains a modular pipeline for facial image preprocessing, feature extraction, and classification using Local Binary Patterns (LBP), Generalized Procrustes Analysis, and Linear Discriminant Analysis (LDA).

---

## 🛠️ Installation Guide

### 📌 Operating System
- **Tested on:** Ubuntu 18.04.5 LTS (64-bit)
- **Other supported OS:** Any Linux-based system; macOS support possible with minor adjustments.
- **Windows support:** Not officially tested; may require `pathlib` and `OpenCV` compatibility fixes.

### 💻 Programming Language
- **Language:** Python 3.6.9
- **Environment suggestion:** Use a virtual environment (e.g., `venv` or `conda`) to isolate dependencies.

### 📦 Software Dependencies

The following Python packages are required:

| Package            | Version (suggested) | Notes                            |
|--------------------|---------------------|----------------------------------|
| `numpy`            | ≥ 1.19              | Matrix and array operations      |
| `opencv-python`    | ≥ 4.5               | Image processing                 |
| `scikit-learn`     | ≥ 0.22              | LDA, cross-validation, SVM       |
| `scipy`            | ≥ 1.4               | Statistical tests (Mann-Whitney) |
| `pandas`           | ≥ 1.0               | Exporting results to Excel       |
| `openpyxl`         | ≥ 3.0               | Excel writing support            |
| `matplotlib`       | (optional)          | For visual debugging/plots       |

#### 🧩 Installation Command

```bash
python3.6 -m venv venv
source venv/bin/activate
pip install numpy opencv-python scikit-learn scipy pandas openpyxl
```

---

### ⚙️ Non-Standard Hardware or Resources

| Resource          | Requirement                             |
|-------------------|------------------------------------------|
| GPU               | ❌ Not required                          |
| RAM               | ✅ Minimum 4 GB recommended              |
| Disk              | ✅ At least 1 GB free (for images, results) |
| Display           | Optional GUI display if using OpenCV's `imshow` |

---

### ⏱️ Typical Installation Time

| Phase                          | Estimated Time (on modern machine) |
|-------------------------------|------------------------------------|
| Python venv creation          | 10–15 seconds                      |
| Package installation (pip)    | 1–3 minutes (depending on network) |
| Script preprocessing runtime  | 10–30 seconds per image (if large) |

---

### 📁 Folder Expectations

Your pipeline assumes:
- Input image folders with consistent naming.
- Output folders for storing LBP features, aligned images, and Excel results.
- Modify paths in `FacialImageAnalysisPipeline.py` if running on a different directory structure.

---

### ✅ Final Check

After setup, run:

```bash
python FacialImageAnalysisPipeline.py
```

To verify everything is installed correctly and the pipeline works end-to-end.
