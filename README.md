
# Facial Image Analysis Pipeline

This repository contains a modular pipeline for facial image preprocessing, feature extraction, and classification, along with tools for visualizing appearance features and generating average facial representations.

---


---

## 🧩 Main Pipeline Description

```plaintext
BEGIN FacialImageAnalysisPipeline

IMPORT MachineLearningLibrary

// Step 1: Initialize folder structure and preprocess images
FUNCTION Initialize():
    createOutputFolderStrucure() // Create required folders
    rescaleImagesAlignLandmarks(): // Rescales and aligns all images in the dataset
    FOR each image IN dataset:
        Convert RGB image to grayscale image
        normalizeImageAndLandmarks() // Generalized Procrustes Analysis for single image. Output is aligned and cropped face region with 150x150 resolution
        SaveLocalBinaryPatterns() // Calculate and save the local binary patterns

// Step 2: Compute features and prepare for cross-validation
FUNCTION CalculateFeaturesForCrossValidation():
    CalculateLDAMatrices_CrossValidation() // Create matrices used to calculate the appearance features from local binary patterns 
    FOR each image IN dataset:
        GetFeatures() // Calculate all features including the appearance and geometric features

// Step 3: Perform leave-one-out cross-validation using the calculated features
FUNCTION CrossValidate():
    Perform Mann Whitney U test for all the features
    For k FROM 1 TO maximum number of features to select:
        Standardize features by removing the mean and scaling to unit variance
        Use Recursive Feature Elimination with a linear SVM model to select k features
        Perform leave-one-out cross-validation using the selected k features

// Step 4: Export the results to an Excel file
FUNCTION ExportCrossValidationToExcel()

// Main pipeline
CALL Initialize()
CALL CalculateFeaturesForCrossValidation()
CALL CrossValidate()
CALL ExportCrossValidationToExcel()

END
```

---

## 🧪 Other Functions

**Appearance Feature Visualization**  
- `ExportTextureFeatures.py`: Plots the area around a particular landmark used in the local binary pattern calculation at a specific resolution.

**Average Face Generation**  
- `CreateStandardizedImages.py`: Generates standardized face images by registering to a reference using landmarks and cropping the face region. Optional background removal.  
- `AverageFace.py`: Constructs an average face from the standardized dataset.

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

---

## 📚 Citation

If you use and/or refer to this software in your research, please cite the following papers:

- Cerrolaza JJ, Porras AR, Mansoor A, Zhao Q, Summar M, Linguraru MG. *Identification of dysmorphic syndromes using landmark-specific local texture descriptors*. Proceedings - International Symposium on Biomedical Imaging. 2016;2016-June:1080-1083. https://doi.org/10.1109/ISBI.2016.7493453

- Porras AR, Summar M, Linguraru MG. *Objective differential diagnosis of Noonan and Williams-Beuren syndromes in diverse populations using quantitative facial phenotyping*. Mol Genet Genomic Med. 2021;9(5). https://doi.org/10.1002/MGG3.1636

- Porras AR, Bramble MS, Mosema Be Amoti K, et al. *Facial analysis technology for the detection of Down syndrome in the Democratic Republic of the Congo*. Eur J Med Genet. 2021;64(9):104267. https://doi.org/10.1016/J.EJMG.2021.104267

---

## 🪪 License

This work is licensed under a  
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

![License Badge](license-banner.png)  
![CC BY-NC-SA](cc-by-nc-sa-badge.png)
