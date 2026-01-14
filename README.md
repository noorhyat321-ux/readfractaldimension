<p align="center">
  <img src="./Figure%201" alt="Project header image" width="600"/>
</p>

# Geometric Heterogeneity of the Necrotic Core: A Radiomic Analysis of Tumor Morphology in Glioblastoma

This repository contains the full computational pipeline, statistical analysis, and source code for a retrospective radiomic study analyzing 231 Glioblastoma (GBM) patients from the **BraTS 2020** dataset.

## 📌 Project Overview
**Objective:** To determine if the **Fractal Dimension (FD)** of the necrotic core serves as an independent prognostic biomarker for overall survival and surgical resectability, distinct from tumor volume.

**Methodology:**
1.  **Preprocessing:** Automated extraction of the necrotic core (Label 1) from T1-weighted contrast-enhanced MRI (T1-CE).
2.  **Radiomics:** Calculation of 3D Morphological Complexity using the **Minkowski-Bouligand Box-Counting Algorithm**.
3.  **Statistics:** Univariate and Multivariate Cox Proportional Hazards Regression, Kaplan-Meier Survival Analysis, and Pearson Correlation.

---

## 📊 Key Findings
*   **Morphological Law:** Glioblastoma necrosis follows a fractal architecture with high linearity ($R^2 > 0.99$).
*   **Volume-Complexity Scaling:** There is a massive, positive linear correlation between Fractal Dimension and Log-Necrosis Volume ($r = 0.85, p < 10^{-67}$).
*   **Survival Analysis:**
    *   **Univariate:** Necrosis volume is a significant predictor of mortality ($p = 0.01$), whereas Fractal Dimension is not ($p = 0.28$).
    *   **Multivariate:** When controlled for tumor volume and age, Fractal Dimension provides **no independent prognostic value** ($p = 0.96$).
*   **Conclusion:** Geometric complexity in GBM necrosis is a morphological surrogate for tumor burden rather than an independent driver of aggression.

---

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `GBM_Fractal_Analysis.ipynb` | **The Master Notebook.** Contains the full pipeline: Image Loading, 3D Box-Counting, Statistical Analysis, and Figure Generation. |
| `requirements.txt` | List of Python dependencies required to run the analysis. |
| `Supplementary_Data.csv` | Anonymized processed data containing calculated FD, Volume, and Survival data for the cohort (n=231). |

---

## ⚙️ Installation & Usage

### 1. Prerequisites
This project requires Python 3.7+ and the following libraries:
*   `nibabel` (Neuroimaging data handling)
*   `numpy` & `pandas` (Data manipulation)
*   `scipy` (Linear regression & image filtering)
*   `lifelines` (Survival analysis)
*   `matplotlib` & `scikit-image` (Visualization & Marching Cubes)
  
Install dependencies using:
```bash
pip install -r requirements.txt

2. Data Access (BraTS 2020)
Note: The raw MRI scans (.nii.gz files) are not included in this repository due to licensing restrictions.
Researchers must obtain the dataset from the official BraTS 2020 Challenge.
Once downloaded, update the base_path variable in the notebook to point to your local data directory.
3. Running the Pipeline
Open GBM_Fractal_Analysis.ipynb in Jupyter Notebook or Google Colab. The notebook is structured in 5 parts:
Setup: Library imports.
Math Engine: The raw 3D Box-Counting function.
Data Extraction: Loops through patient folders to extract FD and Volume.
Statistics: Runs Univariate/Multivariate Cox Models and T-Tests.
Visualization: Generates the 4-panel manuscript figures (Anatomy, Correlation, Survival).
📜 Citation & Data Acknowledgments
This study utilized data from the Multimodal Brain Tumor Segmentation Challenge (BraTS). We acknowledge the data contributors and organizers:
Menze BH, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)." IEEE Transactions on Medical Imaging. 2015.
Bakas S, et al. "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features." Nature Scientific Data. 2017.
Bakas S, et al. "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation." arXiv preprint. 2018.
