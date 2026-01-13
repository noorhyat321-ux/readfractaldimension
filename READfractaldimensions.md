![Image](image.png)



# Geometric Heterogeneity of Glioblastoma Necrosis (BraTS 2020)

This repository contains the source code for the retrospective radiomic analysis of 231 Glioblastoma patients from the BraTS 2020 dataset.

## Study Summary
*   **Objective:** To determine if the Fractal Dimension (FD) of the necrotic core predicts overall survival independent of tumor volume.
*   **Methods:** 3D Minkowski-Bouligand Box-Counting algorithm.
*   **Key Findings:** FD is strongly correlated with tumor volume ($p < 10^{-67}$) but does not independently predict survival ($p = 0.96$) or surgical resectability ($p = 0.33$).

## Repository Structure
*   `GBM_Fractal_Analysis.ipynb`: The main pipeline (Preprocessing -> Feature Extraction -> Statistics -> Visualization).
*   `requirements.txt`: Python dependencies.

## Data Availability
Data was obtained from the [BraTS 2020 Challenge](https://www.med.upenn.edu/sbia/brats2018/data.html). Due to licensing, the raw MRI scans are not included here. Users must obtain the dataset from the official source.
