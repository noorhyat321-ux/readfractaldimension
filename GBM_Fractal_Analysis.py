# ==============================================================================
# PART 1: SETUP & LIBRARIES
# ==============================================================================
import os
import glob
import time
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import linregress, ttest_ind, pearsonr
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Rectangle

print("Libraries loaded. Ready for analysis.")

# ==============================================================================
# PART 2: FRACTAL DIMENSION ALGORITHM (3D BOX COUNTING)
# ==============================================================================
def get_fractal_dimension(binary_mask):
    """
    Calculates the 3D Minkowski-Bouligand Fractal Dimension.
    """
    if np.sum(binary_mask) == 0:
        return np.nan, np.nan

    # Pad to power of 2 to allow perfect division
    p = binary_mask.shape
    max_dim = max(p)
    s = 2**int(np.ceil(np.log2(max_dim)))
    padded = np.zeros((s, s, s))
    padded[:p[0], :p[1], :p[2]] = binary_mask
    
    box_sizes = []
    counts = []
    box = padded
    k = 1
    
    # The Box-Counting Loop
    while k <= s/2:
        count = np.sum(box > 0)
        box_sizes.append(k)
        counts.append(count)
        # Vectorized Downsampling (Merge 2x2x2 blocks)
        sh = box.shape
        box = box.reshape(sh[0]//2, 2, sh[1]//2, 2, sh[2]//2, 2).sum(axis=(1, 3, 5))
        k *= 2
        
    if len(counts) < 3: 
        return np.nan, np.nan

    # Linear Regression on Log-Log scale
    x = np.log(1 / np.array(box_sizes))
    y = np.log(np.array(counts))
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    return slope, r_value**2

# ==============================================================================
# PART 3: DATA EXTRACTION PIPELINE (The Factory Loop)
# ==============================================================================
# NOTE: This block requires the BraTS 2020 Raw Data path
base_path = '/content/BraTS_Data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'

def run_extraction_pipeline():
    results = []
    all_folders = sorted(glob.glob(os.path.join(base_path, 'BraTS20*')))
    
    print(f"Processing {len(all_folders)} patients...")
    
    for i, folder in enumerate(all_folders):
        patient_id = os.path.basename(folder)
        # Find segmentation file
        seg_files = glob.glob(os.path.join(folder, '*seg*'))
        if not seg_files: continue
        
        try:
            # Load Data
            seg_data = nib.load(seg_files[0]).get_fdata()
            
            # Isolate Necrosis (Label 1)
            necrosis_mask = np.where(seg_data == 1, 1, 0)
            vol = np.sum(necrosis_mask)
            
            if vol < 100: continue # Skip empty/tiny
            
            # Crop to bounding box (Optimization)
            r = np.any(necrosis_mask, axis=(1, 2))
            c = np.any(necrosis_mask, axis=(0, 2))
            z = np.any(necrosis_mask, axis=(0, 1))
            rmin, rmax = np.where(r)[0][[0, -1]]
            cmin, cmax = np.where(c)[0][[0, -1]]
            zmin, zmax = np.where(z)[0][[0, -1]]
            cropped = necrosis_mask[rmin:rmax+1, cmin:cmax+1, zmin:zmax+1]
            
            # Calculate FD
            fd, r2 = get_fractal_dimension(cropped)
            
            results.append({
                'Brats20ID': patient_id,
                'FractalDimension': fd,
                'R_Squared': r2,
                'NecrosisVolume': vol
            })
            
        except Exception as e:
            print(f"Error on {patient_id}: {e}")
            
    return pd.DataFrame(results)

# Uncomment to run raw extraction:
# df_results = run_extraction_pipeline()
# df_results.to_csv('raw_fractal_data.csv', index=False)

# ==============================================================================
# PART 4: STATISTICAL ANALYSIS
# ==============================================================================
# Load the cleaned Master Data (Results + Survival merged)
df = pd.read_csv('FINAL_MASTER_DATA.csv')

# A. Correlation (Morphology)
# ---------------------------
df['LogVolume'] = np.log1p(df['Necrosis_Volume_mm3'])
r_val, p_val = pearsonr(df['LogVolume'], df['FractalDimension'])
print(f"Correlation (FD vs Vol): r={r_val:.4f}, p={p_val:.2e}")

# B. Survival Analysis (Cox Regression)
# ---------------------------
cph = CoxPHFitter()
stats_df = df.dropna(subset=['Survival_Days', 'Age', 'FractalDimension', 'LogVolume'])
stats_df['Event'] = 1 

print("\n--- Multivariate Cox Regression ---")
cph.fit(stats_df[['Survival_Days', 'Event', 'FractalDimension', 'Age', 'LogVolume']], 
        duration_col='Survival_Days', event_col='Event')
cph.print_summary()

# C. Surgery Analysis
# ---------------------------
print("\n--- Surgical Resectability (T-Test) ---")
# Map GTR/STR text to numbers if needed, or use existing groups
if 'Extent_of_Resection' in df.columns:
    gtr = df[df['Extent_of_Resection'] == 'GTR']['FractalDimension']
    str_group = df[df['Extent_of_Resection'] == 'STR']['FractalDimension']
    t_stat, p_surgery = ttest_ind(gtr, str_group)
    print(f"GTR vs STR: p={p_surgery:.4f}")

# ==============================================================================
# PART 5: FIGURE GENERATION (Unified Dark Mode)
# ==============================================================================
# This generates the 4-panel Figure 1 used in the manuscript
# Requires loading one high-FD patient (e.g., BraTS20_Training_020)

def generate_figure_1(t1ce_path, seg_path):
    img = nib.load(t1ce_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()
    necrosis = np.where(seg == 1, 1, 0)
    
    # Find Best Slices
    z_best = np.argmax(np.sum(necrosis, axis=(0,1)))
    y_best = np.argmax(np.sum(necrosis, axis=(0,2)))
    x_best = np.argmax(np.sum(necrosis, axis=(1,2)))
    
    # Helper to make images square
    def pad_square(arr):
        h, w = arr.shape
        max_dim = max(h, w)
        return np.pad(arr, ((0, max_dim-h), (0, max_dim-w)), mode='constant')

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor='black')
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    # Plot Anatomy
    ax1.imshow(pad_square(np.rot90(img[:,:,z_best])), cmap='gray')
    ax1.contour(pad_square(np.rot90(necrosis[:,:,z_best])), colors='red')
    ax1.axis('off'); ax1.set_title("Axial", color='white')
    
    ax2.imshow(pad_square(np.rot90(img[:,y_best,:])), cmap='gray')
    ax2.contour(pad_square(np.rot90(necrosis[:,y_best,:])), colors='red')
    ax2.axis('off'); ax2.set_title("Coronal", color='white')

    ax3.imshow(pad_square(np.rot90(img[x_best,:,:])), cmap='gray')
    ax3.contour(pad_square(np.rot90(necrosis[x_best,:,:])), colors='red')
    ax3.axis('off'); ax3.set_title("Sagittal", color='white')
    
    # Plot Math (Log-Log) - Simplified for demo
    ax4.set_facecolor('black')
    ax4.text(0.5, 0.5, "Log-Log Plot Generated Here", color='white', ha='center')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('Figure1_Reproduced.png', dpi=300, facecolor='black')
    print("Figure 1 generated.")
