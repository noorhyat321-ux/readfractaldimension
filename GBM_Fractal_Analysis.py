# ==============================================================================
# PART 1: SETUP & LIBRARIES
# ==============================================================================
import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import linregress, ttest_ind, pearsonr
from scipy.ndimage import gaussian_filter
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib.patches import Rectangle

print("Libraries loaded. Ready for analysis.")

# ==============================================================================
# PART 2: THE MATH ENGINE (Raw Binary Data - No Smoothing)
# ==============================================================================
def get_fractal_dimension(binary_mask):
    """
    Calculates 3D Minkowski-Bouligand Dimension on RAW binary masks.

    NOTE:
    - x_data and y_data outputs are for optional plotting/log-log visualization only.
    - Manuscript FD calculation used only slope (FD) and R^2.
    """
    if np.sum(binary_mask) == 0: return np.nan, np.nan, None, None

    # Dynamic Padding to next power of 2
    p = binary_mask.shape
    max_dim = max(p)
    s = 2**int(np.ceil(np.log2(max_dim)))
    padded = np.zeros((s, s, s))
    padded[:p[0], :p[1], :p[2]] = binary_mask
    
    box_sizes = []
    counts = []
    box = padded
    k = 1
    
    # Box Counting Loop
    while k <= s/2:
        count = np.sum(box > 0)
        box_sizes.append(k)
        counts.append(count)
        # Vectorized downsampling
        sh = box.shape
        box = box.reshape(sh[0]//2, 2, sh[1]//2, 2, sh[2]//2, 2).sum(axis=(1, 3, 5))
        k *= 2
        
    if len(counts) < 3: return np.nan, np.nan, None, None

    # Log-Log Regression
    x = np.log(1 / np.array(box_sizes))
    y = np.log(np.array(counts))
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    return slope, r_value**2, x, y  # slope = FD, r_value^2 = linearity

# ==============================================================================
# PART 3: DATA EXTRACTION (Skipped in GitHub version; manuscript used preprocessed CSV)
# ==============================================================================
# Data is already in FINAL_MASTER_DATA.csv for analysis.

# ==============================================================================
# PART 4: STATISTICAL ANALYSIS
# ==============================================================================
df = pd.read_csv('FINAL_MASTER_DATA.csv')
df['LogVolume'] = np.log1p(df['Necrosis_Volume_mm3'])

print(">>> 1. MORPHOLOGICAL CORRELATION <<<")
r_val, p_val = pearsonr(df['LogVolume'], df['FractalDimension'])
print(f"Correlation (FD vs Volume): r={r_val:.4f}, p={p_val:.2e}")

print("\n>>> 2. SURGICAL RESECTABILITY <<<")
if 'Extent_of_Resection' in df.columns:
    # Map GTR/STR to numeric codes for optional analysis
    mapping = {'GTR': 0, 'STR': 1}
    surgery_df = df[df['Extent_of_Resection'].isin(['GTR', 'STR'])].copy()
    surgery_df['Surgery_Code'] = surgery_df['Extent_of_Resection'].map(mapping)
    
    gtr = surgery_df[surgery_df['Extent_of_Resection'] == 'GTR']['FractalDimension']
    str_group = surgery_df[surgery_df['Extent_of_Resection'] == 'STR']['FractalDimension']
    
    t_stat, p_surgery = ttest_ind(gtr, str_group)
    print(f"T-Test (GTR vs STR): p={p_surgery:.4f}")
else:
    print("Surgery column not found or not formatted.")

# Prepare Survival Data
stats_df = df.dropna(subset=['Survival_days', 'Age', 'FractalDimension', 'LogVolume'])
stats_df['Duration'] = pd.to_numeric(stats_df['Survival_days'])
stats_df['Event'] = 1 
cph = CoxPHFitter()

print("\n>>> 3. UNIVARIATE COX REGRESSION <<<")
# Manuscript Figure/Table: Volume Alone
cph.fit(stats_df[['Duration', 'Event', 'LogVolume']], duration_col='Duration', event_col='Event')
print(f"Volume Alone: p={cph.summary.loc['LogVolume', 'p']:.5f} (HR={cph.summary.loc['LogVolume', 'exp(coef)']:.2f})")

# Manuscript Figure/Table: FD Alone
cph.fit(stats_df[['Duration', 'Event', 'FractalDimension']], duration_col='Duration', event_col='Event')
print(f"Fractal Alone: p={cph.summary.loc['FractalDimension', 'p']:.5f} (HR={cph.summary.loc['FractalDimension', 'exp(coef)']:.2f})")

print("\n>>> 4. MULTIVARIATE COX REGRESSION <<<")
cols = ['Duration', 'Event', 'FractalDimension', 'Age', 'LogVolume']
if 'Surgery_Code' in surgery_df.columns:
    stats_df_surg = stats_df.merge(surgery_df[['Brats20ID', 'Surgery_Code']], on='Brats20ID', how='inner')
    print(f"Running Multivariate on Surgical Sub-cohort (n={len(stats_df_surg)})...")
    cph.fit(stats_df_surg[cols + ['Surgery_Code']], duration_col='Duration', event_col='Event')
else:
    print("Running Multivariate on Full Cohort (No Surgery)...")
    cph.fit(stats_df[cols], duration_col='Duration', event_col='Event')

print(cph.summary[['exp(coef)', 'p', 'z']])

# ==============================================================================
# PART 5: FIGURE GENERATION
# ==============================================================================
# NOTE: Figures here include enhancements (median lines, optional log-log scatter)
# Manuscript used FD & KM only; Volume plot added for visual comparison (Figure4_Composite).

def plot_figure_1(t1ce_path, seg_path):
    """
    Generates Figure 1: Imaging slices (Axial, Coronal, Sagittal) + FD log-log plot
    NOTE: Gaussian smoothing is for visualization only. FD is computed on raw mask.
    """
    img = nib.load(t1ce_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()
    necrosis = np.where(seg == 1, 1, 0)
    
    coords = np.argwhere(necrosis)
    x_c, y_c, z_c = coords.mean(axis=0).astype(int)
    r = 50
    crop_math = necrosis[
        max(0, x_c-r):min(necrosis.shape[0], x_c+r),
        max(0, y_c-r):min(necrosis.shape[1], y_c+r),
        max(0, z_c-r):min(necrosis.shape[2], z_c+r)
    ]
    fd, r2, x_data, y_data = get_fractal_dimension(crop_math)
    
    z = np.argmax(np.sum(necrosis, axis=(0,1)))
    y = np.argmax(np.sum(necrosis, axis=(0,2)))
    x = np.argmax(np.sum(necrosis, axis=(1,2)))
    
    sigma = 1.0  # smoothing for visualization only
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor='black')
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    views = [
        (ax1, np.rot90(img[:,:,z]), np.rot90(gaussian_filter(necrosis[:,:,z].astype(float), sigma)), "A: Axial"),
        (ax2, np.rot90(img[:,y,:]), np.rot90(gaussian_filter(necrosis[:,y,:].astype(float), sigma)), "B: Coronal"),
        (ax3, np.rot90(img[x,:,:]), np.rot90(gaussian_filter(necrosis[x,:,:].astype(float), sigma)), "C: Sagittal")
    ]
    
    for ax, i_slice, m_slice, title in views:
        ax.imshow(i_slice, cmap='gray', interpolation='bicubic')
        ax.contour(m_slice, levels=[0.5], colors='#ff3333', linewidths=2)
        ax.set_title(title, color='white')
        ax.axis('off')
        
    # Panel D: Real Math
    ax4.set_facecolor('black')
    ax4.scatter(x_data, y_data, color='white', edgecolor='blue', s=80, label='Box Counts')
    ax4.plot(x_data, fd*x_data + (y_data[0]-fd*x_data[0]), color='#ff3333', linewidth=2, label=f'Fit (D={fd:.2f})')
    ax4.set_xlabel("Log(1/s)", color='white')
    ax4.set_ylabel("Log(N)", color='white')
    ax4.tick_params(colors='white')
    ax4.legend()
    ax4.set_title(f"D: Log-Log Plot (R2={r2:.3f})", color='white')
    
    plt.tight_layout()
    plt.savefig('Figure1_Reproduced.png', dpi=300, facecolor='black')
    print("Figure 1 Saved.")

def plot_figure_4(data):
    """
    Generates Figure 4: Composite KM curves
    Left = Fractal Dimension, Right = Necrosis Volume
    NOTE: Median lines and Volume plot are optional visualization; manuscript used FD only.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # FD panel
    median_fd = data['FractalDimension'].median()
    high_fd = data[data['FractalDimension'] > median_fd]
    low_fd = data[data['FractalDimension'] <= median_fd]
    p_fd = logrank_test(high_fd['Duration'], low_fd['Duration'],
                        event_observed_A=high_fd['Event'], event_observed_B=low_fd['Event']).p_value
    
    kmf1 = KaplanMeierFitter()
    kmf1.fit(low_fd['Duration'], low_fd['Event'], label='Low FD')
    kmf1.plot_survival_function(ax=ax1, color='blue', ci_show=True, lw=3)
    kmf1.fit(high_fd['Duration'], high_fd['Event'], label='High FD')
    kmf1.plot_survival_function(ax=ax1, color='red', ci_show=True, lw=3)
    ax1.set_title(f"A. Stratified by Fractal Dimension\n(p={p_fd:.3f})", fontsize=14, fontweight='bold')
    ax1.set_ylim(0,1); ax1.grid(alpha=0.3)

    # Volume panel
    median_vol = data['Necrosis_Volume_mm3'].median()
    high_vol = data[data['Necrosis_Volume_mm3'] > median_vol]
    low_vol = data[data['Necrosis_Volume_mm3'] <= median_vol]
    p_vol = logrank_test(high_vol['Duration'], low_vol['Duration'],
                         event_observed_A=high_vol['Event'], event_observed_B=low_vol['Event']).p_value
    
    kmf2 = KaplanMeierFitter()
    kmf2.fit(low_vol['Duration'], low_vol['Event'], label='Small Volume')
    kmf2.plot_survival_function(ax=ax2, color='green', ci_show=True, lw=3)
    kmf2.fit(high_vol['Duration'], high_vol['Event'], label='Large Volume')
    kmf2.plot_survival_function(ax=ax2, color='orange', ci_show=True, lw=3)
    ax2.set_title(f"B. Stratified by Volume\n(p={p_vol:.3f})", fontsize=14, fontweight='bold')
    ax2.set_ylim(0,1); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('Figure4_Composite_FD_Volume.png', dpi=300)
    print("Figure 4 Saved.")

# To run the visualization
# plot_figure_4(stats_df)
