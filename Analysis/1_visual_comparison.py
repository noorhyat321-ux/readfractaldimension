import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from skimage import measure
from scipy.ndimage import gaussian_filter
import os

# ==========================================
# CONFIGURATION
# ==========================================
# User: Replace these filenames with your actual sample data placed in the 'data' folder
# If you don't have the data, this script will skip execution gracefully.
T1CE_FILENAME = 'BraTS20_Training_020_t1ce.nii.gz'
SEG_FILENAME  = 'BraTS20_Training_020_seg.nii.gz'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
T1CE_PATH = os.path.join(DATA_DIR, T1CE_FILENAME)
SEG_PATH  = os.path.join(DATA_DIR, SEG_FILENAME)

def generate_comparison_figure():
    print(f"--- Running Visual Comparison Script ---")
    
    # 1. Check if Data Exists
    if not os.path.exists(T1CE_PATH) or not os.path.exists(SEG_PATH):
        print(f"⚠️  Data files not found in {DATA_DIR}.")
        print(f"    Please place a sample .nii.gz file there to run the visual check.")
        return

    # 2. Load Data
    print(f"Loading: {T1CE_FILENAME}...")
    img = nib.load(T1CE_PATH).get_fdata()
    seg = nib.load(SEG_PATH).get_fdata()

    # 3. Isolate Necrotic Core (Label 1)
    mask = (seg == 1).astype(np.uint8)

    # 4. Find the 'Best' Slice (Max Necrosis)
    z = np.argmax(np.sum(mask, axis=(0,1)))
    
    # 5. Crop to Tumor Center
    coords = np.argwhere(mask[:,:,z])
    if len(coords) == 0:
        print("No necrosis found in this file.")
        return
        
    center_x, center_y = coords.mean(axis=0).astype(int)
    r = 60 # Crop radius
    
    # Handle edge cases where crop goes out of bounds
    x_min, x_max = max(0, center_x-r), min(img.shape[0], center_x+r)
    y_min, y_max = max(0, center_y-r), min(img.shape[1], center_y+r)

    img_slice = img[x_min:x_max, y_min:y_max, z]
    mask_slice = mask[x_min:x_max, y_min:y_max, z]

    # 6. Normalize Image (2nd-98th percentile)
    p2, p98 = np.percentile(img_slice, (2, 98))
    img_norm = np.clip((img_slice - p2) / (p98 - p2), 0, 1)

    # 7. Generate Contours
    # A: Raw (No smoothing)
    raw_contours = measure.find_contours(mask_slice, 0.5)
    
    # B: Gaussian Smoothed (For visual comparison)
    mask_smooth = gaussian_filter(mask_slice.astype(float), sigma=1.2)
    smooth_contours = measure.find_contours(mask_smooth, 0.5)

    # 8. Plotting
    plt.rcParams['font.family'] = 'serif'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5))

    # Panel A
    ax1.imshow(img_norm, cmap='gray')
    for c in raw_contours:
        ax1.plot(c[:,1], c[:,0], color='lime', linewidth=1) # Lime shows up well on grey
    ax1.set_title("A. Raw Binary Mask (Analysis Input)", fontsize=10)
    ax1.axis('off')

    # Panel B
    ax2.imshow(img_norm, cmap='gray')
    for c in smooth_contours:
        ax2.plot(c[:,1], c[:,0], color='red', linewidth=1.5)
    ax2.set_title("B. Gaussian-Smoothed (Display Only)", fontsize=10)
    ax2.axis('off')

    plt.tight_layout()
    
    output_path = os.path.join(BASE_DIR, "Supplementary_Figure_S1.png")
    plt.savefig(output_path, dpi=300)
    print(f"✅ Figure saved to: {output_path}")

if __name__ == "__main__":
    generate_comparison_figure()
