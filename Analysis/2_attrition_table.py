
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import os

# ==========================================
# CONFIGURATION
# ==========================================
EXCEL_FILENAME = 'suplementary_data_GBM.xlsx'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', EXCEL_FILENAME)

def generate_attrition_table():
    print(f"--- Running Attrition Analysis Script ---")

    if not os.path.exists(DATA_PATH):
        print(f"⚠️  Data file not found at: {DATA_PATH}")
        print("    Run 'data/generate_mock_data.py' first if you don't have the real data.")
        return

    df = pd.read_excel(DATA_PATH)
    print(f"Loaded {len(df)} patients.")
    
    # 1. Define Cohorts
    # Included: Patients with known GTR or STR status
    included = df[df['Extent_of_Resection'].isin(['GTR', 'STR'])]
    # Excluded: Patients with missing or NA status
    excluded = df[~df['Extent_of_Resection'].isin(['GTR', 'STR'])]

    # 2. Helper Functions
    def get_stats(group, col):
        return f"{group[col].mean():.1f} ± {group[col].std():.1f}"

    def format_p(p):
        return "<0.001" if p < 0.001 else f"{p:.3f}"

    # 3. Calculate Statistics (T-Tests)
    # Note: nan_policy='omit' ensures we don't crash on empty cells
    p_age = ttest_ind(included['Age'], excluded['Age'], nan_policy='omit').pvalue
    p_vol = ttest_ind(included['NecrosisVolume'], excluded['NecrosisVolume'], nan_policy='omit').pvalue
    
    # Optional: Check FD as well if available, though Selection Bias usually cares about Age/Vol
    p_fd  = ttest_ind(included['FractalDimension'], excluded['FractalDimension'], nan_policy='omit').pvalue

    # 4. Construct Data Structure
    data = [
        ["Age, years", 
         get_stats(included, 'Age'), 
         get_stats(excluded, 'Age'), 
         format_p(p_age)],
         
        ["Necrosis volume, mm³", 
         get_stats(included, 'NecrosisVolume'), 
         get_stats(excluded, 'NecrosisVolume'), 
         format_p(p_vol)],
         
        ["Fractal dimension", 
         get_stats(included, 'FractalDimension'), 
         get_stats(excluded, 'FractalDimension'), 
         format_p(p_fd)]
    ]

    columns = ["Variable", f"Included (n={len(included)})", f"Excluded (n={len(excluded)})", "P value"]
    df_table = pd.DataFrame(data, columns=columns)

    # 5. Render Table as Image
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    
    table = ax.table(
        cellText=df_table.values, 
        colLabels=df_table.columns, 
        loc='center', 
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Bold the headers
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
    
    plt.title("Supplementary Table 1: Attrition Analysis (Included vs. Excluded)", fontsize=11, weight='bold', pad=10)
    
    output_path = os.path.join(BASE_DIR, "Supplementary_Table1.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Table saved to: {output_path}")

if __name__ == "__main__":
    generate_attrition_table()
