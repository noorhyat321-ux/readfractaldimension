import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import os

# ==========================================
# CONFIGURATION
# ==========================================
EXCEL_FILENAME = 'suplementary_data_GBM.xlsx'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', EXCEL_FILENAME)

def calculate_vif():
    print(f"--- Running Multicollinearity (VIF) Check ---")

    if not os.path.exists(DATA_PATH):
        print(f"⚠️  Data file not found at: {DATA_PATH}")
        return

    df = pd.read_excel(DATA_PATH)

    # 1. Prepare Dataframe
    # We select the variables used in the Multivariate Cox Model
    # Note: Log-transformation is crucial because raw volume is skewed
    
    vif_df = pd.DataFrame()
    vif_df['Age'] = df['Age']
    vif_df['FractalDimension'] = df['FractalDimension']
    vif_df['LogVolume'] = np.log1p(df['NecrosisVolume']) # log(1+x) to handle zeros
    
    # Drop rows with any missing values to ensure fair calculation
    vif_df = vif_df.dropna()
    
    # 2. Add Constant (Intercept)
    # Statsmodels requires an explicit constant column to calculate VIF correctly
    X = add_constant(vif_df)

    # 3. Calculate VIF
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF Score"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

    # 4. Display Results
    print("\nVariance Inflation Factor (VIF) Results:")
    print("-" * 40)
    print(vif_data)
    print("-" * 40)
    print("Interpretation:")
    print("VIF ~ 1: No correlation")
    print("VIF 1-5: Moderate correlation (Stable)")
    print("VIF > 5: High correlation (Unstable)")
    
    # Check for collinearity issue
    fd_vif = vif_data.loc[vif_data['Feature'] == 'FractalDimension', 'VIF Score'].values[0]
    vol_vif = vif_data.loc[vif_data['Feature'] == 'LogVolume', 'VIF Score'].values[0]
    
    if abs(fd_vif - vol_vif) < 1.0 and fd_vif > 3.0:
        print("\n✅ RESULT: FD and Volume have similar, elevated VIF scores.")
        print("   This confirms they explain the same variance (Collinearity).")

if __name__ == "__main__":
    calculate_vif()
