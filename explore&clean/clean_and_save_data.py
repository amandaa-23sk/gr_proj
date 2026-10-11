import os
from pathlib import Path
import pandas as pd
import numpy as np

# Determine project root and paths reliably
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name in ["explore&clean", "pdf_pre", "notebooks"] else SCRIPT_DIR

# Locate raw dataset
raw_candidates = [
    SCRIPT_DIR / "PV_Diagnostic_Final_Labels.csv",
    PROJECT_ROOT / "explore&clean" / "PV_Diagnostic_Final_Labels.csv",
    PROJECT_ROOT / "PV_Diagnostic_Final_Labels.csv"
]

raw_path = next((p for p in raw_candidates if p.exists()), None)
if raw_path is None:
    raise FileNotFoundError(f"Could not locate PV_Diagnostic_Final_Labels.csv in candidates: {raw_candidates}")

print(f"=== 1. Loading raw dataset from: {raw_path} ===")
df = pd.read_csv(raw_path, sep=";")
print(f"Loaded {len(df):,} rows and {df.shape[1]} columns.")

# Record initial error stats
neg_p1_initial = (df['P1'] < 0).sum()
neg_p2_initial = (df['P2'] < 0).sum()
outliers_p1_initial = (df['P1'] > 147.25).sum()  # due to max voltage * max current
outliers_p2_initial = (df['P2'] > 147.25).sum()

print(f"Initial Negative P1: {neg_p1_initial:,} rows")
print(f"Initial Negative P2: {neg_p2_initial:,} rows")
print(f"Initial Outliers P1 (>147.25W): {outliers_p1_initial:,} rows")
print(f"Initial Outliers P2 (>147.25W): {outliers_p2_initial:,} rows")

print("\n=== 2. Calculating correct physical power (P = I * V) ===")
df['P1'] = df['I1'] * df['V1']
df['P2'] = df['I2'] * df['V2']

# Verification checks
neg_p1_after = (df['P1'] < 0).sum()
neg_p2_after = (df['P2'] < 0).sum()
outliers_p1_after = (df['P1'] > 147.25).sum()
outliers_p2_after = (df['P2'] > 147.25).sum()

print("\n=== 3. Verification Post-Correction ===")
print(f"Negative P1 count: {neg_p1_after} (Min: {df['P1'].min():.6f} W, Max: {df['P1'].max():.6f} W)")
print(f"Negative P2 count: {neg_p2_after} (Min: {df['P2'].min():.6f} W, Max: {df['P2'].max():.6f} W)")
print(f"Outliers P1 (>147.25W): {outliers_p1_after}")
print(f"Outliers P2 (>147.25W): {outliers_p2_after}")

# Ensure exact column order
col_order = ['timestamp', 'I1', 'I2', 'V1', 'V2', 'Irr', 'T1', 'T2', 'P1', 'P2', 'Label_Num_S1', 'Label_Num_S2', 'Label_S1', 'Label_S2']
df = df[col_order]

# Save full clean dataset to explore&clean directory
clean_csv_path = SCRIPT_DIR / "PV_Diagnostic_Cleaned.csv"
df.to_csv(clean_csv_path, sep=";", index=False)
print(f"\nSaved full clean dataset to: {clean_csv_path}")

# Ensure eda_outputs directory exists
eda_outputs_dir = PROJECT_ROOT / "eda_outputs"
eda_outputs_dir.mkdir(parents=True, exist_ok=True)

# Also update the separate Panel 1 & Panel 2 CSVs in eda_outputs
p1_clean = df[['timestamp', 'I1', 'V1', 'P1', 'T1', 'Irr', 'Label_Num_S1', 'Label_S1']].rename(
    columns={'I1': 'Current', 'V1': 'Voltage', 'P1': 'Power', 'T1': 'Temperature', 'Label_Num_S1': 'Label_Num', 'Label_S1': 'Fault_Label'}
)
p1_path = eda_outputs_dir / "panel_1_data_clean.csv"
p1_clean.to_csv(p1_path, index=False)

p2_clean = df[['timestamp', 'I2', 'V2', 'P2', 'T2', 'Irr', 'Label_Num_S2', 'Label_S2']].rename(
    columns={'I2': 'Current', 'V2': 'Voltage', 'P2': 'Power', 'T2': 'Temperature', 'Label_Num_S2': 'Label_Num', 'Label_S2': 'Fault_Label'}
)
p2_path = eda_outputs_dir / "panel_2_data_clean.csv"
p2_clean.to_csv(p2_path, index=False)
print(f"Saved separate cleaned panel datasets:\n  - {p1_path}\n  - {p2_path}")
