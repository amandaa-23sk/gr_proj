import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Determine project root and paths reliably
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name in ["explore&clean", "pdf_pre", "notebooks"] else SCRIPT_DIR

# Create clean EDA outputs directory
eda_clean_dir = PROJECT_ROOT / "eda_clean_outputs"
eda_clean_dir.mkdir(parents=True, exist_ok=True)

# Locate cleaned dataset
clean_candidates = [
    SCRIPT_DIR / "PV_Diagnostic_Cleaned.csv",
    PROJECT_ROOT / "explore&clean" / "PV_Diagnostic_Cleaned.csv",
    PROJECT_ROOT / "PV_Diagnostic_Cleaned.csv"
]
clean_path = next((p for p in clean_candidates if p.exists()), None)
if clean_path is None:
    raise FileNotFoundError(f"Could not locate PV_Diagnostic_Cleaned.csv in candidates: {clean_candidates}")

# 1. Load Clean Dataset
print(f"Loading clean dataset from: {clean_path}")
df = pd.read_csv(clean_path, sep=";")
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("=== DATASET SHAPE ===")
print(f"Total Rows: {len(df):,}, Columns: {df.shape[1]}")

# Panel 1 & Panel 2 subsets
df_p1 = df[['timestamp', 'I1', 'V1', 'P1', 'T1', 'Irr', 'Label_Num_S1', 'Label_S1']].copy()
df_p1.columns = ['timestamp', 'Current', 'Voltage', 'Power', 'Temperature', 'Irradiance', 'Label_Num', 'Fault_Label']
df_p1['Panel'] = 'Panel 1'

df_p2 = df[['timestamp', 'I2', 'V2', 'P2', 'T2', 'Irr', 'Label_Num_S2', 'Label_S2']].copy()
df_p2.columns = ['timestamp', 'Current', 'Voltage', 'Power', 'Temperature', 'Irradiance', 'Label_Num', 'Fault_Label']
df_p2['Panel'] = 'Panel 2'

# Print Clean Summary Statistics
print("\n=== PANEL 1 CLEAN SUMMARY ===")
print(df_p1[['Current', 'Voltage', 'Power', 'Temperature', 'Irradiance']].describe().T)

print("\n=== PANEL 2 CLEAN SUMMARY ===")
print(df_p2[['Current', 'Voltage', 'Power', 'Temperature', 'Irradiance']].describe().T)

print("\n=== PANEL 1 MEAN METRICS BY FAULT ===")
print(df_p1.groupby('Fault_Label')[['Current', 'Voltage', 'Power', 'Temperature', 'Irradiance']].mean())

print("\n=== PANEL 2 MEAN METRICS BY FAULT ===")
print(df_p2.groupby('Fault_Label')[['Current', 'Voltage', 'Power', 'Temperature', 'Irradiance']].mean())

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fault_labels = ['Normal', 'Degradation', 'Open Circuit', 'Shading', 'Short Circuit']
palette = {'Normal': '#2ca02c', 'Degradation': '#ff7f0e', 'Open Circuit': '#d62728', 'Shading': '#9467bd', 'Short Circuit': '#8c564b'}

# 1. P-V Curves Comparison (100% physically clean)
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
for label in fault_labels:
    sub1 = df_p1[df_p1['Fault_Label'] == label].sample(min(2500, len(df_p1[df_p1['Fault_Label'] == label])), random_state=42)
    axes[0].scatter(sub1['Voltage'], sub1['Power'], label=label, color=palette[label], alpha=0.45, s=14)

axes[0].set_title("Panel 1: Clean P-V Characteristics", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Voltage (V)", fontsize=11)
axes[0].set_ylabel("Power (W)", fontsize=11)
axes[0].legend(loc='upper left', frameon=True)
axes[0].grid(True, alpha=0.3)

for label in fault_labels:
    sub2 = df_p2[df_p2['Fault_Label'] == label].sample(min(2500, len(df_p2[df_p2['Fault_Label'] == label])), random_state=42)
    axes[1].scatter(sub2['Voltage'], sub2['Power'], label=label, color=palette[label], alpha=0.45, s=14)

axes[1].set_title("Panel 2: Clean P-V Characteristics", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Voltage (V)", fontsize=11)
axes[1].legend(loc='upper left', frameon=True)
axes[1].grid(True, alpha=0.3)

plt.suptitle("Clean P-V Characteristics by Fault Class (Panel 1 vs Panel 2)", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(eda_clean_dir / "01_clean_pv_curves.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Power vs Irradiance (P vs Irr)
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
for label in fault_labels:
    sub1 = df_p1[df_p1['Fault_Label'] == label].sample(min(2500, len(df_p1[df_p1['Fault_Label'] == label])), random_state=42)
    axes[0].scatter(sub1['Irradiance'], sub1['Power'], label=label, color=palette[label], alpha=0.45, s=14)

axes[0].set_title("Panel 1: Power vs. Solar Irradiance", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Irradiance (W/m²)", fontsize=11)
axes[0].set_ylabel("Power (W)", fontsize=11)
axes[0].legend(loc='upper left', frameon=True)
axes[0].grid(True, alpha=0.3)

for label in fault_labels:
    sub2 = df_p2[df_p2['Fault_Label'] == label].sample(min(2500, len(df_p2[df_p2['Fault_Label'] == label])), random_state=42)
    axes[1].scatter(sub2['Irradiance'], sub2['Power'], label=label, color=palette[label], alpha=0.45, s=14)

axes[1].set_title("Panel 2: Power vs. Solar Irradiance", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Irradiance (W/m²)", fontsize=11)
axes[1].legend(loc='upper left', frameon=True)
axes[1].grid(True, alpha=0.3)

plt.suptitle("Clean Power Output vs Solar Irradiance (Panel 1 vs Panel 2)", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(eda_clean_dir / "02_power_vs_irradiance.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Clean Boxplots for Power and Efficiency
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
data_p = [df_p1['Power'], df_p2['Power']]
axes[0].boxplot(data_p, tick_labels=['Panel 1 Power', 'Panel 2 Power'], patch_artist=True,
                boxprops=dict(facecolor='#a8dadc', color='#1d3557'),
                medianprops=dict(color='#e63946', linewidth=2))
axes[0].set_title("Clean Power Distribution (W)", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Power (W)")

# Power distribution by fault type in Panel 1
box_data = [df_p1[df_p1['Fault_Label'] == l]['Power'] for l in fault_labels]
axes[1].boxplot(box_data, tick_labels=fault_labels, patch_artist=True,
                boxprops=dict(facecolor='#fed9b7', color='#f07167'),
                medianprops=dict(color='#0081a7', linewidth=2))
axes[1].set_title("Panel 1 Clean Power by Fault Class", fontsize=12, fontweight='bold')
axes[1].tick_params(axis='x', rotation=20)
axes[1].set_ylabel("Power (W)")

plt.suptitle("Clean Power Distribution & Fault Profiling", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(eda_clean_dir / "03_clean_power_distributions.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Clean Correlation Heatmaps
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
corr1 = df_p1[['Current', 'Voltage', 'Power', 'Temperature', 'Irradiance']].corr()
corr2 = df_p2[['Current', 'Voltage', 'Power', 'Temperature', 'Irradiance']].corr()

im1 = axes[0].matshow(corr1, cmap='coolwarm', vmin=-1, vmax=1)
axes[0].set_title("Panel 1 Clean Correlation Matrix", fontsize=12, fontweight='bold', pad=20)
axes[0].set_xticks(range(len(corr1.columns)))
axes[0].set_yticks(range(len(corr1.columns)))
axes[0].set_xticklabels(corr1.columns, rotation=45)
axes[0].set_yticklabels(corr1.columns)
for i in range(len(corr1.columns)):
    for j in range(len(corr1.columns)):
        axes[0].text(j, i, f"{corr1.iloc[i, j]:.2f}", ha="center", va="center", color="white" if abs(corr1.iloc[i, j]) > 0.5 else "black")

im2 = axes[1].matshow(corr2, cmap='coolwarm', vmin=-1, vmax=1)
axes[1].set_title("Panel 2 Clean Correlation Matrix", fontsize=12, fontweight='bold', pad=20)
axes[1].set_xticks(range(len(corr2.columns)))
axes[1].set_yticks(range(len(corr2.columns)))
axes[1].set_xticklabels(corr2.columns, rotation=45)
axes[1].set_yticklabels(corr2.columns)
for i in range(len(corr2.columns)):
    for j in range(len(corr2.columns)):
        axes[1].text(j, i, f"{corr2.iloc[i, j]:.2f}", ha="center", va="center", color="white" if abs(corr2.iloc[i, j]) > 0.5 else "black")

fig.colorbar(im2, ax=axes, orientation='horizontal', fraction=0.05, pad=0.15)
plt.savefig(eda_clean_dir / "04_clean_correlations.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Clean Differential Diagnostics (ΔP = P1 - P2)
df['Delta_P_clean'] = df['P1'] - df['P2']
df['Delta_I'] = df['I1'] - df['I2']
df['Delta_V'] = df['V1'] - df['V2']

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].hist(df['Delta_P_clean'], bins=100, color='#3a86ff', edgecolor='black', alpha=0.75)
axes[0].set_title(r"$\Delta P = P_1 - P_2$ (Clean Differential)", fontweight='bold')
axes[0].set_xlabel("Power Difference (W)")

axes[1].hist(df['Delta_I'], bins=100, color='#8338ec', edgecolor='black', alpha=0.75)
axes[1].set_title(r"$\Delta I = I_1 - I_2$", fontweight='bold')
axes[1].set_xlabel("Current Difference (A)")

axes[2].hist(df['Delta_V'], bins=100, color='#ff006e', edgecolor='black', alpha=0.75)
axes[2].set_title(r"$\Delta V = V_1 - V_2$", fontweight='bold')
axes[2].set_xlabel("Voltage Difference (V)")

plt.suptitle("Clean Inter-Panel Differential Features", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(eda_clean_dir / "05_clean_differentials.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"All Clean EDA plots successfully generated in: {eda_clean_dir}")
