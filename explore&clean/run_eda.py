import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Determine project root and paths reliably
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name in ["explore&clean", "pdf_pre", "notebooks"] else SCRIPT_DIR

# Create figures directory
eda_outputs_dir = PROJECT_ROOT / "eda_outputs"
eda_outputs_dir.mkdir(parents=True, exist_ok=True)

# Locate raw dataset
raw_candidates = [
    SCRIPT_DIR / "PV_Diagnostic_Final_Labels.csv",
    PROJECT_ROOT / "explore&clean" / "PV_Diagnostic_Final_Labels.csv",
    PROJECT_ROOT / "PV_Diagnostic_Final_Labels.csv"
]
raw_path = next((p for p in raw_candidates if p.exists()), None)
if raw_path is None:
    raise FileNotFoundError(f"Could not locate PV_Diagnostic_Final_Labels.csv in candidates: {raw_candidates}")

# 1. Load Data
print(f"Loading raw dataset from: {raw_path}")
df = pd.read_csv(raw_path, sep=";")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 2. Separate Panel 1 & Panel 2
df_p1 = df[['timestamp', 'I1', 'V1', 'P1', 'T1', 'Irr', 'Label_Num_S1', 'Label_S1']].rename(
    columns={'I1': 'Current', 'V1': 'Voltage', 'P1': 'Power', 'T1': 'Temperature', 'Label_Num_S1': 'Label_Num', 'Label_S1': 'Fault_Label'}
)
df_p1['Panel'] = 'Panel 1'

df_p2 = df[['timestamp', 'I2', 'V2', 'P2', 'T2', 'Irr', 'Label_Num_S2', 'Label_S2']].rename(
    columns={'I2': 'Current', 'V2': 'Voltage', 'P2': 'Power', 'T2': 'Temperature', 'Label_Num_S2': 'Label_Num', 'Label_S2': 'Fault_Label'}
)
df_p2['Panel'] = 'Panel 2'

# Save separated CSVs
p1_csv = eda_outputs_dir / "panel_1_data.csv"
p2_csv = eda_outputs_dir / "panel_2_data.csv"
df_p1.to_csv(p1_csv, index=False)
df_p2.to_csv(p2_csv, index=False)
print(f"Saved separate CSVs:\n  - {p1_csv}\n  - {p2_csv}")

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Chart 1: Class Distribution Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
p1_counts = df_p1['Fault_Label'].value_counts()
p2_counts = df_p2['Fault_Label'].value_counts()

axes[0].bar(p1_counts.index, p1_counts.values, color='#2b5c8f', edgecolor='black', alpha=0.85)
axes[0].set_title("Panel 1 Fault Distribution", fontsize=13, fontweight='bold')
axes[0].set_ylabel("Count")
axes[0].tick_params(axis='x', rotation=30)
for i, v in enumerate(p1_counts.values):
    axes[0].text(i, v + 1500, f"{v:,}", ha='center', fontsize=9)

axes[1].bar(p2_counts.index, p2_counts.values, color='#e06666', edgecolor='black', alpha=0.85)
axes[1].set_title("Panel 2 Fault Distribution", fontsize=13, fontweight='bold')
axes[1].tick_params(axis='x', rotation=30)
for i, v in enumerate(p2_counts.values):
    axes[1].text(i, v + 1500, f"{v:,}", ha='center', fontsize=9)

plt.suptitle("Fault Class Distribution: Panel 1 vs Panel 2", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(eda_outputs_dir / "01_fault_distribution_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# Chart 2: I-V Curves by Fault Class for Panel 1 and Panel 2
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
fault_labels = sorted(df_p1['Fault_Label'].unique())

for label in fault_labels:
    sub_p1 = df_p1[df_p1['Fault_Label'] == label].sample(min(2000, len(df_p1[df_p1['Fault_Label'] == label])), random_state=42)
    axes[0].scatter(sub_p1['Voltage'], sub_p1['Current'], label=label, alpha=0.4, s=12)

axes[0].set_title("Panel 1: I-V Characteristic by Fault Class", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Voltage (V)")
axes[0].set_ylabel("Current (A)")
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

for label in fault_labels:
    sub_p2 = df_p2[df_p2['Fault_Label'] == label].sample(min(2000, len(df_p2[df_p2['Fault_Label'] == label])), random_state=42)
    axes[1].scatter(sub_p2['Voltage'], sub_p2['Current'], label=label, alpha=0.4, s=12)

axes[1].set_title("Panel 2: I-V Characteristic by Fault Class", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Voltage (V)")
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.suptitle("I-V Characteristic Comparison (Panel 1 vs Panel 2)", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(eda_outputs_dir / "02_iv_curves_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# Chart 3: P-V Curves by Fault Class for Panel 1 and Panel 2
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
for label in fault_labels:
    sub_p1 = df_p1[df_p1['Fault_Label'] == label].sample(min(2000, len(df_p1[df_p1['Fault_Label'] == label])), random_state=42)
    axes[0].scatter(sub_p1['Voltage'], sub_p1['Power'], label=label, alpha=0.4, s=12)

axes[0].set_title("Panel 1: P-V Curve by Fault Class", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Voltage (V)")
axes[0].set_ylabel("Power (W)")
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

for label in fault_labels:
    sub_p2 = df_p2[df_p2['Fault_Label'] == label].sample(min(2000, len(df_p2[df_p2['Fault_Label'] == label])), random_state=42)
    axes[1].scatter(sub_p2['Voltage'], sub_p2['Power'], label=label, alpha=0.4, s=12)

axes[1].set_title("Panel 2: P-V Curve by Fault Class", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Voltage (V)")
axes[1].legend(loc='upper left')
axes[1].grid(True, alpha=0.3)

plt.suptitle("P-V Characteristic Comparison (Panel 1 vs Panel 2)", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(eda_outputs_dir / "03_pv_curves_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# Chart 4: Feature Distributions (Boxplots)
features = ['Current', 'Voltage', 'Power', 'Temperature']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, feat in enumerate(features):
    data = [df_p1[feat].dropna(), df_p2[feat].dropna()]
    axes[idx].boxplot(data, tick_labels=['Panel 1', 'Panel 2'], patch_artist=True,
                      boxprops=dict(facecolor='#a8dadc', color='#1d3557'),
                      medianprops=dict(color='#e63946', linewidth=2))
    axes[idx].set_title(f"Distribution of {feat}", fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle("Key Parameter Distributions (Panel 1 vs Panel 2)", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(eda_outputs_dir / "04_boxplots_features.png", dpi=300, bbox_inches='tight')
plt.close()

# Chart 5: Correlation Heatmaps
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
corr1 = df_p1[['Current', 'Voltage', 'Power', 'Temperature', 'Irr']].corr()
corr2 = df_p2[['Current', 'Voltage', 'Power', 'Temperature', 'Irr']].corr()

cax1 = axes[0].matshow(corr1, cmap='coolwarm', vmin=-1, vmax=1)
axes[0].set_title("Panel 1 Correlation Matrix", fontsize=13, fontweight='bold', pad=20)
axes[0].set_xticks(range(len(corr1.columns)))
axes[0].set_yticks(range(len(corr1.columns)))
axes[0].set_xticklabels(corr1.columns, rotation=45)
axes[0].set_yticklabels(corr1.columns)
for i in range(len(corr1.columns)):
    for j in range(len(corr1.columns)):
        axes[0].text(j, i, f"{corr1.iloc[i, j]:.2f}", ha="center", va="center", color="black" if abs(corr1.iloc[i, j]) < 0.6 else "white")

cax2 = axes[1].matshow(corr2, cmap='coolwarm', vmin=-1, vmax=1)
axes[1].set_title("Panel 2 Correlation Matrix", fontsize=13, fontweight='bold', pad=20)
axes[1].set_xticks(range(len(corr2.columns)))
axes[1].set_yticks(range(len(corr2.columns)))
axes[1].set_xticklabels(corr2.columns, rotation=45)
axes[1].set_yticklabels(corr2.columns)
for i in range(len(corr2.columns)):
    for j in range(len(corr2.columns)):
        axes[1].text(j, i, f"{corr2.iloc[i, j]:.2f}", ha="center", va="center", color="black" if abs(corr2.iloc[i, j]) < 0.6 else "white")

fig.colorbar(cax2, ax=axes, orientation='horizontal', fraction=0.05, pad=0.15)
plt.savefig(eda_outputs_dir / "05_correlation_matrices.png", dpi=300, bbox_inches='tight')
plt.close()

# Chart 6: Paired Differential Analysis (Panel 1 - Panel 2)
df['Delta_I'] = df['I1'] - df['I2']
df['Delta_V'] = df['V1'] - df['V2']
df['Delta_P'] = df['P1'] - df['P2']
df['Delta_T'] = df['T1'] - df['T2']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].hist(df['Delta_I'], bins=100, color='#3a86ff', edgecolor='black', alpha=0.7)
axes[0, 0].set_title(r"$\Delta I$ (I1 - I2) Distribution", fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel("Current Difference (A)")

axes[0, 1].hist(df['Delta_V'], bins=100, color='#8338ec', edgecolor='black', alpha=0.7)
axes[0, 1].set_title(r"$\Delta V$ (V1 - V2) Distribution", fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel("Voltage Difference (V)")

axes[1, 0].hist(df['Delta_P'], bins=100, color='#ff006e', edgecolor='black', alpha=0.7)
axes[1, 0].set_title(r"$\Delta P$ (P1 - P2) Distribution", fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel("Power Difference (W)")

axes[1, 1].hist(df['Delta_T'], bins=100, color='#fb5607', edgecolor='black', alpha=0.7)
axes[1, 1].set_title(r"$\Delta T$ (T1 - T2) Distribution", fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel("Temperature Difference (°C)")

plt.suptitle("Inter-Panel Differential Diagnostics (Panel 1 vs Panel 2)", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(eda_outputs_dir / "06_differential_distributions.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"All visual EDA plots generated successfully in: {eda_outputs_dir}")
