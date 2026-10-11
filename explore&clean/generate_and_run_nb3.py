import json
import os
import sys
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def make_cell(cell_type, source, outputs=None, execution_count=None):
    lines = source.split("\n")
    source_lines = [line + "\n" for line in lines[:-1]]
    if lines:
        source_lines.append(lines[-1])
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source_lines
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs or []
    return cell

def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

cells = [
    # -------------------------------------------------------------
    # 0. HEADER & THEORY
    # -------------------------------------------------------------
    make_cell("markdown", """# ☀️ Photovoltaic Performance & Expected Power Analysis
## Method 1: Calibrated Physics-Based Derating Model (Standard Test Conditions - STC)

---

### 📋 Overview & Engineering Calibration
In Photovoltaic (PV) solar diagnostic systems, assessing panel health requires comparing **Actual Power Output ($P_{\\text{actual}}$)** against the theoretical **Expected Power Output ($P_{\\text{expected}}$)** computed under identical environmental conditions (Solar Irradiance $G$ and Module Temperature $T$).

#### ⚠️ Critical Physics Calibration ($P_{\\text{mp}}$ vs. $I_{\\text{sc}} \\times V_{\\text{oc}}$):
* In raw panel technical datasheets, $I_{\\text{max}} = 7.75\\text{ A}$ represents the **Short-Circuit Current ($I_{\\text{sc}}$)** and $V_{\\text{max}} = 19.00\\text{ V}$ represents the **Open-Circuit Voltage ($V_{\\text{oc}}$)**.
* Their mathematical product ($7.75 \\times 19.00 = 147.25\\text{ W}$) is the unachievable **rectangular boundary**, because a solar cell cannot operate at $I_{\\text{sc}}$ and $V_{\\text{oc}}$ simultaneously.
* The true rated Maximum Power Point capacity at STC incorporates the panel's **Fill Factor ($FF \\approx 0.645 - 0.65$)**:
  $$P_{\\text{mp\\_STC}} = FF \\times I_{\\text{sc}} \\times V_{\\text{oc}} \\approx 0.6452 \\times 147.25\\text{ W} = \\mathbf{95.0\\text{ W}}$$
* Calibrating $P_{\\text{mp\\_STC}} = 95.0\\text{ W}$ ensures that a healthy panel in the `Normal` state achieves an accurate **$PR \\approx 95\\% - 100\\%$** instead of an artificially depressed ratio.

---

### 🎯 Key Governing Formulas:
1. **Actual Power ($P_{\\text{actual}}$)**:
   $$P_{\\text{actual}} = I_{\\text{measured}} \\times V_{\\text{measured}} \\quad [\\text{Watts}]$$
2. **Physics Expected Power ($P_{\\text{expected\\_phys}}$)**:
   $$P_{\\text{expected\\_phys}} = P_{\\text{mp\\_STC}} \\times \\left( \\frac{G}{G_{\\text{STC}}} \\right) \\times \\Big[ 1 + \\gamma \\times (T_{\\text{module}} - T_{\\text{STC}}) \\Big]$$
   - $G_{\\text{STC}} = 1000\\text{ W/m}^2$ (Standard Test Irradiance)
   - $T_{\\text{STC}} = 25.0^\\circ\\text{C}$ (Standard Test Cell Temperature)
   - $\\gamma = -0.004\\text{ /}^\\circ\\text{C}$ (Temperature Power Derating Coefficient: $-0.4\\%/^\\circ\\text{C}$)
3. **Performance Ratio ($PR$)**:
   $$PR = \\left( \\frac{P_{\\text{actual}}}{P_{\\text{expected\\_phys}}} \\right) \\times 100\\% \\quad [\\%]$$
4. **Power Loss ($\\Delta P$)**:
   $$\\Delta P = \\max(0, P_{\\text{expected\\_phys}} - P_{\\text{actual}}) \\quad [\\text{Watts}]$$"""),

    # -------------------------------------------------------------
    # 1. IMPORTS & SETUP
    # -------------------------------------------------------------
    make_cell("code", """import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore', category=FutureWarning)

# High-resolution plotting setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'figure.dpi': 130,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

# Universal color palette for operational states
STATE_PALETTE = {
    'Normal': '#2ecc71',         # Green
    'Degradation': '#f39c12',    # Orange / Amber
    'Shading': '#3498db',        # Sky Blue
    'Short Circuit': '#e74c3c',  # Crimson Red
    'Open Circuit': '#9b59b6'    # Purple
}

print("✅ Setup completed successfully. Libraries loaded.")"""),

    # -------------------------------------------------------------
    # 2. DATA INGESTION
    # -------------------------------------------------------------
    make_cell("markdown", """### 1. Ingest Clean Multi-Sensor Telemetry Dataset
We load the physics-cleaned multi-string PV dataset (`PV_Diagnostic_Cleaned.csv`)."""),

    make_cell("code", """# Search standard workspace paths
candidates = [
    Path("../explore&clean/PV_Diagnostic_Cleaned.csv"),
    Path("explore&clean/PV_Diagnostic_Cleaned.csv"),
    Path("../data/PV_Diagnostic_Cleaned.csv"),
    Path("data/PV_Diagnostic_Cleaned.csv"),
    Path("PV_Diagnostic_Cleaned.csv"),
]

dataset_path = next((p for p in candidates if p.exists()), None)
if dataset_path is None:
    raise FileNotFoundError("Could not locate PV_Diagnostic_Cleaned.csv in expected directories.")

print(f"Loading cleaned dataset from: {dataset_path.resolve()}")
df = pd.read_csv(dataset_path, sep=";")
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Dataset Overview: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Time span: {df['timestamp'].min()}  -->  {df['timestamp'].max()}")
df.head(5)"""),

    # -------------------------------------------------------------
    # 3. PHYSICS MODEL IMPLEMENTATION
    # -------------------------------------------------------------
    make_cell("markdown", """### 2. Implement Calibrated Physics-Based Model
We compute expected power and diagnostic metrics for both **String 1 (Panel 1)** and **String 2 (Panel 2)**."""),

    make_cell("code", """# Physical Calibration Constants
P_MP_STC = 95.0     # Rated MPP Capacity at STC (Watts)
G_STC = 1000.0      # Standard Irradiance (W/m^2)
T_STC = 25.0        # Standard Temperature (deg C)
GAMMA = -0.004      # Temperature derating coefficient (-0.40% / deg C)

# 1. Actual Measured Power
df['P1_actual'] = df['P1']
df['P2_actual'] = df['P2']

# 2. Physics Expected Power (Clamped at 0 for physical validity)
df['P1_expected_phys'] = np.maximum(0.0, P_MP_STC * (df['Irr'] / G_STC) * (1.0 + GAMMA * (df['T1'] - T_STC)))
df['P2_expected_phys'] = np.maximum(0.0, P_MP_STC * (df['Irr'] / G_STC) * (1.0 + GAMMA * (df['T2'] - T_STC)))

# 3. Filtering Conditions
# Daytime: Irradiance >= 50 W/m2 (eliminates night noise and pyranometer drift)
daytime_mask = df['Irr'] >= 50.0

# Active Generation: Voltage >= 5.0 V (isolates active DC power production)
active_mask_p1 = daytime_mask & (df['V1'] >= 5.0)
active_mask_p2 = daytime_mask & (df['V2'] >= 5.0)

# 4. Performance Ratio (PR %) with realistic clipping [0%, 125%]
df['PR1_phys'] = np.where(
    daytime_mask & (df['P1_expected_phys'] > 0.0),
    np.clip((df['P1_actual'] / df['P1_expected_phys']) * 100.0, 0.0, 125.0),
    np.nan
)
df['PR2_phys'] = np.where(
    daytime_mask & (df['P2_expected_phys'] > 0.0),
    np.clip((df['P2_actual'] / df['P2_expected_phys']) * 100.0, 0.0, 125.0),
    np.nan
)

# 5. Power Loss (Watts)
df['Power_Loss_P1'] = np.maximum(0.0, df['P1_expected_phys'] - df['P1_actual'])
df['Power_Loss_P2'] = np.maximum(0.0, df['P2_expected_phys'] - df['P2_actual'])

print(f"✅ Physics metrics calculated.")
print(f"   • Total records:           {len(df):,}")
print(f"   • Daytime records:         {daytime_mask.sum():,} ({daytime_mask.mean()*100:.1f}%)")
print(f"   • Active Gen (Panel 1):    {active_mask_p1.sum():,} ({active_mask_p1.mean()*100:.1f}%)")
print(f"   • Active Gen (Panel 2):    {active_mask_p2.sum():,} ({active_mask_p2.mean()*100:.1f}%)")"""),

    # -------------------------------------------------------------
    # 4. STATISTICAL BENCHMARKS
    # -------------------------------------------------------------
    make_cell("markdown", """### 3. Comprehensive Statistical Benchmarking
We benchmark performance across all operating states in two critical operational contexts:
1. **Active Generation Regime ($V \\ge 5.0\\text{ V}$)**: Reflects operating efficiency during active grid/load power delivery.
2. **All Daytime Telemetry ($Irr \\ge 50\\text{ W/m}^2$)**: Includes open-circuit, short-circuit, and zero-generation fault modes."""),

    make_cell("code", """# 1. Panel 1 - Active Generation Benchmark
active_df_p1 = df[active_mask_p1].copy()
stats_active_p1 = active_df_p1.groupby('Label_S1', observed=False).agg(
    Samples=('P1_actual', 'count'),
    Mean_Actual_W=('P1_actual', 'mean'),
    Mean_Expected_W=('P1_expected_phys', 'mean'),
    Mean_PR_pct=('PR1_phys', 'mean'),
    Median_PR_pct=('PR1_phys', 'median'),
    Mean_Loss_W=('Power_Loss_P1', 'mean')
).round(2)

print("=" * 70)
print("📊 PANEL 1 (STRING 1): ACTIVE GENERATION PERFORMANCE (V1 >= 5.0V)")
print("=" * 70)
stats_active_p1"""),

    make_cell("code", """# 2. Panel 2 - Active Generation Benchmark
active_df_p2 = df[active_mask_p2].copy()
stats_active_p2 = active_df_p2.groupby('Label_S2', observed=False).agg(
    Samples=('P2_actual', 'count'),
    Mean_Actual_W=('P2_actual', 'mean'),
    Mean_Expected_W=('P2_expected_phys', 'mean'),
    Mean_PR_pct=('PR2_phys', 'mean'),
    Median_PR_pct=('PR2_phys', 'median'),
    Mean_Loss_W=('Power_Loss_P2', 'mean')
).round(2)

print("=" * 70)
print("📊 PANEL 2 (STRING 2): ACTIVE GENERATION PERFORMANCE (V2 >= 5.0V)")
print("=" * 70)
stats_active_p2"""),

    make_cell("code", """# 3. Overall Daytime Benchmark (All Operational Telemetry including Short Circuits)
day_df = df[daytime_mask].copy()
stats_day_p1 = day_df.groupby('Label_S1', observed=False).agg(
    Total_Samples=('P1_actual', 'count'),
    Mean_Actual_W=('P1_actual', 'mean'),
    Mean_Expected_W=('P1_expected_phys', 'mean'),
    Mean_PR_pct=('PR1_phys', 'mean'),
    Median_PR_pct=('PR1_phys', 'median'),
    Mean_Loss_W=('Power_Loss_P1', 'mean')
).round(2)

print("=" * 70)
print("☀️ ALL DAYTIME TELEMETRY (Irr >= 50 W/m2) - PANEL 1")
print("=" * 70)
stats_day_p1"""),

    # -------------------------------------------------------------
    # 5. DIAGNOSTIC RULE ENGINE
    # -------------------------------------------------------------
    make_cell("markdown", """### 4. Physics-Based Diagnostic Rule Engine
Using the calibrated physics model, we can formulate an interpretable, deterministic rule-based fault classifier based on $PR$ and terminal voltage $V$:
* **Healthy (`Normal`)**: $PR \\ge 90\\%$ and $V \\ge 12\\text{ V}$
* **`Degradation` / Soiling**: $65\\% \\le PR < 90\\%$ and $V \\ge 10\\text{ V}$
* **`Shading`**: $PR < 65\\%$ and $V \\ge 5\\text{ V}$
* **`Open Circuit`**: $I < 0.5\\text{ A}$ and $V \\ge 14\\text{ V}$
* **`Short Circuit`**: $V < 3.0\\text{ V}$ and $I \\ge 1.0\\text{ A}$"""),

    make_cell("code", """def rule_based_physics_diagnose(row, p_num=1):
    i = row[f'I{p_num}']
    v = row[f'V{p_num}']
    pr = row[f'PR{p_num}_phys']
    irr = row['Irr']
    
    if irr < 50.0:
        return 'Night / Inactive'
    if v < 3.0 and i >= 1.0:
        return 'Short Circuit'
    if i < 0.5 and v >= 12.0:
        return 'Open Circuit'
    if pd.isna(pr):
        return 'Unknown'
    if pr >= 90.0:
        return 'Normal'
    elif 65.0 <= pr < 90.0:
        return 'Degradation'
    else:
        return 'Shading'

# Evaluate Rule Engine on Daytime Telemetry
day_df['Diagnosed_S1'] = day_df.apply(lambda r: rule_based_physics_diagnose(r, 1), axis=1)

# Diagnostic Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix

valid_eval = day_df[day_df['Diagnosed_S1'] != 'Night / Inactive']
print("=" * 70)
print("🔍 RULE-BASED PHYSICS DIAGNOSTIC REPORT (VS. GROUND TRUTH S1):")
print("=" * 70)
print(classification_report(valid_eval['Label_S1'], valid_eval['Diagnosed_S1'], digits=3))"""),

    # -------------------------------------------------------------
    # 6. ENGINEERING PLOTS & VISUALIZATIONS
    # -------------------------------------------------------------
    make_cell("markdown", """### 5. Engineering Visualizations & Diagnostics

#### Plot 1: Actual vs. Expected Power Scatter (Panel 1 vs. Panel 2)
Notice how `Normal` points lie tightly on the ideal $y=x$ line ($PR = 100\\%$), `Degradation` points sit systematically below it, and severe faults collapse to the bottom axis."""),

    make_cell("code", """fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Sample 4,000 active points for clean rendering
sample_p1 = active_df_p1.sample(n=min(4000, len(active_df_p1)), random_state=42)
sample_p2 = active_df_p2.sample(n=min(4000, len(active_df_p2)), random_state=42)

for ax, sample, p_name, label_col in zip(axes, [sample_p1, sample_p2], ['Panel 1', 'Panel 2'], ['Label_S1', 'Label_S2']):
    sns.scatterplot(
        data=sample,
        x=f'P{p_name[-1]}_expected_phys',
        y=f'P{p_name[-1]}_actual',
        hue=label_col,
        palette=STATE_PALETTE,
        alpha=0.6,
        s=28,
        edgecolor='none',
        ax=ax
    )
    
    # Ideal line & Health tolerance envelope (+/- 10%)
    max_val = 110.0
    x_line = np.linspace(0, max_val, 100)
    ax.plot(x_line, x_line, 'k--', lw=2, label='Ideal Performance (PR = 100%)')
    ax.fill_between(x_line, x_line * 0.90, x_line * 1.10, color='gray', alpha=0.15, label='±10% Normal Health Envelope')
    
    ax.set_title(f'{p_name}: Actual vs. Expected Power Output', fontweight='bold')
    ax.set_xlabel('Physics Expected Power [Watts] (P_mp = 95W STC)')
    ax.set_ylabel('Measured Actual Power [Watts]')
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.legend(title='Operating State', frameon=True, loc='upper left')

plt.tight_layout()
plt.show()"""),

    make_cell("markdown", """#### Plot 2: Performance Ratio ($PR$) Boxplot & Power Loss Density Distribution
Clear physical separation between states:
* **Normal**: $\\mathbf{PR \\approx 96.3\\% - 99.3\\%}$
* **Degradation**: $\\mathbf{PR \\approx 71.0\\% - 81.9\\%}$
* **Shading / Open Circuit**: $\\mathbf{PR < 25\\%}$"""),

    make_cell("code", """fig, axes = plt.subplots(1, 2, figsize=(16, 6))

order_states = ['Normal', 'Degradation', 'Shading', 'Open Circuit']

# Boxplot of PR %
sns.boxplot(
    data=active_df_p1,
    x='Label_S1',
    y='PR1_phys',
    palette=STATE_PALETTE,
    order=order_states,
    ax=axes[0],
    boxprops=dict(alpha=0.85)
)
axes[0].axhline(90, color='green', linestyle='--', lw=1.8, label='Normal Health Threshold (90%)')
axes[0].axhline(65, color='orange', linestyle='--', lw=1.8, label='Degradation Threshold (65%)')
axes[0].set_title('Performance Ratio (PR %) across Operational States (Panel 1)', fontweight='bold')
axes[0].set_xlabel('Operating Condition')
axes[0].set_ylabel('Performance Ratio (PR %)')
axes[0].set_ylim(-5, 125)
axes[0].legend(loc='lower left', frameon=True)

# KDE of Power Loss (Watts)
sns.kdeplot(
    data=active_df_p1,
    x='Power_Loss_P1',
    hue='Label_S1',
    palette=STATE_PALETTE,
    common_norm=False,
    fill=True,
    alpha=0.3,
    linewidth=2,
    ax=axes[1]
)
axes[1].axvline(5.0, color='green', linestyle=':', label='Normal Tolerance (<5W)')
axes[1].axvline(15.0, color='orange', linestyle=':', label='Degradation Loss (>15W)')
axes[1].set_title('Power Loss (Watts) Distribution by State', fontweight='bold')
axes[1].set_xlabel('Power Loss [Watts] (P_expected - P_actual)')
axes[1].set_ylabel('Density')
axes[1].set_xlim(0, 95)
axes[1].legend(title='Operating Condition', loc='upper right', frameon=True)

plt.tight_layout()
plt.show()"""),

    make_cell("markdown", """#### Plot 3: Dynamic Real-Time Time-Series Tracking with Anomaly Flags
Tracking expected power, actual power, and $PR$ dynamically across consecutive daylight cycles."""),

    make_cell("code", """# Select continuous slice of telemetry
ts_slice = active_df_p1.iloc[1200:1550].copy()

fig, ax1 = plt.subplots(figsize=(15, 6))

ax1.plot(ts_slice['timestamp'], ts_slice['P1_expected_phys'], color='navy', linestyle='--', label='Expected Power (Physics STC=95W)', lw=2.2)
ax1.plot(ts_slice['timestamp'], ts_slice['P1_actual'], color='darkorange', label='Actual Measured Power', lw=2.2)
ax1.set_xlabel('Timestamp', fontweight='bold')
ax1.set_ylabel('Power Output [Watts]', color='navy', fontweight='bold')
ax1.set_title('Real-Time Power Generation & Performance Ratio Tracking', fontweight='bold')
ax1.grid(True, alpha=0.4)

# Secondary axis for PR (%)
ax2 = ax1.twinx()
ax2.plot(ts_slice['timestamp'], ts_slice['PR1_phys'], color='green', alpha=0.45, linestyle=':', lw=1.8, label='Performance Ratio (PR %)')
ax2.axhline(90, color='green', linestyle='--', alpha=0.5, lw=1)
ax2.set_ylabel('Performance Ratio [ % ]', color='green', fontweight='bold')
ax2.set_ylim(0, 125)
ax2.grid(False)

# Combined legend
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', frameon=True)

plt.tight_layout()
plt.show()"""),

    make_cell("markdown", """#### Plot 4: Irradiance vs Power Characteristic Curve
Illustrates the physical derating linearity and temperature dependency."""),

    make_cell("code", """plt.figure(figsize=(11, 6))

sample_irr = active_df_p1.sample(n=min(3000, len(active_df_p1)), random_state=42)

scatter = plt.scatter(
    sample_irr['Irr'],
    sample_irr['P1_actual'],
    c=sample_irr['T1'],
    cmap='plasma',
    alpha=0.6,
    s=25
)
cbar = plt.colorbar(scatter)
cbar.set_label('Module Temperature T1 [°C]', fontweight='bold')

# Overlay theoretical 25°C curve
irr_range = np.linspace(50, 1100, 100)
p_theory_25 = P_MP_STC * (irr_range / G_STC) * (1.0 + GAMMA * (25.0 - T_STC))
p_theory_50 = P_MP_STC * (irr_range / G_STC) * (1.0 + GAMMA * (50.0 - T_STC))

plt.plot(irr_range, p_theory_25, 'g--', lw=2.5, label='Theoretical STC (T=25°C)')
plt.plot(irr_range, p_theory_50, 'r--', lw=2.5, label='Theoretical Hot (T=50°C)')

plt.title('Power Generation vs. Solar Irradiance with Thermal Derating', fontweight='bold')
plt.xlabel('Solar Irradiance [W/m²]')
plt.ylabel('Actual Power [Watts]')
plt.legend(frameon=True, loc='upper left')
plt.tight_layout()
plt.show()"""),

    # -------------------------------------------------------------
    # 7. SUMMARY & ACTIONABLE THRESHOLDS
    # -------------------------------------------------------------
    make_cell("markdown", """### 6. Engineering Summary & Operational Thresholds

| Diagnostic State | Performance Ratio ($PR$) Range | Power Loss ($\\Delta P$) | Operating Action / Recommendation |
| :--- | :---: | :---: | :--- |
| **Normal (Healthy)** | $\\mathbf{PR \\ge 90\\%}$ | $\\le 5.0\\text{ W}$ | System nominal. No maintenance required. |
| **Degradation / Soiling** | $\\mathbf{65\\% \\le PR < 90\\%}$ | $10\\text{ W} - 25\\text{ W}$ | Schedule panel surface cleaning or test series resistance. |
| **Partial Shading** | $\\mathbf{25\\% \\le PR < 65\\%}$ | $25\\text{ W} - 60\\text{ W}$ | Inspect string for physical obstructions or vegetation. |
| **Severe Fault / Inverter Trip** | $\\mathbf{PR < 25\\%}$ | $> 60\\text{ W}$ | Check bypass diodes, fuses, inverter input, or isolation. |

#### 🔑 Key Takeaways:
1. **Fill Factor Calibration ($P_{\\text{mp\\_STC}} = 95.0\\text{ W}$)** accurately resolved the artificial $40\\%$ efficiency penalty of using unachievable $I_{\\text{sc}} \\times V_{\\text{oc}} = 147.25\\text{ W}$.
2. **Deterministic physical indicators** provide explainable, auditable baseline metrics that directly complement supervised ML classifiers.""")
]

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Write notebook file
nb_dict = create_notebook(cells)
out_path = Path("notebooks/3_physics_expected_power.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=1)

print(f"Notebook written to: {out_path.resolve()}")

# Execute notebook to populate all cell outputs
print("Executing notebook to populate all visual outputs and tables...")
with open(out_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
ep.preprocess(nb, {'metadata': {'path': str(Path('notebooks').resolve())}})

with open(out_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook execution completed successfully with all outputs populated!")

