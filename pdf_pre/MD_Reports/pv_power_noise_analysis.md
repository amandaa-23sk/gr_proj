# 🔬 Photovoltaic Dataset Noise & Data Quality Analysis Report
**Dataset**: `PV_Diagnostic_Final_Labels.csv` (226,175 records)  
**Target Columns**: `P1` (Panel 1 Power), `P2` (Panel 2 Power), `Irr` (Irradiance), `timestamp`

---

## 1. Executive Summary

This report provides a granular audit of data corruption, noise, and calculation anomalies present in the PV diagnostic dataset. 

A critical finding is that **11.19% of records in Panel 1** and **8.71% in Panel 2** contain corrupted power values resulting from:
1. **Polarity / Sign Inversion ($P = -I \times V$)**: Caused by sensor polarity or datalogger sign inversion.
2. **Extreme Physical Outliers**: Values violating theoretical maximum solar generation ($P_{\text{max}} = 147.25\text{ W}$).
3. **Calculation Mismatches**: Unexplained arbitrary values diverging from $I \times V$.
4. **Night-time / Low-Irradiance artifacts**: Distortions during zero-sunlight hours.

---

## 2. Panel 1 Power ($P_1$) Noise Breakdown

Theoretical physical bounds based on maximum panel ratings ($I_{1,\text{max}} = 7.75\text{ A}$, $V_{1,\text{max}} = 19.00\text{ V}$):
$$\mathbf{P_{1,\text{physical\_max}} = 7.75\text{ A} \times 19.00\text{ V} = 147.25\text{ W}}$$

### $P_1$ Quality Statistics (Total Rows: 226,175)

| Category / Anomaly Type | Row Count | Percentage | Minimum Value | Maximum Value | Physical Impact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **✅ Perfectly Correct ($P_1 = I_1 \times V_1$)** | **200,874** | **88.81%** | 0.00 W | 146.95 W | Healthy physical data |
| **❌ Inverted Sign ($-I_1 \times V_1$)** | **24,533** | **10.85%** | -473.67 W | -0.0001 W | Negative power generation (Sensor bug) |
| **❌ Calculation Mismatch / Glitch** | **768** | **0.34%** | -0.91 W | 334.72 W | Uncorrelated corrupted readings |
| **⚠️ Extreme Outliers ($|P_1| > 147.25\text{ W}$)** | **711** | **0.31%** | -473.67 W | 334.72 W | Impossible physical power |
| **🔴 Total Corrupted Rows in $P_1$** | **25,301** | **11.19%** | **-473.67 W** | **334.72 W** | **Destroys LSTM & ML loss convergence** |

---

### $P_1$ Noise Distribution Across Fault Classes

| Fault Class (`Label_S1`) | Total Rows | Correct ($P_1 = I_1 \times V_1$) | Sign Inverted ($-I_1 \times V_1$) | Mismatch / Glitch | Noise Percentage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Degradation** | 103,885 | 102,105 (98.29%) | 1,780 (1.71%) | 0 (0.00%) | **1.71%** |
| **Normal** | 47,303 | 43,794 (92.58%) | 2,770 (5.86%) | 739 (1.56%) | **7.42%** |
| **Open Circuit** | 58,171 | 42,245 (72.62%) | 15,926 (27.38%) | 0 (0.00%) | **27.38%** |
| **Shading** | 16,763 | 12,716 (75.86%) | 4,047 (24.14%) | 0 (0.00%) | **24.14%** |
| **Short Circuit** | 53 | 14 (26.42%) | 10 (18.87%) | 29 (54.72%) | **73.58%** |
| **Total** | **226,175** | **200,874 (88.81%)** | **24,533 (10.85%)** | **768 (0.34%)** | **11.19%** |

> [!CAUTION]
> In the **`Open Circuit`** and **`Shading`** categories, over **24% to 27%** of all data points have inverted negative power signs. In **`Short Circuit`**, **73.58%** of samples are corrupted!

---

## 3. Panel 2 Power ($P_2$) Noise Breakdown

Theoretical physical bounds ($I_{2,\text{max}} = 7.75\text{ A}$, $V_{2,\text{max}} = 18.97\text{ V}$):
$$\mathbf{P_{2,\text{physical\_max}} = 7.75\text{ A} \times 18.97\text{ V} = 147.02\text{ W}}$$

### $P_2$ Quality Statistics (Total Rows: 226,175)

| Category / Anomaly Type | Row Count | Percentage | Minimum Value | Maximum Value | Physical Impact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **✅ Perfectly Correct ($P_2 = I_2 \times V_2$)** | **206,470** | **91.29%** | 0.00 W | 146.95 W | Healthy physical data |
| **❌ Inverted Sign ($-I_2 \times V_2$)** | **19,619** | **8.67%** | -6.28 W | -0.0001 W | Negative power generation |
| **❌ Calculation Mismatch / Glitch** | **86** | **0.04%** | -0.0084 W | 232.15 W | Corrupted sensor reading |
| **⚠️ Extreme Outliers ($|P_2| > 147.02\text{ W}$)** | **26** | **0.01%** | 150.12 W | 232.15 W | Values exceed physical max |
| **🔴 Total Corrupted Rows in $P_2$** | **19,705** | **8.71%** | **-6.28 W** | **232.15 W** | Corrupted signal |

---

### $P_2$ Noise Distribution Across Fault Classes

| Fault Class (`Label_S2`) | Total Rows | Correct ($P_2 = I_2 \times V_2$) | Sign Inverted ($-I_2 \times V_2$) | Mismatch / Glitch | Noise Percentage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Degradation** | 105,672 | 104,118 (98.53%) | 1,554 (1.47%) | 0 (0.00%) | **1.47%** |
| **Normal** | 51,864 | 49,361 (95.17%) | 2,441 (4.71%) | 62 (0.12%) | **4.83%** |
| **Open Circuit** | 52,104 | 37,673 (72.30%) | 14,431 (27.70%) | 0 (0.00%) | **27.70%** |
| **Shading** | 16,402 | 15,216 (92.77%) | 1,186 (7.23%) | 0 (0.00%) | **7.23%** |
| **Short Circuit** | 133 | 102 (76.69%) | 7 (5.26%) | 24 (18.05%) | **23.31%** |
| **Total** | **226,175** | **206,470 (91.29%)** | **19,619 (8.67%)** | **86 (0.04%)** | **8.71%** |

---

## 4. Environmental & Temporal Signal Noise

### A. Low Irradiance & Night-Time Artifacts ($Irr < 50\text{ W/m}^2$)
* **Total Rows**: **4,985 rows (2.20%)**
* **Issue**: During night/early morning hours, $I \approx 0$ and $P \approx 0$ irrespective of panel health.
* **Breakdown by Label**:
  * Shading: 4,093 rows (82.1%)
  * Normal: 783 rows (15.7%)
  * Degradation: 90 rows (1.8%)
  * Short Circuit: 17 rows (0.3%)
  * Open Circuit: 2 rows (0.1%)

### B. Time Discontinuities (Gaps)
* **Short Gaps (> 5 seconds)**: 56 occurrences
* **Long Night Gaps (> 1 hour)**: 11 occurrences (representing day-night transitions)
* **Risk**: Sliding windows (e.g. sequences of 30 or 60 steps) that cross these gaps combine non-contiguous physical time periods.

---

## 5. Concrete Solution & Cleansing Script

To eliminate **100% of the power calculation noise**, execute the following preprocessing pipeline before training any Machine Learning or LSTM model:

```python
import pandas as pd
import numpy as np

def clean_pv_dataset(csv_path="PV_Diagnostic_Final_Labels.csv"):
    df = pd.read_csv(csv_path, sep=";")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. Fix Power Noise programmatically (P = I * V)
    df['P1'] = df['I1'] * df['V1']
    df['P2'] = df['I2'] * df['V2']
    
    # 2. Filter out dark / night periods
    df_clean = df[df['Irr'] >= 50].sort_values('timestamp').reset_index(drop=True)
    
    # 3. Validation Check
    assert (df_clean['P1'] < 0).sum() == 0, "Error: Negative P1 detected!"
    assert (df_clean['P2'] < 0).sum() == 0, "Error: Negative P2 detected!"
    assert (df_clean['P1'] > 150).sum() == 0, "Error: Outlier P1 detected!"
    
    print(f"Data Cleansed Successfully! Remaining valid rows: {len(df_clean):,}")
    return df_clean

if __name__ == "__main__":
    df_clean = clean_pv_dataset()
```
