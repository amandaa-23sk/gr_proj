# ☀️ Clean PV Dataset Exploratory Data Analysis (EDA)
**Clean Dataset File**: [PV_Diagnostic_Cleaned.csv](file:///c:/Users/kirok/Desktop/New%20folder%20%287%29/PV_Diagnostic_Cleaned.csv) (226,175 records)  
**Separated Subsets**: [panel_1_data_clean.csv](file:///c:/Users/kirok/Desktop/New%20folder%20%287%29/eda_outputs/panel_1_data_clean.csv) & [panel_2_data_clean.csv](file:///c:/Users/kirok/Desktop/New%20folder%20%287%29/eda_outputs/panel_2_data_clean.csv)

---

## 1. Clean Summary Statistics (Panel 1 vs Panel 2)

Following physical power reconstruction ($P = I \times V$), all negative values and impossible outliers are completely resolved.

### Panel 1 Clean Parameter Statistics
| Parameter | Mean | Std Dev | Min | 25% (Q1) | Median | 75% (Q3) | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Current $I_1$ (A)** | 1.669 A | 1.731 A | 0.000 A | 0.116 A | 1.124 A | 3.052 A | **7.750 A** |
| **Voltage $V_1$ (V)** | 15.963 V | 3.729 V | 0.000 V | 14.714 V | 16.068 V | 18.787 V | **19.000 V** |
| **Power $P_1$ (W)** | **26.262 W** | **27.639 W** | **0.000 W** | **1.559 W** | **15.790 W** | **49.770 W** | **146.947 W** |
| **Temperature $T_1$ (°C)** | 39.861 °C | 5.103 °C | 14.860 °C | 36.435 °C | 40.688 °C | 44.667 °C | **45.000 °C** |
| **Irradiance $Irr$ ($W/m^2$)** | 599.581 | 266.939 | 0.000 | 437.000 | 632.000 | 827.000 | **1154.000** |

### Panel 2 Clean Parameter Statistics
| Parameter | Mean | Std Dev | Min | 25% (Q1) | Median | 75% (Q3) | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Current $I_2$ (A)** | 1.680 A | 1.709 A | 0.000 A | 0.176 A | 0.999 A | 3.048 A | **7.750 A** |
| **Voltage $V_2$ (V)** | 15.805 V | 3.981 V | 0.000 V | 14.498 V | 16.033 V | 18.826 V | **18.972 V** |
| **Power $P_2$ (W)** | **26.113 W** | **26.776 W** | **0.000 W** | **2.786 W** | **13.156 W** | **47.293 W** | **146.126 W** |
| **Temperature $T_2$ (°C)** | 39.686 °C | 4.990 °C | 15.020 °C | 36.315 °C | 40.756 °C | 44.156 °C | **45.000 °C** |
| **Irradiance $Irr$ ($W/m^2$)** | 599.581 | 266.939 | 0.000 | 437.000 | 632.000 | 827.000 | **1154.000** |

---

## 2. Physical Clean Signatures by Fault Class

| Fault Class | Panel 1 Clean Mean Power | Panel 2 Clean Mean Power | Panel 1 Mean Voltage | Panel 1 Mean Current | Mean Irradiance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal (Healthy)** | **41.025 W** | **39.239 W** | 12.301 V | 2.813 A | 596.50 W/m² |
| **Degradation** | **37.371 W** | **35.144 W** | 16.599 V | 2.284 A | 604.11 W/m² |
| **Open Circuit** | **1.464 W** | **2.277 W** | 18.821 V | 0.078 A | 627.75 W/m² |
| **Shading** | **1.901 W** | **2.352 W** | 12.493 V | 0.146 A | 483.12 W/m² |
| **Short Circuit** | **0.418 W** | **1.180 W** | 0.137 V | 4.714 A | 389.09 W/m² |

![Clean PV Curves](eda_clean_outputs/01_clean_pv_curves.png)

---

## 3. Power vs. Solar Irradiance ($P$ vs. $Irr$)

In the clean dataset, Power generation exhibits a linear correlation ($r = +0.76$) with solar irradiance under `Normal` operation, whereas fault modes cluster distinctly:
* **Normal**: Power increases linearly up to **140W+** as Irradiance scales towards 1000 W/m².
* **Degradation**: Parallel curve shifted downwards due to higher internal series resistance $R_s$.
* **Open & Short Circuit**: Power remains pegged near **0 W** regardless of solar irradiance.

![Power vs Irradiance](eda_clean_outputs/02_power_vs_irradiance.png)

---

## 4. Inter-Panel Differential Diagnostics ($\Delta$)

The differential metrics ($\Delta P = P_1 - P_2$, $\Delta I = I_1 - I_2$, $\Delta V = V_1 - V_2$) now provide a pristine baseline residual:
* Under symmetrical healthy operation, $\Delta P \approx 0\text{ W}$.
* Under single-panel fault events, $|\Delta P| > 25\text{ W}$, providing a clear threshold for unsupervised fault detection.

![Clean Differentials](eda_clean_outputs/05_clean_differentials.png)

---

## 5. Clean Correlation Heatmaps

Power ($P$) now demonstrates mathematical consistency with Current ($I$) and Voltage ($V$), resolving previous inversion noise:

![Clean Correlations](eda_clean_outputs/04_clean_correlations.png)
