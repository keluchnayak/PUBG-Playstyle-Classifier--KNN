# PUBG Playstyle Classifier: KNN Binary Classification Pipeline

A comprehensive machine learning implementation using **K-Nearest Neighbors (KNN)** binary classification to analyze PlayerUnknown's Battlegrounds (PUBG) player telemetry and classify playstyle profiles.

---

## Project Folder Structure

PUBG Aggressive Fragger vs. Passive Looter Classifier/
     |
     ├── pubg_playstyle_knn.ipynb      
     |
     |
     └── data.csv  

## 1. Project Overview & Problem Statement

In competitive battle royale matches, player behaviors exhibit distinct tactical patterns ranging from stealthy survival to high-intensity combat. This project constructs an end-to-end classification pipeline to categorize PUBG players into two distinct operational profiles:

* **Aggressive Fragger (`1`)**: High-combat players securing 3 or more eliminations (`kills >= 3`).
* **Passive / Stealth Looter (`0`)**: Defensive or loot-focused players securing fewer than 3 eliminations (`kills < 3`).

Built using **Python**, **Pandas**, and **Scikit-Learn**, the pipeline implements Z-score feature scaling, stratified data partitioning, distance metric benchmarking (Euclidean, Minkowski, Manhattan, Chebyshev, and Hamming), decision tree comparison, unseen sample inference, and 3D spatial cluster visualization.

---

## 2. Dataset Schema & Feature Selection

Download Dataset - https://www.kaggle.com/datasets/razamh/pubg-dataset

The dataset incorporates four core player telemetry fields extracted from match logs:

### Predictor Features Matrix ($X$)
1. **`damageDealt`** *(Float)*: Total damage inflicted on opponent players in Hit Points (HP).
2. **`headshotKills`** *(Integer)*: Count of enemy eliminations secured via headshots.
3. **`killStreaks`** *(Integer)*: Maximum number of kills achieved within a short time window.

### Target Outcome Vector ($y$)
* **`playstyle_label`** *(Binary Integer)*:
  * `1` — Aggressive Fragger (derived from `kills >= 3`)
  * `0` — Passive / Stealth Looter (derived from `kills < 3`)

---

## 3. Machine Learning Pipeline Architecture

The workflow follows eight sequential steps aligned with standard machine learning practices:

```
[ Dataset Loading / Synthetic Fallback ]
                  │
                  ▼
[ Feature (X) & Label (y) Separation ]
                  │
                  ▼
[ Stratified Train-Test Split (80/20) ]
                  │
                  ▼
[ Z-Score Feature Standardization (StandardScaler) ]
                  │
                  ▼
[ Baseline KNN Classifier (K=5, Euclidean) ]
                  │
                  ▼
[ Distance Metrics Benchmarking (5 Metrics + Decision Tree) ]
                  │
                  ▼
[ Real-Time Unseen Player Prediction & Confidence ]
                  │
                  ▼
[ 3D Feature Space Visualization (Matplotlib mplot3d) ]
```

### Step 1: Data Ingestion & Preprocessing
The program checks for a local Kaggle `data.csv` file containing match records. If unavailable, it dynamically generates a balanced 1,000-sample synthetic dataset (500 passive, 500 aggressive players) reflecting realistic gameplay stat distributions.

### Step 2: Feature & Label Separation ($X$ vs $y$)
The predictor columns (`damageDealt`, `headshotKills`, `killStreaks`) are isolated into feature matrix $X$, while `playstyle_label` is assigned to target vector $y$.

### Step 3: Stratified Train-Test Split (80/20)
The data is partitioned using `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`. Stratification ensures both training (800 samples) and testing (200 samples) sets maintain an exact 50:50 class balance.

### Step 4: Z-Score Feature Standardization (`StandardScaler`)
Because `damageDealt` values range up to 1,200 HP while `headshotKills` and `killStreaks` range from 0 to 5, unscaled distance metrics would be dominated by damage values. **Z-score standardization** translates features into standard deviation units ($z = \frac{x - \mu}{\sigma}$), ensuring equal weighting in distance calculations.

### Step 5: Baseline KNN Training & Evaluation
A baseline KNN model ($K = 5$, `metric='euclidean'`) is trained on scaled features. Predictions on the test set are generated and accuracy is computed.

### Step 6: Distance Metrics Benchmarking
The script evaluates model sensitivity across five distance metrics and compares performance against a Decision Tree baseline:
* **Euclidean Distance ($p = 2$)**: Standard straight-line metric ($d(x, y) = \sqrt{\sum (x_i - y_i)^2}$).
* **Minkowski Distance ($p = 3$)**: Generalized metric ($d(x, y) = (\sum |x_i - y_i|^p)^{1/p}$).
* **Manhattan Distance ($L_1$)**: City-block grid distance ($d(x, y) = \sum |x_i - y_i|$).
* **Chebyshev Distance ($L_{\infty}$)**: Maximum coordinate difference ($d(x, y) = \max(|x_i - y_i|)$).
* **Hamming Distance**: Discrete attribute mismatch ratio.
* **Decision Tree Classifier**: Non-parametric baseline using recursive binary splitting (`random_state=42`).

### Step 7: Real-Time Unseen Sample Inference
New player telemetry inputs are transformed through the fitted `StandardScaler` and classified using `predict()` and `predict_proba()` to compute probability confidence scores.

### Step 8: 3D Spatial Cluster Visualization
Using Matplotlib's `mplot3d` toolkit, the script plots all 1,000 player samples in 3D space (`damageDealt` $\times$ `headshotKills` $\times$ `killStreaks`), rendering Aggressive Fraggers in red and Passive Looters in blue.

---

## 4. Distance Metrics & Baseline Performance Comparison

| Algorithm / Metric | Parameters | Mathematical Concept | Classification Accuracy |
| :--- | :--- | :--- | :---: |
| **KNN (Euclidean)** | `n_neighbors=5`, `p=2` | Straight-line spatial distance | **100.0%** |
| **KNN (Minkowski)** | `n_neighbors=5`, `p=3` | Higher-order norm distance | **100.0%** |
| **KNN (Manhattan)** | `n_neighbors=5`, `metric='manhattan'` | Absolute grid distance ($L_1$) | **100.0%** |
| **KNN (Chebyshev)** | `n_neighbors=5`, `metric='chebyshev'` | Maximum coordinate offset ($L_{\infty}$) | **100.0%** |
| **KNN (Hamming)** | `n_neighbors=5`, `metric='hamming'` | Feature value mismatch ratio | **92.5%** |
| **Decision Tree** | `random_state=42` | Recursive feature thresholding | **100.0%** |

---

## 5. Sample Unseen Player Inferences

```python
# Sample 1: High Damage & Combat Streaks
Input: [[650.0 HP, 2 Headshots, 2 Kill Streak]]
Predicted Playstyle: Aggressive Fragger
Confidence: 100.0%

# Sample 2: Low Damage & Zero Combat Streaks
Input: [[85.0 HP, 0 Headshots, 0 Kill Streak]]
Predicted Playstyle: Passive / Stealth Looter
Confidence: 100.0%
```

---

## 6. Execution Instructions

### Installation
Ensure Python 3.8+ and standard scientific computing packages are installed:
```bash
pip install numpy pandas matplotlib scikit-learn
```

### Running the Script
Run the script directly from your terminal:
```bash
python pubg_playstyle_complete_notebook.py
```

### Generated Artifacts
* **Terminal Summary**: Displays dataset shapes, class distributions, distance benchmark accuracy comparisons, and sample inferences.
* **Graphic File**: Generates and saves `/workspace/scratch/pubg_playstyle_3d_cluster.png`.

---

## 7. Conclusion

Feature scaling via `StandardScaler` plays a vital role in balancing continuous damage metrics with discrete kill streak counters. The benchmarking results show that Euclidean, Minkowski, Manhattan, and Chebyshev metrics achieve optimal separation between PUBG playstyle profiles, providing a robust framework for player behavior classification.
