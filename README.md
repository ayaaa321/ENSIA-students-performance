# ENSIA Student Performance Prediction

A machine learning pipeline for predicting ENSIA student performance using Ridge regression and Random Forest ensemble models with stacking meta-learner.

## 📊 Project Overview

This project implements a comprehensive regression pipeline to predict student performance scores based on 31 engineered features derived from academic, behavioral, and engagement metrics. The final model uses a stacking ensemble approach achieving **1.2008 MAE** (3.19% improvement over baseline).

## 🎯 Key Results

- **Best Model**: Stacking Ensemble (Ridge + RandomForest)
- **MAE**: 1.2008 ± 0.1139 (on 9-17 point scale)
- **RMSE**: 1.5653
- **R²**: 0.1016
- **Performance**: Consistent across 5-fold cross-validation

## 📁 Project Structure

```
├── notebooks/
│   ├── regression_full.ipynb           # Main regression pipeline
│   ├── preprocessing_lab_methodology.ipynb
│   └── Feature_selection_Non_Guided_.ipynb
├── data/
│   ├── raw/                            # Original datasets
│   ├── processed/                      # Cleaned and merged data
│   ├── intermediate/                   # Preprocessing outputs
│   └── outputs/                        # Model predictions & visualizations
├── models/                             # Trained model artifacts
├── scripts/                            # Utility scripts
├── FEATURE_ENGINEERING.txt             # Feature documentation
├── ENSEMBLE_METHODS_ANALYSIS.txt       # Ensemble methodology
└── requirements.txt                    # Python dependencies
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.10+
- pip or conda

### Quick Start

```bash
# Clone repository
git clone <repo-url>
cd ENSIA-students-performance

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Jupyter
jupyter notebook notebooks/regression_full.ipynb
```

## 📊 Data Pipeline

### 1. Data Preparation
- **Source**: 225 pooled samples (147 1Y + 70 2Y + 8 3Y cohorts)
- **Features**: 31 engineered features
  - 20 One-Hot Encoded (categorical)
  - 8 Numeric (scaled)
  - 3 Frequency-encoded

### 2. Preprocessing
- Unified ColumnTransformer fitted on pooled data
- StandardScaler for numeric features
- OneHotEncoder for categorical variables
- Frequency encoding for engagement categories

### 3. Feature Engineering
- Tested 16 interaction & ratio features
- Conclusion: **Original 31 features optimal** (engineered features degraded performance)

## 🤖 Model Architecture

### Baseline Models (Tuned)
- **Ridge Regression**: α=10.0 (MAE: 1.2404)
- **RandomForest**: n_estimators=100, max_depth=10, max_features=0.5 (MAE: 1.2644)

### Ensemble Methods (5-Fold CV)
1. **Voting (Average)**: 1.2186 MAE (+1.75% vs Ridge)
2. **Voting (Weighted)**: 1.2185 MAE (+1.76% vs Ridge)
3. **Stacking** ✅: 1.2008 MAE (+3.19% vs Ridge) **← SELECTED**

### Stacking Architecture
```
Input Features (31)
    ↓
Ridge (α=10.0) ──┐
                  ├→ Meta-Features (2) → Ridge Meta-Learner → Final Prediction
RandomForest ────┘
```

## 📈 Model Evaluation

### Alternative Models Tested
| Model | MAE | vs Ridge |
|-------|-----|----------|
| **Stacking Ensemble** | **1.2008** | **+3.19%** ✅ |
| Ridge (Baseline) | 1.2404 | — |
| RandomForest | 1.2644 | -1.93% |
| GradientBoosting | 1.2786 | -3.08% |
| Lasso | 1.2796 | -3.16% |
| SVR | 1.2915 | -4.12% |
| ElasticNet | 1.2938 | -4.31% |
| ExtraTrees | 1.3230 | -6.66% |

## 📊 Outputs

### Predictions
- `data/outputs/predictions/renamed/ensemble/ensemble_predictions_kfold.csv`
  - 225 samples with actual vs predicted values
  - Per-fold breakdown
  - Error metrics

### Visualizations
- `ensemble_actual_vs_predicted.png` - 4-panel analysis
  - Scatter: Actual vs Predicted
  - Residuals: Prediction errors
  - Error distribution
  - Per-fold MAE comparison

## 📚 Documentation

- **FEATURE_ENGINEERING.txt** (450+ lines)
  - 31 features with rationale
  - Data leakage verification
  - Preprocessing pipeline details

- **ENSEMBLE_METHODS_ANALYSIS.txt** (700+ lines)
  - Voting and stacking methodology
  - Cross-fold performance
  - Architecture justification

- **COMPREHENSIVE_MODEL_COMPARISON_ANALYSIS.txt**
  - K-fold vs Train/Test comparison
  - Model reliability analysis
  - Performance metrics

## 🚀 Deployment

The ensemble model is production-ready:
- ✅ Properly validated with k-fold CV
- ✅ No overfitting detected
- ✅ Best performer among 8 models tested
- ✅ Fast inference (~6ms per prediction)

### Expected Performance
- **MAE**: 1.20 ± 0.11 points
- **Suitable for**: Identifying at-risk students, trend analysis
- **Not suitable for**: Exact score prediction (±1.2 uncertainty)

## 💡 Limitations & Recommendations

### Current Limitations
1. **Low Predictability** (R²=0.1): 90% of variance unexplained
2. **Compressed Predictions**: Regresses toward mean
3. **Small Dataset**: 225 samples limits complexity
4. **Missing Features**: Study habits, prior grades, external factors not captured

### Recommendations for Improvement
1. **Collect Additional Data**
   - Attendance records
   - Assignment completion rates
   - Previous semester grades
   - Study time logs

2. **Feature Enhancement**
   - Temporal patterns (grade trends)
   - Interaction effects
   - Domain-specific educational metrics

3. **Model Considerations**
   - Monitor performance on new cohorts
   - Implement confidence intervals
   - Use predictions for risk flagging (>1.2 MAE = manual review)

## 📦 Requirements

```
numpy==1.26.4
pandas==2.1.4
scikit-learn==1.7.2
matplotlib==3.8.2
joblib==1.3.2
```

See `requirements.txt` for full specifications.

## 👥 Author

Created as part of ENSIA data mining curriculum

## 📝 License

[Specify your license here]
