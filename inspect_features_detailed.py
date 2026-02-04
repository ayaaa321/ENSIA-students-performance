import joblib
import pandas as pd
from pathlib import Path

meta = joblib.load('models/preprocessor_1y.joblib')
preprocessor = meta['preprocessor']
fit_cols = meta['fit_columns']

print('=' * 80)
print('DETAILED FEATURE ENGINEERING ANALYSIS')
print('=' * 80)

print('\n[LABEL TO PREDICT]')
print(f'  Target: {meta.get("target_column")}')
print(f'  Type: Student performance score (1Y/2Y/3Y average)')
print(f'  Range: ~10-16 (university grades)')

print('\n[INPUT DATA SOURCES]')
print('  - CSV files: data/outputs/grouped_imputed/ensia_cleaned_<version>_grouped_imputed.csv')
print('  - Contains: Student survey responses + behavioral/background data')

print('\n[COLUMNS REMOVED (LEAKAGE / REDUNDANCY)]')
print('  - Dropped: Any "average", "avg", "mean" columns except the target')
print('  - Dropped: *_n columns (suffixes indicating count/frequency per category)')
print('  - Dropped: year_ordinal, inferred_year_from_grades, record_id')

print('\n[ACTUAL FEATURE COLUMNS KEPT]')
print(f'  Total preprocessor input features (before encoding): check preprocessor state')

# Try to inspect the preprocessor's input columns
try:
    # Check ColumnTransformer transformers
    ct = preprocessor
    if hasattr(ct, 'transformers_'):
        print(f'\n[COLUMN TRANSFORMER SUMMARY]')
        for name, transformer, cols in ct.transformers_:
            print(f'  Transformer: {name}')
            print(f'    Columns: {cols}')
            print(f'    Count: {len(cols) if isinstance(cols, list) else "unknown"}')
except Exception as e:
    print(f'  (Could not inspect: {e})')

print(f'\n[ENCODED FEATURES OUTPUT]')
print(f'  Total after encoding: {len(fit_cols)}')
print(f'  Feature names: {fit_cols}')

print('\n[ENCODING STRATEGY DETAILED]')
print('  1. FREQUENCY ENCODING (high-cardinality categorical):')
print('     - For each category value, map to its frequency (proportion) in training data')
freq_maps = meta.get('freq_maps', {})
for col, fmap in freq_maps.items():
    print(f'\n     {col}:')
    for val, freq in list(fmap.items())[:5]:
        print(f'       {val} -> {freq:.4f}')
    if len(fmap) > 5:
        print(f'       ... {len(fmap)-5} more values')

print('\n  2. ONE-HOT ENCODING (low-cardinality categorical, <10 unique):')
print('     - Each category value becomes a binary column (0 or 1)')
print('     - Note: none in this dataset had cardinality < 10')

print('\n  3. NUMERIC SCALING (continuous features):')
print('     - Mean imputation (fill missing with column mean)')
print('     - StandardScaler (subtract mean, divide by std: z = (x - mean)/std)')
print('     - Result: mean ≈ 0, std ≈ 1')
print('     - Note: no explicit numeric features found in fit_columns')

print('\n[WHY THIS ENCODING?]')
print('  - Frequency encoding: captures ordinal info in text (category prevalence)')
print('  - One-hot: prevents artificial ordering in categorical data')
print('  - Scaling: ensures all features on same scale (no single feature dominates)')
print('  - Unified fit: preprocessor fit once on ALL cohorts concatenated')
print('    → ensures consistent feature alignment across 1Y/2Y/3Y')

print('\n' + '=' * 80)
