import joblib
from pathlib import Path

meta = joblib.load('models/preprocessor_1y.joblib')

print('=' * 80)
print('INPUT FEATURES & LABEL INFORMATION')
print('=' * 80)

print('\n[1] TARGET LABEL (what we predict):')
print(f'    {meta.get("target_column")}')

print('\n[2] ID-LIKE COLUMNS (excluded from features):')
for col in meta.get('id_like', []):
    print(f'    - {col}')

print('\n[3] FREQUENCY-ENCODED CATEGORICAL FEATURES:')
freq_cols = meta.get('freq_cols', [])
print(f'    Count: {len(freq_cols)}')
for i, col in enumerate(freq_cols, 1):
    print(f'    {i}. {col}')

print('\n[4] NUMERIC FEATURES (mean-imputed + standardscaled):')
num_cols = [c for c in meta.get('fit_columns', []) if c.startswith('num__')]
print(f'    Count: {len(num_cols)}')
for i, col in enumerate(num_cols, 1):
    print(f'    {i}. {col}')

print('\n[5] ONE-HOT ENCODED CATEGORICAL FEATURES (low cardinality):')
ohe_cols = [c for c in meta.get('fit_columns', []) if c.startswith('ohe__')]
print(f'    Count: {len(ohe_cols)}')
print('    Sample (first 25):')
for i, col in enumerate(ohe_cols[:25], 1):
    print(f'    {i}. {col}')
if len(ohe_cols) > 25:
    print(f'    ... and {len(ohe_cols) - 25} more')

print('\n[6] ENCODING PIPELINE SUMMARY:')
fit_cols = meta.get('fit_columns', [])
print(f'    Total transformed features: {len(fit_cols)}')
print(f'    Numeric: {len(num_cols)}')
print(f'    One-hot: {len(ohe_cols)}')
print(f'    Frequency-encoded: {len(freq_cols)}')

print('\n[7] FEATURE TRANSFORMATION STEPS:')
print('    Step 1: Clean column names (lowercase, snake_case)')
print('    Step 2: Drop leakage columns (derived averages except target)')
print('    Step 3: Drop user-specified columns (_n suffixes, year_ordinal, etc.)')
print('    Step 4: Frequency-encode high-cardinality categoricals')
print('    Step 5: One-hot encode low-cardinality categoricals')
print('    Step 6: Mean-impute numeric; StandardScale both')
print('    Step 7: Drop ID-like columns')

print('\n' + '=' * 80)
