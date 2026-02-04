from pathlib import Path
import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib

BASE = Path('data/outputs/grouped_imputed')
FILES = {
    '1Y': BASE / 'ensia_cleaned_1Y_grouped_imputed.csv',
    '2Y': BASE / 'ensia_cleaned_2Y_grouped_imputed.csv',
    '3Y': BASE / 'ensia_cleaned_3Y_grouped_imputed.csv',
}

# global fallback candidates (kept for safety)
TARGET_CANDIDATES = [
    'all_modules_avg', 'all_modules_average', 'average', 'avg', 'mean'
]


def clean_name(name, max_len=40):
    s = str(name).strip()
    s = s.lower()
    s = s.replace("\u2019", "'")
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = re.sub(r'_{2,}', '_', s)
    s = s.strip('_')
    if len(s) > max_len:
        parts = s.split('_')
        if len(parts) > 1:
            s = '_'.join(parts[:max_len//2] + parts[-(max_len//2):])
        s = s[:max_len]
    return s


def validate_target_in_df(df, target):
    if target in df.columns:
        return target
    cols_lower = {c.lower(): c for c in df.columns}
    if target.lower() in cols_lower:
        return cols_lower[target.lower()]
    tparts = [p for p in re.split(r'[^a-z0-9]+', target.lower()) if p]
    for c in df.columns:
        cl = c.lower()
        if all(p in cl for p in tparts):
            return c
    return None


def apply_frequency_encoding(df, cols):
    df = df.copy()
    for c in cols:
        freq = df[c].value_counts(dropna=False, normalize=True)
        df[c + '_freq'] = df[c].map(freq).fillna(0.0)
        df = df.drop(columns=[c])
    return df


def build_preprocessor(df, cat_ohe_thresh=10):
    id_like = [c for c in df.columns if re.search(r'id|index|record|student', c, re.IGNORECASE)]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in id_like]
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    ohe_cols = [c for c in obj_cols if df[c].nunique(dropna=False) <= cat_ohe_thresh]
    freq_cols = [c for c in obj_cols if c not in ohe_cols]

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    # OneHotEncoder parameter name changed in newer scikit-learn versions
    try:
        ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    except TypeError:
        ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', ohe_encoder)
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipe, numeric_cols),
        ('ohe', ohe_pipe, ohe_cols),
    ], remainder='drop')
    return preprocessor, numeric_cols, ohe_cols, freq_cols, id_like


def process_version(version):
    p = FILES.get(version)
    if p is None or not p.exists():
        print(f'SKIP {version}: file missing {p}')
        return False
    print(f'Processing {version} -> {p}')
    df = pd.read_csv(p)
    # rename
    rename_map = {c: clean_name(c) for c in df.columns}
    df = df.rename(columns=rename_map)
    # prefer version-specific all_modules_avg / average names
    version_key = version.lower()
    version_candidates = [
        f"{version_key}_average",
        f"{version_key}_avg",
        f"{version_key}_all_modules_avg",
        f"{version_key}_all_modules_average",
        f"{version_key}_mean",
    ]
    candidate = None
    for t in version_candidates:
        cand = validate_target_in_df(df, t)
        if cand:
            candidate = cand
            break
    # fallback: look for any 'all' + 'avg'/'average' column
    if candidate is None:
        candidates = [c for c in df.columns if ('all' in c.lower() and ('avg' in c.lower() or 'average' in c.lower()))]
        if not candidates:
            candidates = [c for c in df.columns if ('average' in c.lower() or 'avg' in c.lower() or 'mean' in c.lower())]
        # if multiple, prefer columns that include the version key
        if candidates:
            vcands = [c for c in candidates if version_key in c.lower()]
            candidate = vcands[0] if vcands else candidates[0]
    if candidate is None:
        print(f'No target found for {version}; skipping')
        return False
    print('Selected target:', candidate)
    # drop leakage columns
    leakage_cols = [c for c in df.columns if (('avg' in c.lower() or 'average' in c.lower() or 'mean' in c.lower()) and c != candidate)]
    print('Dropping leakage columns:', leakage_cols)
    df_for_preproc = df.drop(columns=leakage_cols, errors='ignore')
    # drop user-specified unneeded features (if present)
    drop_unneeded = [
        '1y_math_n', '1y_programming_n', '1y_other_n', '1y_all_modules_n',
        '2y_math_n', '2y_programming_n', '2y_other_n', '2y_all_modules_n',
        '3y_math_n', '3y_programming_n', '3y_other_n', '3y_all_modules_n',
        'year_ordinal', 'inferred_year_from_grades',
    ]
    df_for_preproc = df_for_preproc.drop(columns=[c for c in drop_unneeded if c in df_for_preproc.columns], errors='ignore')
    # save cleaned csv and return df_for_preproc and target
    out_dir = Path('data/processed')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f'ensia_{version.lower()}_preprocessed.csv'
    df_for_preproc.to_csv(out_csv, index=False)
    print('Saved cleaned CSV:', out_csv)
    return {'version': version, 'df': df_for_preproc, 'target': candidate}


if __name__ == '__main__':
    infos = []
    for v in ['1Y', '2Y', '3Y']:
            print('This preprocessing script was moved to backups/regression_backup_2026-02-02/run_preprocessing.py')
        if info:
            infos.append(info)
    if not infos:
        print('No versions processed; exiting')
        raise SystemExit(1)

    # Build a single preprocessor fit on the concatenated, leakage-cleaned feature set
    print('Building unified preprocessor from all versions...')
    # concatenate dfs after dropping each version's target
    dfs_for_fit = []
    for info in infos:
        df_local = info['df']
        tgt = info['target']
        if tgt and tgt in df_local.columns:
            df_local = df_local.drop(columns=[tgt], errors='ignore')
        dfs_for_fit.append(df_local)
    df_fit_all = pd.concat(dfs_for_fit, ignore_index=True, sort=False)

    # determine freq_cols and id_like using build_preprocessor
    preproc_template, numeric_cols, ohe_cols, freq_cols, id_like = build_preprocessor(df_fit_all)

    # compute frequency encoding maps on df_fit_all to apply consistently
    freq_maps = {}
    for c in freq_cols:
        freq_maps[c] = df_fit_all[c].value_counts(dropna=False, normalize=True).to_dict()

    def apply_frequency_encoding_with_maps(df, freq_maps):
        df = df.copy()
        for c, fmap in freq_maps.items():
            if c in df.columns:
                df[c + '_freq'] = df[c].map(fmap).fillna(0.0)
                df = df.drop(columns=[c])
        return df

    df_fit_all_enc = apply_frequency_encoding_with_maps(df_fit_all, freq_maps) if freq_maps else df_fit_all.copy()
    df_fit_all_enc = df_fit_all_enc.drop(columns=[c for c in id_like if c in df_fit_all_enc.columns], errors='ignore')

    # fit the unified preprocessor
    preproc_template.fit(df_fit_all_enc)
    # record the fitted feature columns for later alignment
    fit_columns = df_fit_all_enc.columns.tolist()

    # save per-version preprocessor files that reference the same fitted preprocessor and freq_maps
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    for info in infos:
        version = info['version']
        out_path = models_dir / f'preprocessor_{version.lower()}.joblib'
        joblib.dump({
            'preprocessor': preproc_template,
            'freq_maps': freq_maps,
            'freq_cols': freq_cols,
            'fit_columns': fit_columns,
            'id_like': id_like,
            'target_column': info['target']
        }, out_path)
        print('Saved unified preprocessor for', version, '->', out_path)

    print('Done')
