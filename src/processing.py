import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MEDICATION_COLUMNS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
    'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone'
]

SUMMARY_URL = 'https://en.wikipedia.org/wiki/'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/115.0 Safari/537.36'
}


def load_data(path='data/diabetic_data.csv'):
    return pd.read_csv(path)


def parse_age(age_bucket):
    if pd.isna(age_bucket):
        return np.nan
    match = re.search(r"\[(\d+)-(\d+)\)", str(age_bucket))
    if match:
        low = int(match.group(1))
        high = int(match.group(2))
        return (low + high) / 2
    return np.nan


def bool_to_numeric(value):
    if pd.isna(value):
        return 0
    value = str(value).strip().lower()
    return 1 if value == 'yes' else 0


def clean_diabetes_data(df):
    df = df.copy()
    df['age_years'] = df['age'].apply(parse_age)
    df['readmitted_30'] = (df['readmitted'] == '<30').astype(int)
    df['change_flag'] = (df['change'] == 'Ch').astype(int)
    df['diabetesMed_flag'] = df['diabetesMed'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    df['gender'] = df['gender'].replace('Unknown/Invalid', np.nan)
    df['weight'] = df['weight'].replace('?', np.nan)

    for med in MEDICATION_COLUMNS:
        if med in df.columns:
            df[med] = df[med].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    numeric_cols = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
        'number_outpatient', 'number_emergency', 'number_inpatient',
        'number_diagnoses', 'age_years'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def audit_data(df):
    summary = {
        'shape': df.shape,
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_counts': df.isna().sum().to_dict(),
        'missing_pct': (df.isna().mean() * 100).round(2).to_dict(),
    }

    numeric = df.select_dtypes(include=['int64', 'float64'])
    outliers = {}
    for col in numeric.columns:
        q1 = numeric[col].quantile(0.25)
        q3 = numeric[col].quantile(0.75)
        iqr = q3 - q1
        outliers[col] = int(((numeric[col] < (q1 - 1.5 * iqr)) |
                             (numeric[col] > (q3 + 1.5 * iqr))).sum())
    summary['outlier_counts'] = outliers
    return summary


def fetch_wikipedia_summary(drug):
    try:
        url = SUMMARY_URL + str(drug).replace(' ', '_')
        resp = requests.get(url, headers=HEADERS, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        paragraph = soup.select_one('p')
        text = paragraph.get_text(strip=True) if paragraph is not None else ''
        return drug, text
    except Exception:
        return drug, ''


def scrape_medication_summaries_sequential(drug_list):
    start = time.time()
    summaries = {}
    for drug in drug_list:
        _, text = fetch_wikipedia_summary(drug)
        summaries[drug] = text
    return summaries, time.time() - start


def scrape_medication_summaries_parallel(drug_list, workers=4):
    start = time.time()
    summaries = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_wikipedia_summary, drug): drug for drug in drug_list}
        for future in futures:
            drug, text = future.result()
            summaries[drug] = text
    return summaries, time.time() - start


def medication_class_from_summary(summary):
    text = str(summary).lower()
    if 'insulin' in text:
        return 'Insulin'
    if 'sulfonylurea' in text:
        return 'Sulfonylurea'
    if 'biguanide' in text or 'metformin' in text:
        return 'Biguanide'
    if 'thiazolidinedione' in text or 'glitazone' in text:
        return 'Thiazolidinedione'
    if 'alpha-glucosidase' in text or 'acarbose' in text:
        return 'Alpha-glucosidase inhibitor'
    if 'meglitinide' in text or 'repaglinide' in text or 'nateglinide' in text:
        return 'Meglitinide'
    return 'Unknown'


def build_medication_class_table(summaries):
    rows = []
    for drug, text in summaries.items():
        rows.append({
            'drug': drug,
            'summary': text,
            'med_class': medication_class_from_summary(text)
        })
    return pd.DataFrame(rows)


def augment_with_medication_classes(df, med_class_table):
    df = df.copy()
    class_map = med_class_table.set_index('drug')['med_class'].to_dict()
    for med_class in sorted(set(class_map.values())):
        if med_class == 'Unknown':
            continue
        valid_drugs = [drug for drug, cls in class_map.items() if cls == med_class and drug in df.columns]
        if valid_drugs:
            df[f'{med_class.lower().replace(" ", "_")}_count'] = df[valid_drugs].sum(axis=1)
    return df


def build_pipeline():
    numeric_features = [
        'age_years', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient',
        'number_diagnoses'
    ]
    categorical_features = ['race', 'gender', 'payer_code', 'medical_specialty']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, solver='liblinear'))
    ])
    return pipeline


def train_readmission_model(df):
    features = [
        'age_years', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient',
        'number_diagnoses', 'change_flag', 'diabetesMed_flag'
    ]
    features += [c for c in df.columns if c.endswith('_count')]
    features = [f for f in features if f in df.columns]

    df = df.dropna(subset=features + ['readmitted_30'])
    X = df[features]
    y = df['readmitted_30']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = build_pipeline()
    param_grid = {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__penalty': ['l2']
    }
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    y_prob = grid.predict_proba(X_test)[:, 1]

    metrics = {
        'best_params': grid.best_params_,
        'roc_auc': roc_auc_score(y_test, y_prob),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    return grid.best_estimator_, metrics


def analyze_change_effect(df):
    df = df.copy()
    cols = ['time_in_hospital', 'num_lab_procedures', 'num_medications',
            'number_inpatient', 'age_years', 'change_flag']
    df = df.dropna(subset=cols + ['readmitted_30'])
    X = df[cols]
    y = df['readmitted_30']
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X, y)
    coeffs = dict(zip(cols, model.coef_[0]))
    return {
        'coefficients': coeffs,
        'intercept': float(model.intercept_[0]),
        'change_flag_odds_ratio': float(np.exp(coeffs['change_flag']))
    }


def save_cleaned_data(df, path='data/diabetic_data_cleaned.csv'):
    df.to_csv(path, index=False)
    return path
