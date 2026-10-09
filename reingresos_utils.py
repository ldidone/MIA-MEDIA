"""Funciones compartidas por los notebooks v2 (UCI) y MIMIC-IV de reingreso a 30 días."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler

SEED = 42
MAX_CATEGORIES = 64
MIN_FREQUENCY = 30
# Razón minoritaria/mayoritaria tras remuestrear (1:2). Igual para undersampling y SMOTENC.
SAMPLING_RATIO = 0.5

# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def top_metrics(yt, prob, q=0.20, seed=SEED):
    """Selecciona la fracción q con mayor score; empates resueltos al azar con semilla fija."""
    yt, prob = np.asarray(yt), np.asarray(prob)
    k = max(1, int(np.ceil(q * len(yt))))
    tie = np.random.default_rng(seed).random(len(yt))
    chosen = np.lexsort((tie, -prob))[:k]
    tp = int(yt[chosen].sum()); total = int(yt.sum())
    return {'selected': k, 'tp': tp, 'fp': k - tp, 'fn': total - tp,
            'precision': tp / k, 'recall': tp / total if total else np.nan,
            'lift': float((tp / k) / yt.mean()) if yt.mean() > 0 else np.nan,
            'fraction': k / len(yt), 'threshold': float(prob[chosen[-1]])}


def metrics(yt, prob, q=0.20):
    m = top_metrics(yt, prob, q)
    return {'AUROC': roc_auc_score(yt, prob), 'AP': average_precision_score(yt, prob),
            'Brier': brier_score_loss(yt, prob), 'Precision@20': m['precision'],
            'Recall@20': m['recall'], 'Lift@20': m['lift']}


def wilson_ci(k, n, z=1.96):
    """IC de Wilson para una proporción k/n."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    den = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return center - half, center + half


def net_benefit(yt, prob, pt):
    """Beneficio neto al umbral pt: TP/n - FP/n * pt/(1-pt)."""
    yt, prob = np.asarray(yt), np.asarray(prob)
    flag = prob >= pt
    n = len(yt)
    tp = np.sum(flag & (yt == 1)); fp = np.sum(flag & (yt == 0))
    return tp / n - fp / n * pt / (1 - pt)

# ---------------------------------------------------------------------------
# Códigos CIE-9 / CIE-10: agrupación por capítulo y Charlson
# ---------------------------------------------------------------------------

def normalize_icd(code, version=9):
    """Código sin punto. CIE-9 numérico se completa a 3 dígitos (UCI elimina ceros iniciales)."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None
    s = str(code).strip().upper()
    if s in ('', '?', 'NAN', 'NONE'):
        return None
    head, _, tail = s.partition('.')
    if version == 9 and head[0] not in 'VE':
        head = head.zfill(3)
    return head + tail


def icd9_group(code, dotless=False):
    """Agrupación de Strack et al. (2014) para CIE-9. `dotless=True` para códigos sin punto (MIMIC)."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return 'Missing'
    s = str(code).strip().upper()
    if s in ('', '?', 'NAN'):
        return 'Missing'
    if s[0] in 'VE':
        return 'Other'
    if dotless:
        s = s[:3] + ('.' + s[3:] if len(s) > 3 else '')
    x = float(s); i = int(x)
    if 390 <= i <= 459 or i == 785: return 'Circulatory'
    if 460 <= i <= 519 or i == 786: return 'Respiratory'
    if 520 <= i <= 579 or i == 787: return 'Digestive'
    if i == 250: return 'Diabetes'
    if 800 <= i <= 999: return 'Injury'
    if 710 <= i <= 739: return 'Musculoskeletal'
    if 580 <= i <= 629 or i == 788: return 'Genitourinary'
    if 140 <= i <= 239: return 'Neoplasms'
    return 'Other'


def icd10_group(code):
    """Equivalente CIE-10 a nivel de capítulo de la agrupación de Strack (código sin punto)."""
    s = normalize_icd(code, 10)
    if s is None:
        return 'Missing'
    letter, num = s[0], s[1:3]
    n = int(num) if num.isdigit() else -1
    if letter == 'I' or (letter == 'R' and 0 <= n <= 3): return 'Circulatory'
    if letter == 'J' or (letter == 'R' and 4 <= n <= 9): return 'Respiratory'
    if letter == 'K' or (letter == 'R' and 10 <= n <= 19): return 'Digestive'
    if letter == 'E' and 8 <= n <= 14: return 'Diabetes'
    if letter in 'ST': return 'Injury'
    if letter == 'M': return 'Musculoskeletal'
    if letter == 'N' or (letter == 'R' and 30 <= n <= 39): return 'Genitourinary'
    if letter == 'C' or (letter == 'D' and 0 <= n <= 49): return 'Neoplasms'
    return 'Other'


def _rng(prefix, a, b, width):
    return [f'{prefix}{i:0{width}d}' for i in range(a, b + 1)]


# Quan et al. (2005), Med Care 43:1130. Prefijos de códigos sin punto.
CHARLSON_ICD9 = {
    'mi': ['410', '412'],
    'chf': ['39891', '40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491',
            '40493', *_rng('425', 4, 9, 1), '428'],
    'pvd': ['0930', '4373', '440', '441', *_rng('443', 1, 9, 1), '4471', '5571', '5579', 'V434'],
    'cevd': ['36234', *_rng('', 430, 438, 3)],
    'dementia': ['290', '2941', '3312'],
    'cpd': ['4168', '4169', *_rng('', 490, 505, 3), '5064', '5081', '5088'],
    'rheum': ['4465', *_rng('710', 0, 4, 1), '7140', '7141', '7142', '7148', '725'],
    'pud': _rng('', 531, 534, 3),
    'mild_liver': ['07022', '07023', '07032', '07033', '07044', '07054', '0706', '0709', '570',
                   '571', '5733', '5734', '5738', '5739', 'V427'],
    # '250' sin decimales cuenta como diabetes; la jerarquía elimina el duplicado si hay complicación.
    'dm': ['250'],
    'dm_comp': _rng('250', 4, 7, 1),
    'hemiplegia': ['3341', '342', '343', *_rng('344', 0, 6, 1), '3449'],
    'renal': ['40301', '40311', '40391', '40402', '40403', '40412', '40413', '40492', '40493',
              '582', *_rng('583', 0, 7, 1), '585', '586', '5880', 'V420', 'V451', 'V56'],
    'cancer': [*_rng('', 140, 172, 3), *_rng('', 174, 194, 3), *_rng('195', 0, 8, 1),
               *_rng('', 200, 208, 3), '2386'],
    'severe_liver': [*_rng('456', 0, 2, 1), *_rng('572', 2, 8, 1)],
    'mets': _rng('', 196, 199, 3),
    'aids': _rng('', 42, 44, 3),
}
_DM10 = ['E08', 'E09', 'E10', 'E11', 'E12', 'E13', 'E14']
CHARLSON_ICD10 = {
    'mi': ['I21', 'I22', 'I252'],
    'chf': ['I099', 'I110', 'I130', 'I132', 'I255', 'I420', *_rng('I42', 5, 9, 1), 'I43', 'I50',
            'P290'],
    'pvd': ['I70', 'I71', 'I731', 'I738', 'I739', 'I771', 'I790', 'I792', 'K551', 'K558', 'K559',
            'Z958', 'Z959'],
    'cevd': ['G45', 'G46', 'H340', *_rng('I', 60, 69, 2)],
    'dementia': [*_rng('F0', 0, 3, 1), 'F051', 'G30', 'G311'],
    'cpd': ['I278', 'I279', *_rng('J', 40, 47, 2), *_rng('J', 60, 67, 2), 'J684', 'J701', 'J703'],
    'rheum': ['M05', 'M06', 'M315', *_rng('M3', 2, 4, 1), 'M351', 'M353', 'M360'],
    'pud': _rng('K2', 5, 8, 1),
    'mild_liver': ['B18', *_rng('K70', 0, 3, 1), 'K709', *_rng('K71', 3, 5, 1), 'K717', 'K73', 'K74',
                   'K760', *_rng('K76', 2, 4, 1), 'K768', 'K769', 'Z944'],
    'dm': [p + d for p in _DM10 for d in '01689'],
    'dm_comp': [p + d for p in _DM10 for d in '23457'],
    'hemiplegia': ['G041', 'G114', 'G801', 'G802', 'G81', 'G82', *_rng('G83', 0, 4, 1), 'G839'],
    'renal': ['I120', 'I131', *_rng('N03', 2, 7, 1), *_rng('N05', 2, 7, 1), 'N18', 'N19', 'N250',
              *_rng('Z49', 0, 2, 1), 'Z940', 'Z992'],
    'cancer': [*_rng('C', 0, 26, 2), *_rng('C', 30, 34, 2), *_rng('C', 37, 41, 2), 'C43',
               *_rng('C', 45, 58, 2), *_rng('C', 60, 76, 2), *_rng('C', 81, 85, 2), 'C88',
               *_rng('C', 90, 97, 2)],
    'severe_liver': ['I850', 'I859', 'I864', 'I982', 'K704', 'K711', 'K721', 'K729', 'K765',
                     'K766', 'K767'],
    'mets': _rng('C', 77, 80, 2),
    'aids': [*_rng('B', 20, 22, 2), 'B24'],
}
CHARLSON_WEIGHTS = {'mi': 1, 'chf': 1, 'pvd': 1, 'cevd': 1, 'dementia': 1, 'cpd': 1, 'rheum': 1,
                    'pud': 1, 'mild_liver': 1, 'dm': 1, 'dm_comp': 2, 'hemiplegia': 2, 'renal': 2,
                    'cancer': 2, 'severe_liver': 3, 'mets': 6, 'aids': 6}


def comorbidity_flags(codes, version=9):
    """Indicadores de Charlson por fila de código. `codes` puede ser Series de códigos crudos."""
    rules = CHARLSON_ICD9 if version == 9 else CHARLSON_ICD10
    codes = pd.Series(codes)
    uniq = codes.dropna().unique()
    table = pd.DataFrame({k: [bool(n) and n.startswith(tuple(p))
                              for n in (normalize_icd(c, version) for c in uniq)]
                          for k, p in rules.items()}, index=uniq)
    out = table.reindex(codes.to_numpy(), fill_value=False).astype(bool)
    out.index = codes.index
    return out


def charlson_score(flags):
    """Puntaje de Charlson con jerarquía (complicada > simple, metástasis > cáncer, severa > leve)."""
    f = flags.astype(bool).copy()
    f['dm'] &= ~f['dm_comp']
    f['cancer'] &= ~f['mets']
    f['mild_liver'] &= ~f['severe_liver']
    return sum(f[k].astype(int) * w for k, w in CHARLSON_WEIGHTS.items())

# ---------------------------------------------------------------------------
# Scores clínicos: LACE y HOSPITAL
# ---------------------------------------------------------------------------

def lace_score(los_days, emergent, charlson, ed_visits):
    """LACE (van Walraven 2010). L: días de estancia; A: ingreso agudo; C: Charlson; E: guardias previas."""
    los = np.floor(np.asarray(los_days, float))
    L = np.select([los < 1, los < 2, los < 3, los < 4, los < 7, los < 14], [0, 1, 2, 3, 4, 5], 7)
    A = 3 * np.asarray(emergent, bool).astype(int)
    c = np.asarray(charlson, float)
    C = np.where(c >= 4, 5, c)
    E = np.minimum(np.nan_to_num(np.asarray(ed_visits, float)), 4)
    return (L + A + C + E).astype(int)


def hospital_score(hb_low, oncology, na_low, procedure, urgent, admissions_12m, los_days):
    """HOSPITAL (Donzé 2013). Componentes faltantes (NaN) suman 0, por lo que el score queda parcial."""
    def b(x):
        return np.nan_to_num(np.asarray(x, float)).astype(bool).astype(int)
    adm = np.nan_to_num(np.asarray(admissions_12m, float))
    T = np.select([adm <= 1, adm <= 5], [0, 2], 5)
    A = 2 * (np.asarray(los_days, float) >= 5).astype(int)
    return (b(hb_low) + 2 * b(oncology) + b(na_low) + b(procedure) + b(urgent) + T + A).astype(int)

# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

class ToCategory(BaseEstimator, TransformerMixin):
    """Convierte la salida ordinal en DataFrame con categorías fijas para splits categóricos nativos."""

    def __init__(self, names, n_num):
        self.names = names
        self.n_num = n_num

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        df = pd.DataFrame(X[:, :self.n_num], columns=self.names[:self.n_num])
        dtype = pd.CategoricalDtype(categories=list(range(-1, MAX_CATEGORIES)))
        for j, c in enumerate(self.names[self.n_num:]):
            df[c] = pd.Categorical(np.rint(X[:, self.n_num + j]).astype(int), dtype=dtype)
        return df


def ordinal_preprocessor(num, cat):
    return ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='Missing')),
                          ('encode', OrdinalEncoder(handle_unknown='use_encoded_value',
                                                    unknown_value=-1, min_frequency=MIN_FREQUENCY,
                                                    max_categories=MAX_CATEGORIES))]), cat)])


XGB_BASE = dict(n_estimators=400, learning_rate=0.05, subsample=0.8, reg_lambda=5.0,
                min_child_weight=10, tree_method='hist', enable_categorical=True,
                max_cat_to_onehot=1, eval_metric='aucpr', n_jobs=1)
LGBM_BASE = dict(n_estimators=400, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.8, reg_lambda=5.0, min_child_samples=100, cat_smooth=50,
                 min_data_per_group=200, n_jobs=1, verbose=-1)
HGB_BASE = dict(max_iter=180, learning_rate=0.07, max_leaf_nodes=15, min_samples_leaf=40,
                l2_regularization=5.0, early_stopping=False)


def make_pipeline(kind, strategy, num, cat, params=None, pos_weight=1.0, seed=SEED):
    """kind: LR, HGB, XGB, LGBM. strategy: none, weights, under, smote.

    El remuestreo vive dentro de un Pipeline de imblearn: se aplica solo en `fit`,
    nunca al predecir, por lo que no toca validación, test ni los folds de evaluación.
    """
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    params = dict(params or {})
    n_num, n_cat = len(num), len(cat)
    cat_idx = list(range(n_num, n_num + n_cat))
    steps = [('prep', ordinal_preprocessor(num, cat))]
    if strategy == 'under':
        steps.append(('sampler', RandomUnderSampler(sampling_strategy=SAMPLING_RATIO,
                                                    random_state=seed)))
    elif strategy == 'smote':
        steps.append(('sampler', SMOTENC(categorical_features=cat_idx, k_neighbors=5,
                                         sampling_strategy=SAMPLING_RATIO, random_state=seed)))
    elif strategy not in ('none', 'weights'):
        raise ValueError(strategy)
    weighted = strategy == 'weights'
    if kind == 'LR':
        steps.append(('encode', ColumnTransformer([
            ('num', StandardScaler(), list(range(n_num))),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_idx)])))
        model = LogisticRegression(C=params.get('C', 1.0), max_iter=2000, random_state=seed,
                                   class_weight='balanced' if weighted else None)
    else:
        steps.append(('typed', ToCategory(list(num) + list(cat), n_num)))
        if kind == 'HGB':
            model = HistGradientBoostingClassifier(**{**HGB_BASE, **params}, random_state=seed,
                                                   categorical_features='from_dtype',
                                                   class_weight='balanced' if weighted else None)
        elif kind == 'XGB':
            model = XGBClassifier(**{**XGB_BASE, **params}, random_state=seed,
                                  scale_pos_weight=pos_weight if weighted else 1.0)
        elif kind == 'LGBM':
            model = LGBMClassifier(**{**LGBM_BASE, **params}, random_state=seed,
                                   scale_pos_weight=pos_weight if weighted else 1.0)
        else:
            raise ValueError(kind)
    steps.append(('model', model))
    return ImbPipeline(steps)


def make_v1_model(kind, columns, num, cat, seed=SEED):
    """Réplica exacta de los modelos de la v1 (LR one-hot y HGB ordinal), para comparar."""
    nc = [c for c in columns if c in num]
    cc = [c for c in columns if c in cat]
    if kind == 'LR':
        prep = ColumnTransformer([
            ('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), nc),
            ('cat', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='Missing')),
                              ('encode', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                                       min_frequency=30, max_categories=64))]), cc)])
        model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=seed)
    else:
        prep = ColumnTransformer([
            ('num', SimpleImputer(strategy='median'), nc),
            ('cat', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='Missing')),
                              ('encode', OrdinalEncoder(handle_unknown='use_encoded_value',
                                                        unknown_value=-1, min_frequency=30,
                                                        max_categories=64))]), cc)])
        model = HistGradientBoostingClassifier(
            **HGB_BASE, categorical_features=list(range(len(nc), len(nc) + len(cc))),
            random_state=seed)
    return Pipeline([('prep', prep), ('model', model)])


MODEL_RANK = {'LR': 0, 'HGB': 1, 'LGBM': 2, 'XGB': 2}
STRATEGY_RANK = {'none': 0, 'weights': 1, 'under': 2, 'smote': 3}


def build_grid(xgb_grid, lgbm_grid, strategies=('none', 'weights', 'under', 'smote')):
    """Combinaciones modelo x configuración x estrategia de desbalance."""
    configs = [('LR', 'LR', {}), ('HGB', 'HGB', {})]
    configs += [('XGB', 'XGB-' + '-'.join(f'{k}{v}' for k, v in p.items()), p) for p in xgb_grid]
    configs += [('LGBM', 'LGBM-' + '-'.join(f'{k}{v}' for k, v in p.items()), p) for p in lgbm_grid]
    return [{'name': f'{label} | {s}', 'kind': kind, 'config': label, 'strategy': s, 'params': p}
            for kind, label, p in configs for s in strategies]

# ---------------------------------------------------------------------------
# Validación cruzada (agrupada repetida o temporal)
# ---------------------------------------------------------------------------

def repeated_group_folds(y, groups, n_splits=5, n_repeats=3, seed=SEED):
    """Folds estratificados y agrupados por paciente, repetidos con distintas semillas."""
    folds = []
    for r in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for k, (tr, te) in enumerate(sgkf.split(np.zeros(len(y)), np.asarray(y), np.asarray(groups))):
            folds.append((f'r{r}k{k}', tr, te))
    return folds


def _fit_eval_fold(X, y, fold_id, tr, te, combos, num, cat, seed):
    rows = []
    Xtr, ytr, Xte, yte = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
    pos_weight = float((ytr == 0).sum() / (ytr == 1).sum())
    with threadpool_limits(limits=1):
        for c in combos:
            t0 = time.perf_counter()
            m = make_pipeline(c['kind'], c['strategy'], num, cat, c['params'], pos_weight, seed)
            m.fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
            rows.append({'combo': c['name'], 'fold': fold_id, **metrics(yte, p),
                         'seconds': time.perf_counter() - t0})
    return rows


def evaluate_folds(X, y, folds, combos, num, cat, seed=SEED, n_jobs=4):
    """Ajusta cada combinación en cada fold. Paraleliza por fold; cada modelo usa 1 hilo."""
    out = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_fit_eval_fold)(X, y, fid, tr, te, combos, num, cat, seed)
        for fid, tr, te in folds)
    return pd.DataFrame([r for rows in out for r in rows])


def summarize_cv(cv, combos, metric_cols=('AP', 'AUROC', 'Brier', 'Lift@20', 'Recall@20')):
    meta = pd.DataFrame(combos).set_index('name')[['kind', 'config', 'strategy']]
    agg = cv.groupby('combo')[list(metric_cols) + ['seconds']].agg(['mean', 'std'])
    agg.columns = [f'{a}_{b}' for a, b in agg.columns]
    agg['n_folds'] = cv.groupby('combo').size()
    out = meta.join(agg, how='inner')
    out['model_rank'] = out.kind.map(MODEL_RANK)
    out['strategy_rank'] = out.strategy.map(STRATEGY_RANK)
    return out.sort_values('AP_mean', ascending=False)


def select_one_se(summary, metric='AP'):
    """Regla de un error estándar: entre las combinaciones cuya media no es inferior a
    (mejor media - EE del mejor), elige la más simple (modelo y luego estrategia)."""
    best = summary[f'{metric}_mean'].idxmax()
    se = summary.loc[best, f'{metric}_std'] / np.sqrt(summary.loc[best, 'n_folds'])
    floor = summary.loc[best, f'{metric}_mean'] - se
    cand = summary[summary[f'{metric}_mean'] >= floor].copy()
    cand['_neg'] = -cand[f'{metric}_mean']
    chosen = cand.sort_values(['model_rank', 'strategy_rank', '_neg']).index[0]
    return chosen, best, float(se), cand.drop(columns='_neg')

# ---------------------------------------------------------------------------
# Recalibración
# ---------------------------------------------------------------------------

def fit_calibrator(model, Xc, yc, method):
    """Recalibra un modelo ya entrenado (congelado) con un conjunto independiente."""
    return CalibratedClassifierCV(FrozenEstimator(model), method=method, ensemble=False).fit(Xc, yc)


def choose_calibration(model, Xc, yc, groups, methods=('sigmoid', 'isotonic'), n_splits=5,
                       seed=SEED):
    """Compara métodos por Brier con ajuste cruzado dentro del conjunto de calibración.

    Devuelve el método elegido, la tabla comparativa y las predicciones fuera de fold.
    """
    raw = model.predict_proba(Xc)[:, 1]
    oof = {m: np.zeros(len(yc)) for m in methods}
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yc = pd.Series(np.asarray(yc))
    for tr, te in sgkf.split(np.zeros(len(yc)), yc, np.asarray(groups)):
        for m in methods:
            cal = fit_calibrator(model, Xc.iloc[tr], yc.iloc[tr], m)
            oof[m][te] = cal.predict_proba(Xc.iloc[te])[:, 1]
    rows = [{'método': 'sin recalibrar', 'Brier': brier_score_loss(yc, raw),
             'media_predicha': raw.mean()}]
    rows += [{'método': m, 'Brier': brier_score_loss(yc, oof[m]), 'media_predicha': oof[m].mean()}
             for m in methods]
    table = pd.DataFrame(rows).assign(prevalencia=float(yc.mean()))
    best = table.iloc[1:].sort_values('Brier').iloc[0]['método']
    return best, table, oof

# ---------------------------------------------------------------------------
# Bootstrap agrupado por paciente
# ---------------------------------------------------------------------------

def grouped_bootstrap(y, scores, groups, n_boot=300, seed=SEED, pairs=(), q=0.20):
    """Remuestrea pacientes completos. Devuelve réplicas por modelo y diferencias pareadas."""
    y = np.asarray(y)
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    n_groups = inverse.max() + 1
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
        ix = np.repeat(np.arange(len(y)), counts[inverse])
        yy = y[ix]
        rep = {}
        for name, s in scores.items():
            pp = np.asarray(s)[ix]
            tm = top_metrics(yy, pp, q)
            ap = average_precision_score(yy, pp)
            rep[name] = {'AUROC': roc_auc_score(yy, pp), 'AP': ap,
                         'AP_minus_prevalence': ap - yy.mean(), 'Brier': brier_score_loss(yy, pp),
                         'Lift@20': tm['lift'], 'Recall@20': tm['recall'],
                         'Precision@20': tm['precision']}
            rows += [{'b': b, 'modelo': name, 'metrica': k, 'valor': v} for k, v in rep[name].items()]
        for a, c in pairs:
            rows += [{'b': b, 'modelo': f'{a} − {c}', 'metrica': k, 'valor': rep[a][k] - rep[c][k]}
                     for k in ('AUROC', 'AP', 'Brier', 'Lift@20', 'Recall@20')]
    return pd.DataFrame(rows)


def bootstrap_ci(boot, level=0.95):
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return (boot.groupby(['modelo', 'metrica']).valor.quantile([lo, hi]).unstack()
            .rename(columns={lo: 'IC95_inferior', hi: 'IC95_superior'}))

# ---------------------------------------------------------------------------
# Registro de corridas (MLflow)
# ---------------------------------------------------------------------------

def mlflow_log_summary(summary, experiment, tracking_dir, common_params=None, tags=None):
    """Una corrida de MLflow por combinación con medias y desvíos de validación cruzada."""
    try:
        import mlflow
    except ImportError:
        print('MLflow no está instalado: se omite el registro de corridas.')
        return None
    tracking_dir = Path(tracking_dir); tracking_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f'sqlite:///{(tracking_dir / "mlflow.db").as_posix()}')
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(tracking_dir / 'artifacts').as_uri())
    mlflow.set_experiment(experiment)
    for name, row in summary.iterrows():
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({**(tags or {}), 'kind': row.kind, 'strategy': row.strategy})
            mlflow.log_params({**(common_params or {}), 'config': row.config,
                               'strategy': row.strategy})
            mlflow.log_metrics({k.replace('@', '_at_'): float(v) for k, v in row.items()
                                if (k.endswith('_mean') or k.endswith('_std')) and pd.notna(v)})
    return mlflow.get_tracking_uri()
