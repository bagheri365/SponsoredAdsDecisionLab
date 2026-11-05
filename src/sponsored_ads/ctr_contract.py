"""M05A versioned LightGBM training/serving feature contract.

Only complete M03A feature rows are supported. This does not synthesize
missing candidate or shopper features from retrieval catalog records.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .ctr_features import STATIC_CATEGORICAL, NUMERIC

VERSION = 1
UNKNOWN_CATEGORY = -1


def encode_category(mapping: dict[str, int], value: str, *, fit: bool = False) -> int:
    """Share the exact train-fitted contiguous encoding with inference."""
    if not isinstance(value, str):
        raise ValueError('Categorical values must be strings')
    if fit and value not in mapping:
        mapping[value] = len(mapping)
    return mapping.get(value, UNKNOWN_CATEGORY)


def make_contract(features, category_maps):
    features = list(features)
    if len(features) != len(set(features)) or not features:
        raise ValueError('Feature names must be unique and nonempty')
    if any(name not in STATIC_CATEGORICAL and name not in NUMERIC for name in features):
        raise ValueError('Unsupported feature name')
    expected = {name for name in features if name in STATIC_CATEGORICAL}
    if set(category_maps) != expected:
        raise ValueError('Categorical mapping keys do not match features')
    maps = {name: dict(category_maps[name]) for name in features if name in expected}
    for mapping in maps.values():
        if any(not isinstance(k, str) or type(v) is not int for k, v in mapping.items()):
            raise ValueError('Invalid categorical mapping')
        if set(mapping.values()) != set(range(len(mapping))):
            raise ValueError('Categorical IDs must be contiguous from zero')
    return {'version': VERSION, 'model_type': 'lightgbm', 'features': features,
            'categorical_features': [name for name in features if name in expected],
            'category_maps': maps, 'unknown_category': UNKNOWN_CATEGORY,
            'numeric_transform': 'float32_direct',
            'missing_field_policy': 'error',
            'feature_source': 'M03A_complete_feature_row'}


def validate_contract(contract):
    if not isinstance(contract, dict) or contract.get('version') != VERSION:
        raise ValueError('Unsupported CTR contract version')
    if contract.get('model_type') != 'lightgbm' or contract.get('unknown_category') != -1:
        raise ValueError('Unsupported CTR encoding')
    if contract.get('numeric_transform') != 'float32_direct' or contract.get('missing_field_policy') != 'error':
        raise ValueError('Unsupported numeric/missing-value policy')
    if contract.get('feature_source') != 'M03A_complete_feature_row':
        raise ValueError('Unsupported feature source')
    validated = make_contract(contract['features'], contract['category_maps'])
    if validated != contract:
        raise ValueError('Unexpected CTR contract fields or values')
    return validated


def transform_rows(rows, contract):
    """Encode complete feature rows in training feature order; no label needed."""
    spec = validate_contract(contract)
    matrix = []
    for row in rows:
        values = []
        for name in spec['features']:
            if name not in row:
                raise ValueError(f'Missing required feature: {name}')
            if name in spec['category_maps']:
                values.append(encode_category(spec['category_maps'][name], row[name]))
            else:
                try:
                    value = float(row[name])
                except (ValueError, TypeError) as exc:
                    raise ValueError(f'Invalid numeric feature: {name}') from exc
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f'{name} must be finite and nonnegative')
                values.append(value)
        matrix.append(values)
    if not matrix:
        raise ValueError('Cannot transform an empty batch')
    result = np.asarray(matrix, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Feature values overflow float32')
    return result


def save_contract(contract, path):
    spec = validate_contract(contract)
    path = Path(path)
    path.write_text(json.dumps(spec, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path):
    return validate_contract(json.loads(Path(path).read_text(encoding='utf-8')))


def load_lightgbm_bundle(model_path, contract_path):
    """Require an explicit matching contract before serving a LightGBM model."""
    import lightgbm as lgb
    spec = load_contract(contract_path)
    model = lgb.Booster(model_file=str(model_path))
    if model.num_feature() != len(spec['features']):
        raise ValueError('Model/contract feature count mismatch')
    model_names = model.feature_name()
    if model_names != spec['features']:
        raise ValueError('Model/contract feature order mismatch')
    return model, spec


def predict_lightgbm(model, contract, rows):
    return model.predict(transform_rows(rows, contract))
