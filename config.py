import json
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent / 'real_datasets' / 'xrf'
MODELS_DIR = CONFIGS_DIR / 'models'


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with *override* merged on top of *base* (recursive for nested dicts)."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_global_defaults() -> dict:
    """Load the shared top-level defaults from models/_defaults.json."""
    defaults_path = MODELS_DIR / '_defaults.json'
    return _load_json(defaults_path) if defaults_path.exists() else {}


def load_model_defaults(model_name: str) -> dict:
    """Load per-model parameter defaults from models/{model_name}.json."""
    model_path = MODELS_DIR / f'{model_name}.json'
    return _load_json(model_path) if model_path.exists() else {}


def load_dataset_config(dataset_name: str) -> dict:
    """Load the raw dataset JSON config by name (no defaults merged)."""
    config_path = CONFIGS_DIR / f'{dataset_name}.json'
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config found for dataset '{dataset_name}' at {config_path}"
        )
    return _load_json(config_path)


def build_effective_config(dataset_name: str, model_name: str | None = None) -> dict:
    """Build the fully-merged config for a dataset (and optionally a specific model).

    Merge order (later wins):
      1. Global defaults  (models/_defaults.json)
      2. Per-model defaults injected into model_params[model_name]  (models/{model}.json)
      3. Dataset-specific config  ({dataset_name}.json)

    The returned config always has a populated ``model_params`` section for every
    compatible model, combining the model defaults with any dataset-level overrides.
    """
    global_def = load_global_defaults()
    dataset_cfg = load_dataset_config(dataset_name)

    # Start from global defaults, then merge the dataset config on top
    effective = _deep_merge(global_def, dataset_cfg)

    # Build model_params: for each compatible model, merge model defaults with dataset overrides
    compatible_models = effective.get('compatible_models', [])
    models_to_expand = [model_name] if model_name else compatible_models
    merged_model_params = {}
    for mdl in models_to_expand:
        if mdl not in compatible_models:
            continue
        mdl_defaults = load_model_defaults(mdl)
        mdl_overrides = effective.get('model_params', {}).get(mdl, {})
        merged_model_params[mdl] = _deep_merge(mdl_defaults, mdl_overrides)

    if merged_model_params:
        # Preserve any model_params entries for models we didn't expand
        existing = effective.get('model_params', {})
        effective['model_params'] = {**existing, **merged_model_params}

    return effective


def list_available_datasets() -> list[str]:
    """List all datasets that have JSON configs."""
    return sorted(p.stem for p in CONFIGS_DIR.glob('*.json'))


def get_compatible_datasets(model_name: str) -> list[str]:
    """List datasets compatible with a given model."""
    result = []
    for ds in list_available_datasets():
        config = load_dataset_config(ds)
        if model_name in config.get('compatible_models', []):
            result.append(ds)
    return result
