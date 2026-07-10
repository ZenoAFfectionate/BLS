"""Model serialization via pickle for BLS / ARBN.

Uses sklearn's BaseEstimator.get_params() for robust parameter extraction,
avoiding the maintenance burden of a manually-curated parameter list.
"""

import pickle


def _get_model_params(model):
    """Extract constructor parameters via sklearn's get_params.

    All BLS/ARBN models inherit from sklearn.base.BaseEstimator whose
    get_params() automatically collects every __init__ keyword argument
    (excluding **kwargs).  This is robust against future parameter
    additions — no manual list maintenance required.

    We complement get_params() with a few extra attributes that are not
    constructor arguments but must survive serialization.
    """
    params = model.get_params(deep=False)

    # ---- per-model extras (not constructor args) ----
    for key in (
        "sig",
        "cls_num_list",
        "adaptive_reg",
        "class_weight_beta",
    ):
        if hasattr(model, key) and key not in params:
            params[key] = getattr(model, key)

    # Merge any extra kwargs stored during init (e.g. sig for BLS)
    if hasattr(model, "_extra_kwargs") and model._extra_kwargs:
        for k, v in model._extra_kwargs.items():
            if k not in params:
                params[k] = v

    return params


def load_model(MODEL, filename):
    """Load a BLS / ARBN model from a pickle file."""
    with open(filename, "rb") as f:
        data = pickle.load(f)

    model = MODEL(**data["model_params"])
    model.mapping_generator = data["mapping_generator"]
    model.enhance_generator = data["enhance_generator"]
    model.W = data["W"]
    model.is_fitted = data["is_fitted"]

    if "_mapping_nodes" in data and data["_mapping_nodes"] is not None:
        model._mapping_nodes = data["_mapping_nodes"]

    return model


def store_model(model, filename):
    """Store a BLS / ARBN model to a pickle file."""
    data = {
        "model_params": _get_model_params(model),
        "mapping_generator": model.mapping_generator,
        "enhance_generator": model.enhance_generator,
        "W": model.W,
        "is_fitted": model.is_fitted,
    }
    if hasattr(model, "_mapping_nodes") and model._mapping_nodes is not None:
        data["_mapping_nodes"] = model._mapping_nodes

    with open(filename, "wb") as f:
        pickle.dump(data, f)
