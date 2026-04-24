import pickle


def _get_model_params(model):
    """Extract constructor params from any BLS/ARBN model."""
    keys = ["feature_times", "enhance_times", "n_classes",
            "mapping_function", "enhance_function", "feature_size",
            "reg", "use_sparse"]
    params = {k: getattr(model, k, None) for k in keys}
    if hasattr(model, "sig"):
        params["sig"] = model.sig
    if hasattr(model, "cls_num_list"):
        params["cls_num_list"] = model.cls_num_list
    if hasattr(model, "adaptive_reg"):
        params["adaptive_reg"] = model.adaptive_reg
    if hasattr(model, "class_weight_beta"):
        params["class_weight_beta"] = model.class_weight_beta
    return params


def load_model(MODEL, filename):
    """Load a BLS/ARBN model from pickle file."""
    with open(filename, "rb") as f:
        data = pickle.load(f)

    model = MODEL(**data["model_params"])
    model.mapping_generator = data["mapping_generator"]
    model.enhance_generator = data["enhance_generator"]
    model.W = data["W"]
    model.is_fitted = data["is_fitted"]

    if "onehot_encoder" in data:
        model.onehot_encoder = data["onehot_encoder"]
    if "_mapping_nodes" in data and data["_mapping_nodes"] is not None:
        model._mapping_nodes = data["_mapping_nodes"]

    return model


def store_model(model, filename):
    """Store a BLS/ARBN model to pickle file."""
    data = {
        "model_params": _get_model_params(model),
        "mapping_generator": model.mapping_generator,
        "enhance_generator": model.enhance_generator,
        "W": model.W,
        "is_fitted": model.is_fitted,
    }
    if hasattr(model, "onehot_encoder"):
        data["onehot_encoder"] = model.onehot_encoder
    if hasattr(model, "_mapping_nodes") and model._mapping_nodes is not None:
        data["_mapping_nodes"] = model._mapping_nodes

    with open(filename, "wb") as f:
        pickle.dump(data, f)
