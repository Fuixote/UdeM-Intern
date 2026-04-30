import os
import torch

from model.model_structure import (
    EDGE_RAW_DIM,
    FAILURE_CONTEXT_DIM,
    LR_SMALL_FEATURE_DIM,
    NODE_FEATURE_DIM,
    KidneyEdgePredictor,
    build_tabular_regression_model,
    infer_tabular_model_family_from_state_dict,
)


def normalize_feature_mode(feature_mode):
    mode = (feature_mode or "full").strip().lower()
    if mode not in {"full", "utility_cpra", "failure_context", "lr_small"}:
        raise ValueError(f"Unsupported tabular feature mode: {feature_mode}")
    return mode


def default_feature_mode_for_family(model_family, feature_mode=None):
    if feature_mode is not None:
        return normalize_feature_mode(feature_mode)
    return "utility_cpra" if model_family == "lr" else "full"


def infer_feature_mode_from_input_dim(input_dim):
    input_dim = int(input_dim)
    if input_dim == 2:
        return "utility_cpra"
    if input_dim == LR_SMALL_FEATURE_DIM:
        return "lr_small"
    if input_dim == FAILURE_CONTEXT_DIM:
        return "failure_context"
    return "full"


def tabular_input_dim(feature_mode):
    mode = normalize_feature_mode(feature_mode)
    if mode == "utility_cpra":
        return 2
    if mode == "lr_small":
        return LR_SMALL_FEATURE_DIM
    if mode == "failure_context":
        return FAILURE_CONTEXT_DIM
    return NODE_FEATURE_DIM * 2 + EDGE_RAW_DIM


def infer_model_type(summary_content, state_dict):
    lowered_summary = summary_content.lower()
    if "GNN" in summary_content:
        return "GNN"
    if "linear regression" in lowered_summary or "linearregressionbaseline" in lowered_summary:
        return "LinearRegression"
    if "Regression" in summary_content or "MLP" in summary_content:
        return "Regression"
    if any(key.startswith("conv1.") or key.startswith("edge_encoder.") for key in state_dict):
        return "GNN"
    return "LinearRegression" if infer_tabular_model_family_from_state_dict(state_dict) == "lr" else "Regression"


def build_model_from_checkpoint(model_type, config):
    if model_type == "GNN":
        return KidneyEdgePredictor(
            node_feature_dim=config.get("NODE_DIM", NODE_FEATURE_DIM),
            edge_raw_dim=config.get("EDGE_RAW_DIM", EDGE_RAW_DIM),
            hidden_dim=config.get("HIDDEN_DIM", 64),
        )
    model_family = config.get("MODEL_FAMILY")
    if model_family is None:
        model_family = "lr" if model_type == "LinearRegression" else "mlp"
    feature_mode = default_feature_mode_for_family(model_family, config.get("FEATURE_MODE"))
    input_dim = int(config.get("INPUT_DIM", tabular_input_dim(feature_mode)))
    return build_tabular_regression_model(
        model_family=model_family,
        node_dim=config.get("NODE_DIM", NODE_FEATURE_DIM),
        edge_dim=config.get("EDGE_RAW_DIM", EDGE_RAW_DIM),
        hidden_dim=config.get("HIDDEN_DIM", 256),
        input_dim=input_dim,
    )


def load_prediction_model(model_path):
    model_dir = os.path.dirname(model_path)
    summary_path = os.path.join(model_dir, "summary.txt")
    summary_content = ""
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary_content = handle.read()

    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = dict(checkpoint.get("config", {}))
    else:
        state_dict = checkpoint
        config = {}

    model_type = infer_model_type(summary_content, state_dict)
    if model_type != "GNN":
        model_family = config.get("MODEL_FAMILY")
        if model_family is None:
            model_family = "lr" if model_type == "LinearRegression" else "mlp"
            config["MODEL_FAMILY"] = model_family
        input_dim = config.get("INPUT_DIM")
        if input_dim is None:
            first_weight = state_dict.get("net.0.weight")
            if first_weight is not None and getattr(first_weight, "ndim", 0) == 2:
                input_dim = int(first_weight.shape[1])
                config["INPUT_DIM"] = input_dim
        if "FEATURE_MODE" not in config:
            if input_dim is not None:
                config["FEATURE_MODE"] = infer_feature_mode_from_input_dim(input_dim)
            else:
                config["FEATURE_MODE"] = default_feature_mode_for_family(model_family)
    model = build_model_from_checkpoint(model_type, config)
    model.load_state_dict(state_dict)
    model.expected_node_dim = config.get("NODE_DIM", NODE_FEATURE_DIM)
    model.expected_edge_raw_dim = config.get("EDGE_RAW_DIM", EDGE_RAW_DIM)
    model.expected_feature_mode = config.get("FEATURE_MODE", "full")
    model.eval()
    return model_type, model


def predict_edge_weights(graph_data, model, model_type):
    if model is None:
        return graph_data["gt_labels"]

    x = graph_data["x"]
    edge_index = graph_data["edge_index"]
    edge_attr = graph_data["edge_attr"]

    expected_node_dim = getattr(model, "expected_node_dim", x.size(-1))
    expected_edge_raw_dim = getattr(model, "expected_edge_raw_dim", edge_attr.size(-1))

    if x.size(-1) != expected_node_dim:
        raise ValueError(
            f"Node feature dimension mismatch: graph has {x.size(-1)}, "
            f"checkpoint expects {expected_node_dim}"
        )
    if edge_attr.size(-1) < expected_edge_raw_dim:
        raise ValueError(
            f"Edge feature dimension mismatch: graph has {edge_attr.size(-1)}, "
            f"checkpoint expects at least {expected_edge_raw_dim}"
        )

    edge_attr_for_model = edge_attr[:, :expected_edge_raw_dim]
    with torch.no_grad():
        if model_type == "GNN":
            return model(x, edge_index, edge_attr_for_model).numpy()
        src, dst = edge_index
        feature_mode = getattr(model, "expected_feature_mode", "full")
        if feature_mode == "utility_cpra":
            utility = edge_attr_for_model[:, :1]
            recipient_cpra = x[dst, 1:2]
            edge_features = torch.cat([utility, recipient_cpra], dim=-1)
        elif feature_mode == "lr_small":
            utility = edge_attr_for_model[:, :1]
            recipient_cpra = x[dst, 1:2]
            source_donor_age = x[src, 7:8]
            edge_features = torch.cat([utility, recipient_cpra, source_donor_age], dim=-1)
        elif feature_mode == "failure_context":
            if "edge_context" not in graph_data:
                raise ValueError(
                    "failure_context checkpoint requires graph_data['edge_context']. "
                    "Regenerate processed data with the current 1-data-processing.py."
                )
            edge_features = graph_data["edge_context"]
        else:
            edge_features = torch.cat([x[src], x[dst], edge_attr_for_model], dim=-1)
        return model(edge_features).numpy()
