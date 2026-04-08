import os
import torch

from model.model_structure import (
    EDGE_RAW_DIM,
    NODE_FEATURE_DIM,
    KidneyEdgePredictor,
    build_tabular_regression_model,
    infer_tabular_model_family_from_state_dict,
)


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
    return build_tabular_regression_model(
        model_family=model_family,
        node_dim=config.get("NODE_DIM", NODE_FEATURE_DIM),
        edge_dim=config.get("EDGE_RAW_DIM", EDGE_RAW_DIM),
        hidden_dim=config.get("HIDDEN_DIM", 256),
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
        config = checkpoint.get("config", {})
    else:
        state_dict = checkpoint
        config = {}

    model_type = infer_model_type(summary_content, state_dict)
    model = build_model_from_checkpoint(model_type, config)
    model.load_state_dict(state_dict)
    model.expected_node_dim = config.get("NODE_DIM", NODE_FEATURE_DIM)
    model.expected_edge_raw_dim = config.get("EDGE_RAW_DIM", EDGE_RAW_DIM)
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
        edge_features = torch.cat([x[src], x[dst], edge_attr_for_model], dim=-1)
        return model(edge_features).numpy()
