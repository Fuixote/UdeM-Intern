"""Phase-0 helpers for moving Step1c-vs-PyEPO validation onto KEP graphs.

This module is a small preflight layer.  It reuses the existing Step1c KEP
record loader/solver path and the shared SPO+ algebra, but it does not train a
PyEPO model or run a trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from surrogate_experiment_results.SPO_validation.step1c_vs_pyepo import (
    validation_core as shortest_path_core,
)
from surrogate_experiment_results.Step1c import spoplus_core
from surrogate_experiment_results.Step1c import step1c_common

shortest_path_core.ensure_local_pyepo_on_path()
from pyepo import EPO  # noqa: E402
from pyepo.model.opt import optModel  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "dataset" / "processed" / "step1_noisy_linear_sigma010"
DEFAULT_JSON_PATH = Path(__file__).resolve().parent / "latest_phase0.json"
DEFAULT_SMALL_SETTING_JSON_PATH = (
    Path(__file__).resolve().parent / "latest_small_setting.json"
)
DEFAULT_FULL_TRAJECTORY_JSON_PATH = (
    Path(__file__).resolve().parent / "latest_full_trajectory.json"
)
DEFAULT_FULL_TRAJECTORY_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "phase2_results" / "train_size=5"
)
DEFAULT_SPLIT_PATH = REPO_ROOT / "results" / "step1b_splits" / "master_split_seed=42.json"

PHASE2_LOSS_CURVE_FIELDS = [
    "epoch",
    "theta_1",
    "theta_2",
    "theta_norm",
    "train_spoplus_loss",
    "validation_spoplus_loss",
    "train_normalized_spoplus_loss",
    "validation_normalized_spoplus_loss",
    "train_decision_gap",
    "validation_decision_gap",
    "train_normalized_decision_gap",
    "validation_normalized_decision_gap",
    "train_y_adv_oracle_equal_rate",
    "validation_y_adv_oracle_equal_rate",
    "train_y_pred_oracle_equal_rate",
    "validation_y_pred_oracle_equal_rate",
]
PHASE2_DIFF_THRESHOLDS = {
    "theta_1": 1e-6,
    "theta_2": 1e-6,
    "theta_norm": 1e-6,
    "train_spoplus_loss": 1e-5,
    "validation_spoplus_loss": 1e-5,
    "train_normalized_spoplus_loss": 1e-5,
    "validation_normalized_spoplus_loss": 1e-5,
    "train_decision_gap": 1e-9,
    "validation_decision_gap": 1e-9,
    "train_normalized_decision_gap": 1e-12,
    "validation_normalized_decision_gap": 1e-12,
    "train_y_adv_oracle_equal_rate": 1e-12,
    "validation_y_adv_oracle_equal_rate": 1e-12,
    "train_y_pred_oracle_equal_rate": 1e-12,
    "validation_y_pred_oracle_equal_rate": 1e-12,
}
LR_SUMMARY_FIELDS = [
    "method",
    "theta_1",
    "theta_2",
    "theta_norm",
    "train_mse_loss",
    "validation_mse_loss",
    "train_decision_gap",
    "validation_decision_gap",
    "train_normalized_gap",
    "validation_normalized_gap",
]
LR_DIFF_THRESHOLDS = {
    "theta": 1e-12,
    "train_prediction": 1e-12,
    "validation_prediction": 1e-12,
    "train_mse": 1e-12,
    "validation_mse": 1e-12,
    "train_decision_gap": 1e-9,
    "validation_decision_gap": 1e-9,
    "train_normalized_gap": 1e-12,
    "validation_normalized_gap": 1e-12,
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    value: float
    threshold: float

    @property
    def passed(self) -> bool:
        return self.value <= self.threshold

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "passed": self.passed,
        }


def _graph_sort_key(path: Path):
    match = re.fullmatch(r"G-(\d+)\.json", path.name)
    if match:
        return (0, int(match.group(1)))
    return (1, path.name)


def select_graph_paths(data_dir: Path, graph_count: int) -> list[Path]:
    data_dir = Path(data_dir)
    if graph_count <= 0:
        raise ValueError("graph_count must be positive")
    paths = sorted(data_dir.glob("G-*.json"), key=_graph_sort_key)
    if len(paths) < graph_count:
        raise FileNotFoundError(
            "Requested {} KEP graphs under {}, found {}".format(
                graph_count,
                data_dir,
                len(paths),
            )
        )
    return paths[:graph_count]


def compute_ols_theta(records) -> np.ndarray:
    records = list(records)
    if not records:
        raise ValueError("Cannot compute OLS theta from an empty record list")
    x_pool = np.vstack([np.asarray(record["X"], dtype=float) for record in records])
    w_pool = np.concatenate(
        [np.asarray(record["w_true"], dtype=float) for record in records]
    )
    theta, *_ = np.linalg.lstsq(x_pool, w_pool, rcond=None)
    return np.asarray(theta, dtype=float)


def make_probe_theta(records, mode="random", seed=42) -> np.ndarray:
    if mode == "random":
        rng = np.random.RandomState(seed)
        return rng.uniform(0.5, 3.5, size=2).astype(float)
    if mode == "ols":
        return compute_ols_theta(records)
    raise ValueError("Unsupported theta mode: {}".format(mode))


class KepCostMinOptModel(optModel):
    """PyEPO-compatible cost-min view over one reward-max KEP record."""

    def __init__(self, record, step1a, env):
        self.record = record
        self.step1a = step1a
        self.env = env
        self.modelSense = EPO.MINIMIZE
        self.x = list(range(len(record["w_true"])))
        self._cost = np.zeros(len(self.x), dtype=float)

    def setObj(self, c):
        c = np.asarray(c, dtype=float)
        if c.shape[0] != len(self.x):
            raise ValueError("Size of cost vector cannot match KEP edge count.")
        self._cost = c.copy()

    def solve(self):
        solution = np.asarray(
            self.step1a.solve_once(-self._cost, self.record["graph"], self.env),
            dtype=float,
        )
        return solution, float(np.dot(self._cost, solution))


def _record_cost_arrays(record):
    w_true = np.asarray(record["w_true"], dtype=float)
    y_optimal = np.asarray(record["y_optimal"], dtype=float)
    true_cost = -w_true
    true_obj = float(np.dot(true_cost, y_optimal))
    return true_cost, y_optimal, true_obj


def _decision_metrics_from_solution(record, solution, normalizer_epsilon=1e-9):
    w_true = np.asarray(record["w_true"], dtype=float)
    y_optimal = np.asarray(record["y_optimal"], dtype=float)
    optimal_obj = float(np.dot(w_true, y_optimal))
    achieved_obj = float(np.dot(w_true, solution))
    gap = optimal_obj - achieved_obj
    return gap, gap / (abs(optimal_obj) + normalizer_epsilon)


def phase1_level0_checks(records, step1a, env) -> list[CheckResult]:
    solution_diffs = []
    objective_diffs = []
    for record in records:
        model = KepCostMinOptModel(record, step1a=step1a, env=env)
        true_cost, true_sol, true_obj = _record_cost_arrays(record)
        model.setObj(true_cost)
        sol, obj = model.solve()
        solution_diffs.append(shortest_path_core.max_abs_diff(sol, true_sol))
        objective_diffs.append(abs(obj - true_obj))
    return [
        CheckResult("level0_solution_max_abs_diff", max(solution_diffs), 0.0),
        CheckResult("level0_objective_max_abs_diff", max(objective_diffs), 1e-7),
    ]


def phase1_level1_checks(records, step1a, env) -> list[CheckResult]:
    reference_theta = compute_ols_theta(records)
    step1c_theta = compute_ols_theta(records)

    prediction_diffs = []
    mse_diffs = []
    gap_diffs = []
    normalized_gap_diffs = []
    for record in records:
        x = np.asarray(record["X"], dtype=float)
        w_true = np.asarray(record["w_true"], dtype=float)
        reference_pred = x @ reference_theta
        step1c_pred = x @ step1c_theta
        prediction_diffs.append(
            shortest_path_core.max_abs_diff(reference_pred, step1c_pred)
        )
        mse_diffs.append(
            abs(
                float(np.mean((reference_pred - w_true) ** 2))
                - float(np.mean((step1c_pred - w_true) ** 2))
            )
        )

        model = KepCostMinOptModel(record, step1a=step1a, env=env)
        model.setObj(-reference_pred)
        reference_sol, _ = model.solve()
        step1c_sol = np.asarray(
            step1a.solve_once(step1c_pred, record["graph"], env),
            dtype=float,
        )
        ref_gap, ref_norm_gap = _decision_metrics_from_solution(record, reference_sol)
        step_gap, step_norm_gap = _decision_metrics_from_solution(record, step1c_sol)
        gap_diffs.append(abs(ref_gap - step_gap))
        normalized_gap_diffs.append(abs(ref_norm_gap - step_norm_gap))

    return [
        CheckResult(
            "level1_ols_theta_max_abs_diff",
            shortest_path_core.max_abs_diff(reference_theta, step1c_theta),
            1e-12,
        ),
        CheckResult(
            "level1_prediction_max_abs_diff",
            max(prediction_diffs),
            1e-12,
        ),
        CheckResult("level1_mse_diff", max(mse_diffs), 1e-12),
        CheckResult("level1_decision_gap_diff", max(gap_diffs), 1e-9),
        CheckResult("level1_normalized_gap_diff", max(normalized_gap_diffs), 1e-12),
    ]


def compute_pyepo_lr_theta(records) -> np.ndarray:
    """PyEPO-style 2-stage LR reference for variable-size KEP records.

    KEP records do not have a fixed cost-vector dimension across graphs, so the
    shortest-path `sklearnPred(LinearRegression())` interface is not directly
    usable.  The equivalent no-bias edge-level linear reward fit is the same
    OLS problem used in Phase 1.
    """

    return compute_ols_theta(records)


def compute_step1c_lr_theta(records) -> np.ndarray:
    """Step1c two-feature LR reward fit under the same no-bias model."""

    return compute_ols_theta(records)


def _lr_predictions(records, theta) -> np.ndarray:
    records = list(records)
    if not records:
        raise ValueError("records must be non-empty")
    theta = np.asarray(theta, dtype=float)
    return np.concatenate(
        [np.asarray(record["X"], dtype=float) @ theta for record in records]
    )


def lr_diagnostics_for_theta(records, theta, step1a, env) -> dict:
    records = list(records)
    if not records:
        raise ValueError("records must be non-empty")
    theta = np.asarray(theta, dtype=float)
    mse_losses = []
    gaps = []
    normalized_gaps = []
    for record in records:
        x = np.asarray(record["X"], dtype=float)
        w_true = np.asarray(record["w_true"], dtype=float)
        w_hat = x @ theta
        mse_losses.append(float(np.mean((w_hat - w_true) ** 2)))
        y_pred = np.asarray(step1a.solve_once(w_hat, record["graph"], env), dtype=float)
        gap, normalized_gap = _decision_metrics_from_solution(record, y_pred)
        gaps.append(gap)
        normalized_gaps.append(normalized_gap)
    return {
        "mse_loss": float(np.mean(mse_losses)),
        "decision_gap": float(np.mean(gaps)),
        "normalized_gap": float(np.mean(normalized_gaps)),
    }


def lr_summary_row(method, theta, train_diagnostics, validation_diagnostics) -> dict:
    theta = np.asarray(theta, dtype=float)
    return {
        "method": method,
        "theta_1": float(theta[0]),
        "theta_2": float(theta[1]),
        "theta_norm": float(np.linalg.norm(theta)),
        "train_mse_loss": float(train_diagnostics["mse_loss"]),
        "validation_mse_loss": float(validation_diagnostics["mse_loss"]),
        "train_decision_gap": float(train_diagnostics["decision_gap"]),
        "validation_decision_gap": float(validation_diagnostics["decision_gap"]),
        "train_normalized_gap": float(train_diagnostics["normalized_gap"]),
        "validation_normalized_gap": float(validation_diagnostics["normalized_gap"]),
    }


def compare_lr_summaries(
    pyepo_theta,
    step1c_theta,
    pyepo_predictions,
    step1c_predictions,
    pyepo_summary,
    step1c_summary,
):
    checks = [
        CheckResult(
            "lr_theta_max_abs_diff",
            shortest_path_core.max_abs_diff(pyepo_theta, step1c_theta),
            LR_DIFF_THRESHOLDS["theta"],
        ),
        CheckResult(
            "lr_train_prediction_max_abs_diff",
            shortest_path_core.max_abs_diff(
                pyepo_predictions["train"],
                step1c_predictions["train"],
            ),
            LR_DIFF_THRESHOLDS["train_prediction"],
        ),
        CheckResult(
            "lr_validation_prediction_max_abs_diff",
            shortest_path_core.max_abs_diff(
                pyepo_predictions["validation"],
                step1c_predictions["validation"],
            ),
            LR_DIFF_THRESHOLDS["validation_prediction"],
        ),
    ]
    field_names = {
        "train_mse_loss": "train_mse",
        "validation_mse_loss": "validation_mse",
        "train_decision_gap": "train_decision_gap",
        "validation_decision_gap": "validation_decision_gap",
        "train_normalized_gap": "train_normalized_gap",
        "validation_normalized_gap": "validation_normalized_gap",
    }
    for field, threshold_key in field_names.items():
        checks.append(
            CheckResult(
                "lr_{}_diff".format(threshold_key),
                abs(float(pyepo_summary[field]) - float(step1c_summary[field])),
                LR_DIFF_THRESHOLDS[threshold_key],
            )
        )
    return {
        "results": [item.as_dict() for item in checks],
        "passed": all(item.passed for item in checks),
    }


def run_paired_lr_bridge(train_records, validation_records, step1a, env) -> dict:
    train_records = list(train_records)
    validation_records = list(validation_records)
    if not train_records:
        raise ValueError("train_records must be non-empty")
    if not validation_records:
        raise ValueError("validation_records must be non-empty")

    pyepo_theta = compute_pyepo_lr_theta(train_records)
    step1c_theta = compute_step1c_lr_theta(train_records)
    pyepo_train = lr_diagnostics_for_theta(
        train_records,
        pyepo_theta,
        step1a=step1a,
        env=env,
    )
    pyepo_validation = lr_diagnostics_for_theta(
        validation_records,
        pyepo_theta,
        step1a=step1a,
        env=env,
    )
    step1c_train = lr_diagnostics_for_theta(
        train_records,
        step1c_theta,
        step1a=step1a,
        env=env,
    )
    step1c_validation = lr_diagnostics_for_theta(
        validation_records,
        step1c_theta,
        step1a=step1a,
        env=env,
    )

    pyepo_summary = lr_summary_row(
        "pyepo_lr",
        pyepo_theta,
        pyepo_train,
        pyepo_validation,
    )
    step1c_summary = lr_summary_row(
        "step1c_lr",
        step1c_theta,
        step1c_train,
        step1c_validation,
    )
    summary = compare_lr_summaries(
        pyepo_theta,
        step1c_theta,
        {
            "train": _lr_predictions(train_records, pyepo_theta),
            "validation": _lr_predictions(validation_records, pyepo_theta),
        },
        {
            "train": _lr_predictions(train_records, step1c_theta),
            "validation": _lr_predictions(validation_records, step1c_theta),
        },
        pyepo_summary,
        step1c_summary,
    )
    return {
        "phase": "step2b_lr_bridge",
        "purpose": "Step2b degree PyEPO-style LR vs Step1c LR bridge",
        "pyepo_theta": np.asarray(pyepo_theta, dtype=float).tolist(),
        "step1c_theta": np.asarray(step1c_theta, dtype=float).tolist(),
        "pyepo_summary": pyepo_summary,
        "step1c_summary": step1c_summary,
        "results": summary["results"],
        "passed": bool(summary["passed"]),
    }


def phase1_spoplus_record_check(record, theta, step1a, env, sgd_lr):
    """Compare PyEPO SPOPlus with Step1c reward-max algebra for one KEP graph."""

    import torch
    import pyepo

    theta = np.asarray(theta, dtype=float)
    x = np.asarray(record["X"], dtype=float)
    w_true = np.asarray(record["w_true"], dtype=float)
    y_optimal = np.asarray(record["y_optimal"], dtype=float)

    w_hat = x @ theta
    true_cost, true_sol, true_obj = _record_cost_arrays(record)
    model = KepCostMinOptModel(record, step1a=step1a, env=env)

    pred_cost_t = torch.tensor(
        (-w_hat).reshape(1, -1).astype(np.float32),
        dtype=torch.float32,
        requires_grad=True,
    )
    true_cost_t = torch.tensor(
        true_cost.reshape(1, -1).astype(np.float32),
        dtype=torch.float32,
    )
    true_sol_t = torch.tensor(
        true_sol.reshape(1, -1).astype(np.float32),
        dtype=torch.float32,
    )
    true_obj_t = torch.tensor([[true_obj]], dtype=torch.float32)

    spop = pyepo.func.SPOPlus(model, processes=1)
    pyepo_loss_vec = spop(pred_cost_t, true_cost_t, true_sol_t, true_obj_t)
    pyepo_loss = pyepo_loss_vec.mean()
    pyepo_loss.backward()

    pyepo_loss_value = float(pyepo_loss_vec.detach().cpu().numpy().reshape(-1)[0])
    pyepo_grad_pred_cost = pred_cost_t.grad.detach().cpu().numpy().reshape(-1)
    pyepo_grad_pred_reward = -pyepo_grad_pred_cost
    pyepo_grad_theta = x.T @ pyepo_grad_pred_reward

    shifted_w = 2.0 * w_hat - w_true
    y_adv = np.asarray(step1a.solve_once(shifted_w, record["graph"], env), dtype=float)
    step1c_loss = float(
        spoplus_core.reward_max_spoplus_loss(w_hat, w_true, y_optimal, y_adv)
    )
    step1c_grad_pred = np.asarray(
        spoplus_core.reward_max_prediction_gradient(y_optimal, y_adv),
        dtype=float,
    )
    step1c_grad_theta = x.T @ step1c_grad_pred

    pyepo_next_theta = theta - float(sgd_lr) * pyepo_grad_theta
    step1c_next_theta = theta - float(sgd_lr) * step1c_grad_theta

    return {
        "graph": record.get("filename", ""),
        "level2_forward_loss_abs_diff": abs(pyepo_loss_value - step1c_loss),
        "level3_grad_pred_max_abs_diff": shortest_path_core.max_abs_diff(
            pyepo_grad_pred_reward,
            step1c_grad_pred,
        ),
        "level3_grad_theta_max_abs_diff": shortest_path_core.max_abs_diff(
            pyepo_grad_theta,
            step1c_grad_theta,
        ),
        "level4_sgd_theta_update_max_abs_diff": shortest_path_core.max_abs_diff(
            pyepo_next_theta,
            step1c_next_theta,
        ),
    }


def phase1_level2_3_4_checks(records, theta, step1a, env, sgd_lr) -> tuple[list[dict], list[CheckResult]]:
    record_checks = [
        phase1_spoplus_record_check(
            record,
            theta,
            step1a=step1a,
            env=env,
            sgd_lr=sgd_lr,
        )
        for record in records
    ]
    checks = [
        CheckResult(
            "level2_forward_loss_max_abs_diff",
            max(item["level2_forward_loss_abs_diff"] for item in record_checks),
            1e-5,
        ),
        CheckResult(
            "level3_grad_pred_max_abs_diff",
            max(item["level3_grad_pred_max_abs_diff"] for item in record_checks),
            1e-6,
        ),
        CheckResult(
            "level3_grad_theta_max_abs_diff",
            max(item["level3_grad_theta_max_abs_diff"] for item in record_checks),
            1e-6,
        ),
        CheckResult(
            "level4_sgd_theta_update_max_abs_diff",
            max(item["level4_sgd_theta_update_max_abs_diff"] for item in record_checks),
            1e-6,
        ),
    ]
    return record_checks, checks


def step1c_spoplus_loss_and_grad_for_record(record, theta, step1a, env):
    """Step1c reward-max SPO+ loss/gradient with injectable KEP oracle."""

    theta = np.asarray(theta, dtype=float)
    x = np.asarray(record["X"], dtype=float)
    w_true = np.asarray(record["w_true"], dtype=float)
    y_optimal = np.asarray(record["y_optimal"], dtype=float)

    w_hat = x @ theta
    shifted_w = 2.0 * w_hat - w_true
    y_adv = np.asarray(step1a.solve_once(shifted_w, record["graph"], env), dtype=float)
    loss = float(
        spoplus_core.reward_max_spoplus_loss(w_hat, w_true, y_optimal, y_adv)
    )
    grad_theta = x.T @ spoplus_core.reward_max_prediction_gradient(y_optimal, y_adv)
    return loss, np.asarray(grad_theta, dtype=float)


def pyepo_spoplus_loss_and_grad_for_record(record, theta, step1a, env):
    """PyEPO cost-min SPOPlus reference, adapted to reward-max KEP weights."""

    import torch
    import pyepo

    theta = np.asarray(theta, dtype=float)
    x = np.asarray(record["X"], dtype=float)
    w_true = np.asarray(record["w_true"], dtype=float)
    w_hat = x @ theta
    true_cost, true_sol, true_obj = _record_cost_arrays(record)
    model = KepCostMinOptModel(record, step1a=step1a, env=env)

    pred_cost_t = torch.tensor(
        (-w_hat).reshape(1, -1).astype(np.float32),
        dtype=torch.float32,
        requires_grad=True,
    )
    true_cost_t = torch.tensor(
        true_cost.reshape(1, -1).astype(np.float32),
        dtype=torch.float32,
    )
    true_sol_t = torch.tensor(
        true_sol.reshape(1, -1).astype(np.float32),
        dtype=torch.float32,
    )
    true_obj_t = torch.tensor([[true_obj]], dtype=torch.float32)

    spop = pyepo.func.SPOPlus(model, processes=1)
    loss_vec = spop(pred_cost_t, true_cost_t, true_sol_t, true_obj_t)
    loss = loss_vec.mean()
    loss.backward()

    pyepo_loss_value = float(loss_vec.detach().cpu().numpy().reshape(-1)[0])
    grad_pred_cost = pred_cost_t.grad.detach().cpu().numpy().reshape(-1)
    grad_pred_reward = -grad_pred_cost
    grad_theta = x.T @ grad_pred_reward
    return pyepo_loss_value, np.asarray(grad_theta, dtype=float)


def phase2_average_spoplus_loss_and_grad(records, theta, step1a, env, source):
    if source not in {"pyepo", "step1c"}:
        raise ValueError("source must be 'pyepo' or 'step1c'")
    losses = []
    grad = np.zeros(2, dtype=float)
    for record in records:
        if source == "pyepo":
            loss, record_grad = pyepo_spoplus_loss_and_grad_for_record(
                record,
                theta,
                step1a=step1a,
                env=env,
            )
        else:
            loss, record_grad = step1c_spoplus_loss_and_grad_for_record(
                record,
                theta,
                step1a=step1a,
                env=env,
            )
        losses.append(float(loss))
        grad += record_grad
    return float(np.mean(losses)), grad / len(records)


def run_phase2_spoplus_trajectory(
    records,
    theta_init,
    n_epochs,
    lr,
    step1a,
    env,
    source,
):
    records = list(records)
    if not records:
        raise ValueError("records must be non-empty")
    if n_epochs < 0:
        raise ValueError("n_epochs must be non-negative")
    opt = step1c_common.Adam(lr)
    theta = np.asarray(theta_init, dtype=float).copy()
    trajectory = [theta.copy()]
    for _ in range(n_epochs):
        _, grad = phase2_average_spoplus_loss_and_grad(
            records,
            theta,
            step1a=step1a,
            env=env,
            source=source,
        )
        theta = opt.step(theta, grad)
        trajectory.append(theta.copy())
    return np.asarray(trajectory, dtype=float)


def phase2_spoplus_diagnostics_for_theta(theta, records, step1a, env, loss_source):
    if loss_source not in {"pyepo", "step1c"}:
        raise ValueError("loss_source must be 'pyepo' or 'step1c'")
    theta = np.asarray(theta, dtype=float)
    rows = []
    for record in records:
        x = np.asarray(record["X"], dtype=float)
        w_true = np.asarray(record["w_true"], dtype=float)
        y_optimal = np.asarray(record["y_optimal"], dtype=float)
        w_hat = x @ theta
        y_pred = np.asarray(step1a.solve_once(w_hat, record["graph"], env), dtype=float)
        shifted_w = 2.0 * w_hat - w_true
        y_adv = np.asarray(step1a.solve_once(shifted_w, record["graph"], env), dtype=float)

        if loss_source == "pyepo":
            spoplus_loss, _ = pyepo_spoplus_loss_and_grad_for_record(
                record,
                theta,
                step1a=step1a,
                env=env,
            )
        else:
            spoplus_loss = float(
                spoplus_core.reward_max_spoplus_loss(
                    w_hat,
                    w_true,
                    y_optimal,
                    y_adv,
                )
            )

        optimal_obj = float(np.dot(w_true, y_optimal))
        achieved_obj = float(np.dot(w_true, y_pred))
        denominator = abs(optimal_obj) + 1e-9
        rows.append(
            {
                "decision_gap": optimal_obj - achieved_obj,
                "normalized_decision_gap": (optimal_obj - achieved_obj) / denominator,
                "spoplus_loss": float(spoplus_loss),
                "normalized_spoplus_loss": float(spoplus_loss) / denominator,
                "y_adv_oracle_equal": bool(np.allclose(y_adv, y_optimal)),
                "y_pred_oracle_equal": bool(np.allclose(y_pred, y_optimal)),
            }
        )
    return {
        "decision_gap": float(np.mean([row["decision_gap"] for row in rows])),
        "normalized_decision_gap": float(
            np.mean([row["normalized_decision_gap"] for row in rows])
        ),
        "spoplus_loss": float(np.mean([row["spoplus_loss"] for row in rows])),
        "normalized_spoplus_loss": float(
            np.mean([row["normalized_spoplus_loss"] for row in rows])
        ),
        "y_adv_oracle_equal_rate": float(
            np.mean([row["y_adv_oracle_equal"] for row in rows])
        ),
        "y_pred_oracle_equal_rate": float(
            np.mean([row["y_pred_oracle_equal"] for row in rows])
        ),
    }


def phase2_evaluate_trajectory_diagnostics(
    trajectory,
    records,
    step1a,
    env,
    indices,
    loss_source,
):
    return [
        phase2_spoplus_diagnostics_for_theta(
            trajectory[int(idx)],
            records,
            step1a=step1a,
            env=env,
            loss_source=loss_source,
        )
        for idx in indices
    ]


def phase2_loss_curve_rows(
    trajectory_subset,
    train_diagnostics,
    validation_diagnostics,
    epoch_indices,
):
    rows = []
    for row_idx, epoch in enumerate(epoch_indices):
        theta = np.asarray(trajectory_subset[row_idx], dtype=float)
        train_row = train_diagnostics[row_idx]
        validation_row = validation_diagnostics[row_idx]
        rows.append(
            {
                "epoch": int(epoch),
                "theta_1": float(theta[0]),
                "theta_2": float(theta[1]),
                "theta_norm": float(np.linalg.norm(theta)),
                "train_spoplus_loss": float(train_row["spoplus_loss"]),
                "validation_spoplus_loss": float(
                    validation_row["spoplus_loss"]
                ),
                "train_normalized_spoplus_loss": float(
                    train_row["normalized_spoplus_loss"]
                ),
                "validation_normalized_spoplus_loss": float(
                    validation_row["normalized_spoplus_loss"]
                ),
                "train_decision_gap": float(train_row["decision_gap"]),
                "validation_decision_gap": float(
                    validation_row["decision_gap"]
                ),
                "train_normalized_decision_gap": float(
                    train_row["normalized_decision_gap"]
                ),
                "validation_normalized_decision_gap": float(
                    validation_row["normalized_decision_gap"]
                ),
                "train_y_adv_oracle_equal_rate": float(
                    train_row["y_adv_oracle_equal_rate"]
                ),
                "validation_y_adv_oracle_equal_rate": float(
                    validation_row["y_adv_oracle_equal_rate"]
                ),
                "train_y_pred_oracle_equal_rate": float(
                    train_row["y_pred_oracle_equal_rate"]
                ),
                "validation_y_pred_oracle_equal_rate": float(
                    validation_row["y_pred_oracle_equal_rate"]
                ),
            }
        )
    return rows


def phase2_compare_loss_curves(pyepo_rows, step1c_rows):
    pyepo_rows = list(pyepo_rows)
    step1c_rows = list(step1c_rows)
    if len(pyepo_rows) != len(step1c_rows):
        raise ValueError(
            "PyEPO and Step1c row counts differ: {} vs {}".format(
                len(pyepo_rows),
                len(step1c_rows),
            )
        )
    if not pyepo_rows:
        raise ValueError("loss curve rows must be non-empty")

    epoch_diffs = [
        abs(int(pyepo["epoch"]) - int(step1c["epoch"]))
        for pyepo, step1c in zip(pyepo_rows, step1c_rows)
    ]
    checks = [
        CheckResult("phase2_epoch_max_abs_diff", max(epoch_diffs), 0.0)
    ]
    for field, threshold in PHASE2_DIFF_THRESHOLDS.items():
        diffs = [
            abs(float(pyepo[field]) - float(step1c[field]))
            for pyepo, step1c in zip(pyepo_rows, step1c_rows)
        ]
        checks.append(
            CheckResult(
                "phase2_{}_max_abs_diff".format(field),
                max(diffs),
                threshold,
            )
        )
    return {
        "results": [item.as_dict() for item in checks],
        "passed": all(item.passed for item in checks),
    }


def run_phase2_paired_spoplus_trajectory(
    train_records,
    validation_records,
    theta_init,
    n_epochs,
    lr,
    metric_stride,
    step1a,
    env,
):
    train_records = list(train_records)
    validation_records = list(validation_records)
    if not train_records:
        raise ValueError("train_records must be non-empty")
    if not validation_records:
        raise ValueError("validation_records must be non-empty")

    pyepo_trajectory = run_phase2_spoplus_trajectory(
        train_records,
        theta_init,
        n_epochs=n_epochs,
        lr=lr,
        step1a=step1a,
        env=env,
        source="pyepo",
    )
    step1c_trajectory = run_phase2_spoplus_trajectory(
        train_records,
        theta_init,
        n_epochs=n_epochs,
        lr=lr,
        step1a=step1a,
        env=env,
        source="step1c",
    )

    epoch_indices = step1c_common.trajectory_epoch_indices(
        len(pyepo_trajectory),
        metric_stride,
    )
    pyepo_train = phase2_evaluate_trajectory_diagnostics(
        pyepo_trajectory,
        train_records,
        step1a=step1a,
        env=env,
        indices=epoch_indices,
        loss_source="pyepo",
    )
    pyepo_validation = phase2_evaluate_trajectory_diagnostics(
        pyepo_trajectory,
        validation_records,
        step1a=step1a,
        env=env,
        indices=epoch_indices,
        loss_source="pyepo",
    )
    step1c_train = phase2_evaluate_trajectory_diagnostics(
        step1c_trajectory,
        train_records,
        step1a=step1a,
        env=env,
        indices=epoch_indices,
        loss_source="step1c",
    )
    step1c_validation = phase2_evaluate_trajectory_diagnostics(
        step1c_trajectory,
        validation_records,
        step1a=step1a,
        env=env,
        indices=epoch_indices,
        loss_source="step1c",
    )

    pyepo_rows = phase2_loss_curve_rows(
        pyepo_trajectory[epoch_indices],
        pyepo_train,
        pyepo_validation,
        epoch_indices=epoch_indices,
    )
    step1c_rows = phase2_loss_curve_rows(
        step1c_trajectory[epoch_indices],
        step1c_train,
        step1c_validation,
        epoch_indices=epoch_indices,
    )
    summary = phase2_compare_loss_curves(pyepo_rows, step1c_rows)
    return {
        "pyepo_trajectory": pyepo_trajectory,
        "step1c_trajectory": step1c_trajectory,
        "epoch_indices": epoch_indices,
        "pyepo_rows": pyepo_rows,
        "step1c_rows": step1c_rows,
        "summary": summary,
    }


def sign_adapter_check(record, theta, step1a, env) -> dict:
    """Compare reward-max KEP SPO+ with the cost-min sign adapter.

    The same KEP solutions are used on both sides.  This isolates the algebraic
    sign conversion from PyEPO wrapper and tie-breaking concerns.
    """

    theta = np.asarray(theta, dtype=float)
    x = np.asarray(record["X"], dtype=float)
    w_true = np.asarray(record["w_true"], dtype=float)
    y_optimal = np.asarray(record["y_optimal"], dtype=float)

    w_hat = x @ theta
    shifted_w = 2.0 * w_hat - w_true
    y_adv = np.asarray(step1a.solve_once(shifted_w, record["graph"], env), dtype=float)

    reward_loss = float(
        spoplus_core.reward_max_spoplus_loss(w_hat, w_true, y_optimal, y_adv)
    )
    cost_adapter_loss = float(
        spoplus_core.cost_min_spoplus_loss(-w_hat, -w_true, y_optimal, y_adv)
    )

    reward_grad_pred = np.asarray(
        spoplus_core.reward_max_prediction_gradient(y_optimal, y_adv),
        dtype=float,
    )
    cost_adapter_grad_pred = -np.asarray(
        spoplus_core.cost_min_prediction_gradient(y_optimal, y_adv),
        dtype=float,
    )

    reward_grad_theta = x.T @ reward_grad_pred
    cost_adapter_grad_theta = x.T @ cost_adapter_grad_pred

    true_obj = float(np.dot(w_true, y_optimal))
    y_optimal_again = np.asarray(
        step1a.solve_once(w_true, record["graph"], env),
        dtype=float,
    )
    true_obj_again = float(np.dot(w_true, y_optimal_again))

    return {
        "graph": record.get("filename", ""),
        "num_edges": int(w_true.shape[0]),
        "theta_dim": int(theta.shape[0]),
        "true_objective_abs_diff": abs(true_obj - true_obj_again),
        "loss_abs_diff": abs(reward_loss - cost_adapter_loss),
        "grad_pred_max_abs_diff": shortest_path_core.max_abs_diff(
            reward_grad_pred,
            cost_adapter_grad_pred,
        ),
        "grad_theta_max_abs_diff": shortest_path_core.max_abs_diff(
            reward_grad_theta,
            cost_adapter_grad_theta,
        ),
        "reward_loss": reward_loss,
        "shifted_weight_min": float(np.min(shifted_w)),
        "shifted_weight_max": float(np.max(shifted_w)),
        "optimal_objective": true_obj,
    }


def step1c_code_path_check(record, theta, step1a, env) -> dict:
    """Compare direct Step1c SPO+ output with the local algebra check."""

    algebra = sign_adapter_check(record, theta, step1a=step1a, env=env)
    step1c_loss, step1c_grad = step1c_common.spo_plus_loss_and_grad(
        record,
        theta=theta,
        env=env,
    )

    x = np.asarray(record["X"], dtype=float)
    w_true = np.asarray(record["w_true"], dtype=float)
    y_optimal = np.asarray(record["y_optimal"], dtype=float)
    w_hat = x @ np.asarray(theta, dtype=float)
    shifted_w = 2.0 * w_hat - w_true
    y_adv = np.asarray(step1a.solve_once(shifted_w, record["graph"], env), dtype=float)
    expected_grad = x.T @ spoplus_core.reward_max_prediction_gradient(
        y_optimal,
        y_adv,
    )

    algebra.update(
        {
            "step1c_loss_abs_diff": abs(float(step1c_loss) - algebra["reward_loss"]),
            "step1c_grad_theta_max_abs_diff": shortest_path_core.max_abs_diff(
                step1c_grad,
                expected_grad,
            ),
        }
    )
    return algebra


def summarize_phase0(record_checks) -> list[CheckResult]:
    record_checks = list(record_checks)
    if not record_checks:
        raise ValueError("record_checks must be non-empty")
    return [
        CheckResult(
            "phase0_true_objective_max_abs_diff",
            max(item["true_objective_abs_diff"] for item in record_checks),
            1e-7,
        ),
        CheckResult(
            "phase0_reward_cost_loss_max_abs_diff",
            max(item["loss_abs_diff"] for item in record_checks),
            1e-9,
        ),
        CheckResult(
            "phase0_reward_cost_grad_pred_max_abs_diff",
            max(item["grad_pred_max_abs_diff"] for item in record_checks),
            1e-9,
        ),
        CheckResult(
            "phase0_reward_cost_grad_theta_max_abs_diff",
            max(item["grad_theta_max_abs_diff"] for item in record_checks),
            1e-9,
        ),
        CheckResult(
            "phase0_step1c_loss_max_abs_diff",
            max(item["step1c_loss_abs_diff"] for item in record_checks),
            1e-9,
        ),
        CheckResult(
            "phase0_step1c_grad_theta_max_abs_diff",
            max(item["step1c_grad_theta_max_abs_diff"] for item in record_checks),
            1e-9,
        ),
    ]


def build_payload(data_dir, graph_paths, theta, record_checks, checks) -> dict:
    return {
        "phase": "phase0",
        "purpose": "KEP SPO+ adapter preflight",
        "data_dir": str(data_dir),
        "graphs": [str(path) for path in graph_paths],
        "theta": np.asarray(theta, dtype=float).tolist(),
        "record_checks": list(record_checks),
        "results": [item.as_dict() for item in checks],
        "passed": all(item.passed for item in checks),
    }


def build_phase1_payload(
    data_dir,
    graph_paths,
    theta,
    level2_3_4_record_checks,
    checks,
) -> dict:
    return {
        "phase": "phase1",
        "purpose": "KEP small validation correctness",
        "data_dir": str(data_dir),
        "graphs": [str(path) for path in graph_paths],
        "theta": np.asarray(theta, dtype=float).tolist(),
        "level2_3_4_record_checks": list(level2_3_4_record_checks),
        "results": [item.as_dict() for item in checks],
        "passed": all(item.passed for item in checks),
    }
