"""Shared Warcraft Level-2 validation utilities.

The Level-2 scripts intentionally mirror the PyEPO Warcraft notebook defaults
while keeping the command-line entrypoints small and consistently named.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR / "warcraft_shortest_path_oneskin"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "plot_results"

OUTPUT_FIELDS = (
    "method",
    "grid_size",
    "epochs",
    "batch_size",
    "learning_rate",
    "seed",
    "train_limit",
    "test_limit",
    "data_root",
    "output_dir",
    "device",
    "final_normalized_regret",
    "final_avg_regret",
    "final_avg_relative_regret",
    "final_path_accuracy",
    "final_optimality_ratio",
    "loss_steps",
    "regret_points",
)


@dataclass
class Level2Config:
    data_root: Path = field(default_factory=lambda: DEFAULT_DATA_ROOT)
    output_root: Path = field(default_factory=lambda: DEFAULT_OUTPUT_ROOT)
    grid_size: int = 12
    batch_size: int = 70
    epochs: int = 50
    learning_rate: float = 5e-4
    log_step: int = 1
    seed: int = 135
    train_limit: int | None = None
    test_limit: int | None = None
    device: str = "auto"


class WarcraftMapDataset(Dataset):
    def __init__(self, terrain_maps: np.ndarray, costs: np.ndarray, paths: np.ndarray):
        self.terrain_maps = terrain_maps
        self.costs = costs
        self.paths = paths
        self.objectives = (costs * paths).sum(axis=(1, 2)).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.costs)

    def __getitem__(self, index: int):
        image = self.terrain_maps[index].transpose(2, 0, 1) / 255.0
        return (
            torch.as_tensor(image, dtype=torch.float32).detach(),
            torch.as_tensor(self.costs[index], dtype=torch.float32).reshape(-1),
            torch.as_tensor(self.paths[index], dtype=torch.float32).reshape(-1),
            torch.as_tensor(self.objectives[index], dtype=torch.float32),
        )


class PartialResNet(nn.Module):
    def __init__(self, grid_size: int):
        super().__init__()
        from torchvision.models import resnet18

        resnet = resnet18(weights=None)
        self.conv1 = resnet.conv1
        self.bn = resnet.bn1
        self.relu = resnet.relu
        self.maxpool1 = resnet.maxpool
        self.block = resnet.layer1
        self.conv2 = nn.Conv2d(
            64, 1, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.maxpool2 = nn.AdaptiveMaxPool2d((grid_size, grid_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.bn(h)
        h = self.relu(h)
        h = self.maxpool1(h)
        h = self.block(h)
        h = self.conv2(h)
        out = self.maxpool2(h)
        out = torch.squeeze(out, 1)
        return out.reshape(out.shape[0], -1)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required Warcraft data file is missing: {path}")
    return path


def _limit_array(values: np.ndarray, limit: int | None) -> np.ndarray:
    if limit is None:
        return values
    if limit <= 0:
        raise ValueError("limits must be positive when provided")
    return values[:limit]


def load_warcraft_arrays(
    data_root: Path, grid_size: int, train_limit: int | None = None, test_limit: int | None = None
) -> Dict[str, np.ndarray]:
    split_dir = Path(data_root) / f"{grid_size}x{grid_size}"
    arrays = {
        "train_maps": np.load(_require_file(split_dir / "train_maps.npy")),
        "test_maps": np.load(_require_file(split_dir / "test_maps.npy")),
        "train_costs": np.load(_require_file(split_dir / "train_vertex_weights.npy")),
        "test_costs": np.load(_require_file(split_dir / "test_vertex_weights.npy")),
        "train_paths": np.load(_require_file(split_dir / "train_shortest_paths.npy")),
        "test_paths": np.load(_require_file(split_dir / "test_shortest_paths.npy")),
    }
    arrays["train_maps"] = _limit_array(arrays["train_maps"], train_limit)
    arrays["train_costs"] = _limit_array(arrays["train_costs"], train_limit)
    arrays["train_paths"] = _limit_array(arrays["train_paths"], train_limit)
    arrays["test_maps"] = _limit_array(arrays["test_maps"], test_limit)
    arrays["test_costs"] = _limit_array(arrays["test_costs"], test_limit)
    arrays["test_paths"] = _limit_array(arrays["test_paths"], test_limit)
    return arrays


def make_dataloaders(config: Level2Config) -> Tuple[DataLoader, DataLoader]:
    arrays = load_warcraft_arrays(
        config.data_root,
        config.grid_size,
        train_limit=config.train_limit,
        test_limit=config.test_limit,
    )
    train_dataset = WarcraftMapDataset(
        arrays["train_maps"], arrays["train_costs"], arrays["train_paths"]
    )
    test_dataset = WarcraftMapDataset(
        arrays["test_maps"], arrays["test_costs"], arrays["test_paths"]
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, test_loader


def build_warcraft_optmodel(grid_size: int):
    import gurobipy as gp
    from gurobipy import GRB
    from pyepo.model.grb.grbmodel import optGrbModel

    class WarcraftShortestPathModel(optGrbModel):
        def __init__(self, grid: Tuple[int, int]):
            self.grid = grid
            self.nodes, self.edges, self.nodes_map = self._get_edges()
            super().__init__()

        def _node_index(self, row: int, col: int) -> int:
            return row * self.grid[1] + col

        def _get_edges(self):
            nodes = []
            edges = []
            nodes_map = {}
            for row in range(self.grid[0]):
                for col in range(self.grid[1]):
                    node = self._node_index(row, col)
                    nodes_map[node] = (row, col)
                    nodes.append(node)
                    if row != 0:
                        edges.append((node, self._node_index(row - 1, col)))
                        if col != self.grid[1] - 1:
                            edges.append((node, self._node_index(row - 1, col + 1)))
                    if col != self.grid[1] - 1:
                        edges.append((node, self._node_index(row, col + 1)))
                        if row != self.grid[0] - 1:
                            edges.append((node, self._node_index(row + 1, col + 1)))
                    if row != self.grid[0] - 1:
                        edges.append((node, self._node_index(row + 1, col)))
                        if col != 0:
                            edges.append((node, self._node_index(row + 1, col - 1)))
                    if col != 0:
                        edges.append((node, self._node_index(row, col - 1)))
                        if row != 0:
                            edges.append((node, self._node_index(row - 1, col - 1)))
            return nodes, edges, nodes_map

        def _getModel(self):
            model = gp.Model("warcraft_shortest_path")
            x = model.addVars(self.edges, ub=1, name="x")
            model.modelSense = GRB.MINIMIZE
            for row in range(self.grid[0]):
                for col in range(self.grid[1]):
                    node = self._node_index(row, col)
                    expr = 0
                    for edge in self.edges:
                        if node == edge[1]:
                            expr += x[edge]
                        elif node == edge[0]:
                            expr -= x[edge]
                    if row == 0 and col == 0:
                        model.addConstr(expr == -1)
                    elif row == self.grid[0] - 1 and col == self.grid[1] - 1:
                        model.addConstr(expr == 1)
                    else:
                        model.addConstr(expr == 0)
            return model, x

        def setObj(self, c):
            c = np.asarray(c, dtype=float).reshape(self.grid)
            obj = c[0, 0] + gp.quicksum(
                c[self.nodes_map[target]] * self.x[source, target]
                for source, target in self.x
            )
            self._model.setObjective(obj)

        def solve(self):
            self._model.update()
            self._model.optimize()
            sol = np.zeros(self.grid, dtype=float)
            for source, target in self.edges:
                if abs(1.0 - self.x[source, target].x) < 1e-3:
                    sol[self.nodes_map[source]] = 1.0
                    sol[self.nodes_map[target]] = 1.0
            return sol.reshape(-1), float(self._model.objVal)

    return WarcraftShortestPathModel((grid_size, grid_size))


def solve_batch(costs: torch.Tensor, optmodel) -> Tuple[torch.Tensor, torch.Tensor]:
    device = costs.device
    cost_np = costs.detach().cpu().numpy()
    solutions: List[np.ndarray] = []
    objectives: List[float] = []
    for cost in cost_np:
        optmodel.setObj(cost)
        solution, objective = optmodel.solve()
        solutions.append(solution)
        objectives.append(objective)
    return (
        torch.as_tensor(np.asarray(solutions, dtype=np.float32), device=device),
        torch.as_tensor(objectives, dtype=torch.float32, device=device),
    )


class OurSPOPlusFunction(Function):
    @staticmethod
    def forward(ctx, pred_cost, true_cost, true_sol, true_obj, optmodel):
        cp = pred_cost.detach()
        c = true_cost.detach()
        w = true_sol.detach()
        z = true_obj.detach()
        shifted_sol, shifted_obj = solve_batch(2.0 * cp - c, optmodel)
        loss = -shifted_obj + 2.0 * torch.einsum("bi,bi->b", cp, w) - z.squeeze(dim=-1)
        ctx.save_for_backward(w, shifted_sol)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        true_sol, shifted_sol = ctx.saved_tensors
        grad = 2.0 * (true_sol - shifted_sol)
        return grad_output.unsqueeze(1) * grad, None, None, None, None


class OurSPOPlus(nn.Module):
    def __init__(self, optmodel):
        super().__init__()
        self.optmodel = optmodel

    def forward(self, pred_cost, true_cost, true_sol, true_obj):
        return OurSPOPlusFunction.apply(
            pred_cost, true_cost, true_sol, true_obj, self.optmodel
        )


def build_pyepo_spoplus(optmodel):
    import pyepo

    return pyepo.func.SPOPlus(optmodel, processes=1, reduction="none")


def build_our_spoplus(optmodel):
    return OurSPOPlus(optmodel)


def evaluate_model(model: nn.Module, optmodel, dataloader: DataLoader) -> Dict[str, float]:
    model.eval()
    rows = {"Regret": [], "Relative Regret": [], "Accuracy": [], "Optimal": []}
    device = next(model.parameters()).device
    with torch.no_grad():
        for x, c, w, z in dataloader:
            x = x.to(device)
            c_np = c.numpy()
            w_np = w.numpy()
            z_np = z.numpy()
            cp = model(x).detach().cpu().numpy()
            for idx in range(cp.shape[0]):
                optmodel.setObj(cp[idx])
                pred_sol, _ = optmodel.solve()
                achieved_obj = float(np.dot(pred_sol, c_np[idx]))
                true_obj = float(z_np[idx, 0])
                regret = achieved_obj - true_obj
                rows["Regret"].append(regret)
                rows["Relative Regret"].append(regret / true_obj)
                rows["Accuracy"].append(float((np.abs(pred_sol - w_np[idx]) < 0.5).mean()))
                rows["Optimal"].append(float(abs(regret) < 1e-5))
    model.train()
    return {
        "avg_regret": float(np.mean(rows["Regret"])),
        "avg_relative_regret": float(np.mean(rows["Relative Regret"])),
        "path_accuracy": float(np.mean(rows["Accuracy"])),
        "optimality_ratio": float(np.mean(rows["Optimal"])),
    }


def normalized_regret(model: nn.Module, optmodel, dataloader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    opt_sum = 0.0
    device = next(model.parameters()).device
    with torch.no_grad():
        for x, c, _, z in dataloader:
            x = x.to(device)
            cp = model(x).detach().cpu().numpy()
            c_np = c.numpy()
            z_np = z.numpy().reshape(-1)
            for idx in range(cp.shape[0]):
                optmodel.setObj(cp[idx])
                pred_sol, _ = optmodel.solve()
                total_loss += float(np.dot(pred_sol, c_np[idx]) - z_np[idx])
                opt_sum += abs(float(z_np[idx]))
    model.train()
    return total_loss / (opt_sum + 1e-10)


def train_warcraft_spoplus(
    config: Level2Config,
    method: str,
    loss_builder: Callable,
    output_subdir: str,
) -> Mapping[str, object]:
    set_seeds(config.seed)
    device = resolve_device(config.device)
    train_loader, test_loader = make_dataloaders(config)
    optmodel = build_warcraft_optmodel(config.grid_size)
    model = PartialResNet(config.grid_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_module = loss_builder(optmodel)

    loss_log: List[Dict[str, float]] = []
    regret_log: List[Dict[str, float]] = [
        {"epoch": 0, "normalized_regret": normalized_regret(model, optmodel, test_loader)}
    ]
    started_at = time.time()
    for epoch in range(config.epochs):
        model.train()
        for x, c, w, z in train_loader:
            x = x.to(device)
            c = c.to(device)
            w = w.to(device)
            z = z.to(device)
            pred_cost = model(x)
            loss = loss_module(pred_cost, c, w, z).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_log.append({"epoch": epoch + 1, "loss": float(loss.item())})
        if epoch + 1 in {int(config.epochs * 0.6), int(config.epochs * 0.8)}:
            for group in optimizer.param_groups:
                group["lr"] /= 10.0
        if (epoch + 1) % config.log_step == 0:
            regret_log.append(
                {
                    "epoch": epoch + 1,
                    "normalized_regret": normalized_regret(model, optmodel, test_loader),
                }
            )

    final_eval = evaluate_model(model, optmodel, test_loader)
    final_regret = normalized_regret(model, optmodel, test_loader)
    output_dir = Path(config.output_root) / output_subdir
    write_outputs(
        output_dir=output_dir,
        method=method,
        config=config,
        final_regret=final_regret,
        final_eval=final_eval,
        loss_log=loss_log,
        regret_log=regret_log,
        model=model,
        elapsed_seconds=time.time() - started_at,
    )
    return {
        "method": method,
        "output_dir": str(output_dir),
        "final_normalized_regret": final_regret,
        **final_eval,
    }


def write_outputs(
    output_dir: Path,
    method: str,
    config: Level2Config,
    final_regret: float,
    final_eval: Mapping[str, float],
    loss_log: Iterable[Mapping[str, float]],
    regret_log: Iterable[Mapping[str, float]],
    model: nn.Module,
    elapsed_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_rows = list(loss_log)
    regret_rows = list(regret_log)
    summary = {
        "method": method,
        "grid_size": config.grid_size,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "train_limit": config.train_limit,
        "test_limit": config.test_limit,
        "data_root": str(config.data_root),
        "output_dir": str(output_dir),
        "device": config.device,
        "final_normalized_regret": final_regret,
        "final_avg_regret": final_eval["avg_regret"],
        "final_avg_relative_regret": final_eval["avg_relative_regret"],
        "final_path_accuracy": final_eval["path_accuracy"],
        "final_optimality_ratio": final_eval["optimality_ratio"],
        "loss_steps": len(loss_rows),
        "regret_points": len(regret_rows),
        "elapsed_seconds": elapsed_seconds,
        "config": {key: str(value) for key, value in asdict(config).items()},
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    _write_csv(output_dir / "loss_log.csv", loss_rows, ["epoch", "loss"])
    _write_csv(output_dir / "regret_log.csv", regret_rows, ["epoch", "normalized_regret"])
    torch.save(model.state_dict(), output_dir / "model_final.pt")


def _write_csv(path: Path, rows: List[Mapping[str, float]], fields: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=70)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--log-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=135)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser


def config_from_args(args: argparse.Namespace) -> Level2Config:
    return Level2Config(
        data_root=args.data_root,
        output_root=args.output_root,
        grid_size=args.grid_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        log_step=args.log_step,
        seed=args.seed,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        device=args.device,
    )
