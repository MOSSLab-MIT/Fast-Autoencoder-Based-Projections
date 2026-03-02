import os
import sys
import argparse
import itertools
import csv
import re
import time
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from autoencoder import ConstraintAwareAutoencoder
import data_generation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parsing
SHAPE_DIMENSIONS = {
    "hyperspherical_shell_3d": 3,
    "hyperspherical_shell_5d": 5,
    "hyperspherical_shell_10d": 10,
    "hyperspherical_shell_50d": 50,
}
def format_lambda(value):
    s = f"{value:.2f}".rstrip('0').rstrip('.')
    return s if s else "0"

def parse_capacity_config(config):
    match = re.match(r'^W(\d+)_D(\d+)$', config)
    if not match:
        raise ValueError(f"Invalid capacity config: {config}")
    return int(match.group(1)), int(match.group(2))

def parse_dim_config(config):
    return int(config.replace('D', '').strip())

def parse_cov_config(config):
    return int(config.replace('Cov_', '').strip())

def parse_num_dec_config(config):
    return int(config.split('_')[0])

def get_sol_dim(shape_name):
    return SHAPE_DIMENSIONS.get(shape_name, 2)

def get_n_samples_and_phase1_epochs(shape, default_epochs):
    if shape == "hyperspherical_shell_3d":
        return 90000, default_epochs
    if shape == "hyperspherical_shell_5d":
        return 150000, default_epochs
    if shape == "hyperspherical_shell_10d":
        return 350000, default_epochs
    if shape == "hyperspherical_shell_50d":
        return 1200000, 300
    return 60000, default_epochs

def sample_latent_points(n_samples, latent_dim, radius=0.5, device="cpu"):
    directions = torch.randn(n_samples, latent_dim, device=device)
    directions = directions / torch.norm(directions, dim=1, keepdim=True)
    u = torch.rand(n_samples, 1, device=device)
    radii = radius * torch.pow(u, 1.0 / latent_dim)
    return directions * radii

def compute_norm_params(shape, n_samples):
    _, _, X_all, feasible_mask = data_generation.generate_nonconvex_data(
        shape_name=shape, n_samples=n_samples
    )
    X_all_train, _, _, _ = train_test_split(
        X_all, feasible_mask.astype(float), test_size=0.2, random_state=42
    )
    norm_mean = torch.tensor(X_all_train.mean(axis=0, keepdims=True), dtype=torch.float32, device=device)
    norm_std = torch.tensor(X_all_train.std(axis=0, keepdims=True), dtype=torch.float32, device=device)
    eps = torch.tensor(1e-8, device=device)
    norm_std = torch.where(norm_std < eps, eps, norm_std)
    input_dim = X_all.shape[1]
    return norm_mean, norm_std, input_dim

def load_phase2_model(
    checkpoint_path,
    input_dim,
    latent_dim,
    num_decoders,
    decoder_hidden_dim=None,
    decoder_num_layers=None,
):
    if not os.path.exists(checkpoint_path):
        return None
    model = ConstraintAwareAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=64,
        num_decoders=num_decoders,
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_num_layers=decoder_num_layers,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Generating optimization problem parameters
def generate_qp_problem(n_vars=2, batch_size=1):
    A = torch.randn(batch_size, n_vars, n_vars, device=device)
    Q = torch.bmm(A.transpose(1, 2), A) + 0.01 * torch.eye(n_vars, device=device).unsqueeze(0)
    p = torch.randn(batch_size, n_vars, device=device)
    return Q, p

def generate_lp_problem(n_vars=2, batch_size=1):
    c = torch.randn(batch_size, n_vars, device=device)
    return c

def generate_distance_problem(n_vars=2, batch_size=1):
    target = torch.randn(batch_size, n_vars, device=device) * 3.0
    return target

def solve_qp_with_projection(Q, p, x_init, shape_name, max_iter=100):
    x = x_init.clone().requires_grad_(True)
    lr = 0.01
    for _ in range(max_iter):
        obj = 0.5 * (x * torch.matmul(Q, x.unsqueeze(-1)).squeeze(-1)).sum(dim=1) + (p * x).sum(dim=1)
        grad = torch.autograd.grad(obj.sum(), x)[0]
        with torch.no_grad():
            x = x - lr * grad
            x_np = x.cpu().numpy()
            if shape_name == "two_moons":
                feasible = data_generation.build_two_moons_oracle()(x_np)
            else:
                feasible = data_generation.check_feasibility(x_np, shape_name)
            if not np.all(feasible):
                X_feasible, _, _, _ = data_generation.generate_nonconvex_data(shape_name, n_samples=1000)
                for i in range(x.shape[0]):
                    if not feasible[i]:
                        distances = np.linalg.norm(X_feasible - x_np[i], axis=1)
                        x_np[i] = X_feasible[np.argmin(distances)]
                x = torch.tensor(x_np, dtype=torch.float32, device=device)
        x.requires_grad_(True)
    return x.detach()

def solve_lp_with_projection(c, x_init, shape_name, max_iter=100):
    x = x_init.clone().requires_grad_(True)
    lr = 0.01
    for _ in range(max_iter):
        obj = (c * x).sum(dim=1)
        grad = torch.autograd.grad(obj.sum(), x)[0]
        with torch.no_grad():
            x = x - lr * grad
            x_np = x.cpu().numpy()
            if shape_name == "two_moons":
                feasible = data_generation.build_two_moons_oracle()(x_np)
            else:
                feasible = data_generation.check_feasibility(x_np, shape_name)
            if not np.all(feasible):
                X_feasible, _, _, _ = data_generation.generate_nonconvex_data(shape_name, n_samples=1000)
                for i in range(x.shape[0]):
                    if not feasible[i]:
                        distances = np.linalg.norm(X_feasible - x_np[i], axis=1)
                        x_np[i] = X_feasible[np.argmin(distances)]
                x = torch.tensor(x_np, dtype=torch.float32, device=device)
        x.requires_grad_(True)
    return x.detach()

def solve_distance_with_projection(target, x_init, shape_name, max_iter=100):
    x = x_init.clone().requires_grad_(True)
    lr = 0.01
    for _ in range(max_iter):
        obj = ((x - target) ** 2).sum(dim=1)
        grad = torch.autograd.grad(obj.sum(), x)[0]
        with torch.no_grad():
            x = x - lr * grad
            x_np = x.cpu().numpy()
            if shape_name == "two_moons":
                feasible = data_generation.build_two_moons_oracle()(x_np)
            else:
                feasible = data_generation.check_feasibility(x_np, shape_name)
            if not np.all(feasible):
                X_feasible, _, _, _ = data_generation.generate_nonconvex_data(shape_name, n_samples=1000)
                for i in range(x.shape[0]):
                    if not feasible[i]:
                        distances = np.linalg.norm(X_feasible - x_np[i], axis=1)
                        x_np[i] = X_feasible[np.argmin(distances)]
                x = torch.tensor(x_np, dtype=torch.float32, device=device)
        x.requires_grad_(True)
    return x.detach()

# Base model
class ProblemSolverNN(nn.Module):
    def __init__(self, input_dim, output_dim=2, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)

def prepare_qp_input(Q, p):
    batch_size = Q.shape[0]
    n_vars = Q.shape[1]
    Q_flat = []
    tri_idx = torch.triu_indices(n_vars, n_vars)
    for i in range(batch_size):
        Q_upper = Q[i][tri_idx[0], tri_idx[1]]
        Q_flat.append(Q_upper)
    Q_flat = torch.stack(Q_flat)
    return torch.cat([Q_flat, p], dim=1)

def prepare_lp_input(c):
    return c

def prepare_distance_input(target):
    return target

def train_nn_model_posthoc(model, shape_name, objective_type, num_vars=2, num_epochs=500, batch_size=32):
    """Train NN, treating the frozen autoencoder projection as a post-hoc projection."""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        if objective_type == "qp":
            Q, p = generate_qp_problem(n_vars=num_vars, batch_size=batch_size)
            problem_params = prepare_qp_input(Q, p)
            x_init = torch.randn(batch_size, num_vars, device=device)
            x_true = solve_qp_with_projection(Q, p, x_init, shape_name, max_iter=50)
        elif objective_type == "lp":
            c = generate_lp_problem(n_vars=num_vars, batch_size=batch_size)
            problem_params = prepare_lp_input(c)
            x_init = torch.randn(batch_size, num_vars, device=device)
            x_true = solve_lp_with_projection(c, x_init, shape_name, max_iter=50)
        else:
            target = generate_distance_problem(n_vars=num_vars, batch_size=batch_size)
            problem_params = prepare_distance_input(target)
            x_init = torch.randn(batch_size, num_vars, device=device)
            x_true = solve_distance_with_projection(target, x_init, shape_name, max_iter=50)
        x_pred = model(problem_params)
        loss = nn.MSELoss()(x_pred, x_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 499 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    return model

def ae_project(x, autoencoder, norm_mean, norm_std, latent_radius=0.5):
    """Normalize, encode, project to latent ball, decode, denormalize."""
    x_norm = (x - norm_mean) / norm_std
    z = autoencoder.encode(x_norm)
    z_norm = torch.norm(z, dim=1, keepdim=True)
    z = torch.where(z_norm > latent_radius, z * (latent_radius / z_norm), z)
    x_dec = autoencoder.decode(z)
    return x_dec * norm_std + norm_mean

def train_nn_model_e2e(
    model, autoencoder, norm_mean, norm_std,
    shape_name, objective_type, num_vars=2, num_epochs=500, batch_size=32,
    latent_radius=0.5,
):
    """Train NN end-to-end through a frozen autoencoder projection."""
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad_(False)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        if objective_type == "qp":
            Q, p = generate_qp_problem(n_vars=num_vars, batch_size=batch_size)
            problem_params = prepare_qp_input(Q, p)
            x_init = torch.randn(batch_size, num_vars, device=device)
            x_true = solve_qp_with_projection(Q, p, x_init, shape_name, max_iter=50)
        elif objective_type == "lp":
            c = generate_lp_problem(n_vars=num_vars, batch_size=batch_size)
            problem_params = prepare_lp_input(c)
            x_init = torch.randn(batch_size, num_vars, device=device)
            x_true = solve_lp_with_projection(c, x_init, shape_name, max_iter=50)
        else:
            target = generate_distance_problem(n_vars=num_vars, batch_size=batch_size)
            problem_params = prepare_distance_input(target)
            x_init = torch.randn(batch_size, num_vars, device=device)
            x_true = solve_distance_with_projection(target, x_init, shape_name, max_iter=50)

        x_pred = model(problem_params)
        x_proj = ae_project(x_pred, autoencoder, norm_mean, norm_std, latent_radius)
        loss = nn.MSELoss()(x_proj, x_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 499 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    return model

_TWO_MOONS_REF_TENSOR = None
def _get_two_moons_ref_tensor():
    """Build and cache reference points for two_moons as a PyTorch tensor."""
    global _TWO_MOONS_REF_TENSOR
    if _TWO_MOONS_REF_TENSOR is None:
        from sklearn.datasets import make_moons as _make_moons
        X_ref, _ = _make_moons(n_samples=60000, noise=0.05, random_state=42)
        _TWO_MOONS_REF_TENSOR = torch.tensor(X_ref, dtype=torch.float32, device=device)
    return _TWO_MOONS_REF_TENSOR

def compute_constraint_violation_torch(x, shape_name):
    """
    Differentiable constraint violation: returns 0 for feasible points,
    positive distance from the feasible boundary for infeasible points.
    """
    if shape_name == "blob_with_bite":
        circle_radius = 2.0
        bite_center = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
        bite_center[0] = 1.0
        bite_radius = 1.0
        dist_to_circle = torch.norm(x, dim=1)
        dist_to_bite = torch.norm(x - bite_center, dim=1)
        return torch.relu(dist_to_circle - circle_radius) + torch.relu(bite_radius - dist_to_bite)

    elif shape_name == "concentric_circles":
        R_inner, R_outer = 1.0, 2.0
        dist = torch.norm(x, dim=1)
        return torch.relu(R_inner - dist) + torch.relu(dist - R_outer)

    elif shape_name == "star_shaped":
        num_points = 5
        R_outer, R_inner = 2.0, 1.0
        dist = torch.norm(x, dim=1)
        angle = torch.atan2(x[:, 1], x[:, 0])
        angle = torch.where(angle < 0, angle + 2 * np.pi, angle)
        shifted_angle = angle + np.pi / num_points
        angle_in_segment = torch.remainder(shifted_angle, 2 * np.pi / num_points)
        relative_angle = angle_in_segment / (np.pi / num_points)
        r_ref = torch.where(relative_angle <= 1,
                            R_outer - (R_outer - R_inner) * relative_angle,
                            R_inner + (R_outer - R_inner) * (relative_angle - 1))
        return torch.relu(dist - r_ref)

    elif shape_name == "two_moons":
        thresh = 3.0 * 0.05
        ref = _get_two_moons_ref_tensor().to(device=x.device, dtype=x.dtype)
        min_dists = torch.cdist(x, ref).min(dim=1).values
        return torch.relu(min_dists - thresh)

    elif shape_name == "torus":
        R_major, r_minor, tolerance = 2.0, 0.5, 0.1
        dist_xy = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + 1e-8)
        dist_from_circle = torch.abs(dist_xy - R_major)
        dist_to_torus = torch.sqrt(dist_from_circle ** 2 + x[:, 2] ** 2 + 1e-8)
        return torch.relu(torch.abs(dist_to_torus - r_minor) - tolerance)

    elif shape_name == "sphere_with_bite":
        R_main = 2.0
        bite_center = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
        bite_center[0] = 1.0
        bite_radius = 1.0
        dist_to_main = torch.norm(x, dim=1)
        dist_to_bite = torch.norm(x - bite_center, dim=1)
        return torch.relu(dist_to_main - R_main) + torch.relu(bite_radius - dist_to_bite)

    elif shape_name == "spherical_shell":
        R_inner, R_outer = 1.0, 2.0
        dist = torch.norm(x, dim=1)
        return torch.relu(R_inner - dist) + torch.relu(dist - R_outer)

    elif shape_name == "disconnected_spherical_shells":
        R_inner, R_outer = 1.0, 1.5
        center1 = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
        center1[0] = -2.0
        center2 = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
        center2[0] = 2.0
        dist1 = torch.norm(x - center1, dim=1)
        dist2 = torch.norm(x - center2, dim=1)
        v1 = torch.relu(R_inner - dist1) + torch.relu(dist1 - R_outer)
        v2 = torch.relu(R_inner - dist2) + torch.relu(dist2 - R_outer)
        return torch.min(v1, v2)

    elif shape_name.startswith("hyperspherical_shell"):
        R_inner, R_outer = 1.0, 2.0
        dist = torch.norm(x, dim=1)
        return torch.relu(R_inner - dist) + torch.relu(dist - R_outer)

    else:
        raise ValueError(f"No differentiable violation defined for shape: {shape_name}")


def train_penalty_nn_model(model, shape_name, objective_type, num_vars=2, num_epochs=500, batch_size=32, penalty_weight=10.0, two_moons_oracle=None):
    """Train a neural network with differentiable distance-based penalty for constraint violations."""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        if objective_type == "qp":
            Q, p = generate_qp_problem(n_vars=num_vars, batch_size=batch_size)
            problem_params = prepare_qp_input(Q, p)
            objective_fn = lambda x: 0.5 * (x * torch.matmul(Q, x.unsqueeze(-1)).squeeze(-1)).sum(dim=1) + (p * x).sum(dim=1)
        elif objective_type == "lp":
            c = generate_lp_problem(n_vars=num_vars, batch_size=batch_size)
            problem_params = prepare_lp_input(c)
            objective_fn = lambda x: (c * x).sum(dim=1)
        else:
            target = generate_distance_problem(n_vars=num_vars, batch_size=batch_size)
            problem_params = prepare_distance_input(target)
            objective_fn = lambda x: ((x - target) ** 2).sum(dim=1)
        
        x_pred = model(problem_params)
        obj_value = objective_fn(x_pred).mean()
        
        violation = compute_constraint_violation_torch(x_pred, shape_name)
        violation_penalty = violation.mean()
        
        loss = obj_value + penalty_weight * violation_penalty
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 499 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f} (Obj: {obj_value.item():.6f}, Penalty: {violation_penalty.item():.6f})")
    return model

def evaluate_model_feasibility(
    model,
    shape,
    norm_mean,
    norm_std,
    n_latent_samples=100000,
    latent_radius=0.5,
    two_moons_oracle=None,
):
    with torch.no_grad():
        z = sample_latent_points(n_latent_samples, model.latent_dim, radius=latent_radius, device=device)
        decoded = model.decode(z)
        decoded = decoded * norm_std + norm_mean
    decoded_np = decoded.detach().cpu().numpy()
    if shape == "two_moons" and two_moons_oracle is not None:
        is_feasible = two_moons_oracle(decoded_np)
    else:
        is_feasible = data_generation.check_feasibility(decoded_np, shape)
    return float(np.mean(is_feasible))

def plot_sampling_and_decoding(
    model,
    shape,
    norm_mean,
    norm_std,
    n_latent_samples=20000,
    latent_radius=0.5,
    two_moons_oracle=None,
    output_path=None,
    show=False,
    title_prefix="",
):
    if model.latent_dim != 2:
        print("Skipping plot: latent_dim is not 2.")
        return
    if norm_mean.shape[1] != 2:
        print("Skipping plot: input dimension is not 2.")
        return

    def add_ground_truth_outline(ax, shape_name, oracle=None, n_samples=6000, grid_size=250):
        try:
            _, _, X_all, _ = data_generation.generate_nonconvex_data(
                shape_name=shape_name, n_samples=n_samples
            )
        except Exception as exc:
            print(f"Skipping outline: failed to sample shape bounds ({exc})")
            return

        if X_all.shape[1] != 2:
            print("Skipping outline: shape is not 2D.")
            return

        min_xy = X_all.min(axis=0)
        max_xy = X_all.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1e-6)
        padding = 0.1 * span
        x_min, y_min = min_xy - padding
        x_max, y_max = max_xy + padding

        xs = np.linspace(x_min, x_max, grid_size)
        ys = np.linspace(y_min, y_max, grid_size)
        Xg, Yg = np.meshgrid(xs, ys)
        grid_points = np.column_stack([Xg.ravel(), Yg.ravel()])

        if shape_name == "two_moons" and oracle is not None:
            feasible = oracle(grid_points)
        else:
            feasible = data_generation.check_feasibility(grid_points, shape_name)

        Z = feasible.reshape(Xg.shape).astype(float)
        ax.contour(Xg, Yg, Z, levels=[0.5], colors="black", linewidths=1.0, zorder=3)

    with torch.no_grad():
        z = sample_latent_points(n_latent_samples, model.latent_dim, radius=latent_radius, device=device)
        decoded = model.decode(z)
        decoded = decoded * norm_std + norm_mean

    z_np = z.detach().cpu().numpy()
    decoded_np = decoded.detach().cpu().numpy()
    if shape == "two_moons" and two_moons_oracle is not None:
        is_feasible = two_moons_oracle(decoded_np)
    else:
        is_feasible = data_generation.check_feasibility(decoded_np, shape)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(z_np[:, 0], z_np[:, 1], s=6, alpha=0.5, color="#1f77b4")
    axes[0].add_patch(plt.Circle((0, 0), latent_radius, color="black", fill=False, linestyle="--", linewidth=1))
    axes[0].set_title("Latent samples")
    axes[0].set_aspect("equal", "box")

    feasible_mask = np.asarray(is_feasible, dtype=bool)
    axes[1].scatter(
        decoded_np[feasible_mask, 0],
        decoded_np[feasible_mask, 1],
        s=6,
        alpha=0.6,
        color="green",
        label="Feasible",
    )
    axes[1].scatter(
        decoded_np[~feasible_mask, 0],
        decoded_np[~feasible_mask, 1],
        s=6,
        alpha=0.6,
        color="red",
        label="Infeasible",
    )
    axes[1].set_title("Decoded samples")
    axes[1].set_aspect("equal", "box")
    axes[1].legend(loc="best", fontsize=8)
    add_ground_truth_outline(axes[1], shape, oracle=two_moons_oracle)

    title = "Sampling vs. decoded feasibility"
    if title_prefix:
        title = f"{title_prefix} | {title}"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved sampling plot to {output_path}")
    if show:
        plt.show()
    plt.close(fig)

def parse_phase2_filename(filename, shapes):
    name = os.path.splitext(os.path.basename(filename))[0]
    for shape in shapes:
        prefix = f"phase2_{shape}_"
        if name.startswith(prefix):
            rest = name[len(prefix):]
            parts = rest.split("_")
            if len(parts) < 7:
                raise ValueError(f"Unexpected phase2 filename format: {filename}")
            exp_type = parts[0]
            lambda_parts = parts[-5:]
            config = "_".join(parts[1:-5])
            return shape, exp_type, config, lambda_parts
    raise ValueError(f"Could not parse shape from phase2 filename: {filename}")

def build_run_tag(shapes_filter=None, exp_types_filter=None, configs_filter=None):
    parts = []
    if shapes_filter:
        parts.append("shape-" + "_".join(shapes_filter))
    if exp_types_filter:
        parts.append("exp-" + "_".join(exp_types_filter))
    if configs_filter:
        parts.append("cfg-" + "_".join(configs_filter))
    if not parts:
        return "all"
    return "__".join(parts)

def run_optimal_ablation_experiments(
    csv_path,
    models_dir,
    results_dir="results",
    num_seeds=5,
    num_problems_per_seed=300,
    train_epochs=500,
    train_batch_size=32,
    shapes_filter=None,
    exp_types_filter=None,
    configs_filter=None,
    penalty_nn_only=False,
):
    shapes = [
        "blob_with_bite", "star_shaped", "two_moons", "concentric_circles",
        "hyperspherical_shell_3d", "hyperspherical_shell_5d",
        "hyperspherical_shell_10d",
    ]
    objective_types = ["qp", "lp", "distance"]

    models_by_shape = {shape: [] for shape in shapes}
    if not penalty_nn_only:
        model_entries = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                model_entries.append(row[0].strip())

        if not model_entries:
            raise ValueError(f"No model entries found in {csv_path}")

        for fname in model_entries:
            shape, exp_type, config, _ = parse_phase2_filename(fname, shapes)
            if shapes_filter and shape not in shapes_filter:
                continue
            if exp_types_filter and exp_type not in exp_types_filter:
                continue
            if configs_filter and config not in configs_filter:
                continue
            sol_dim = get_sol_dim(shape)
            latent_dim = sol_dim
            num_decoders = 1
            decoder_hidden_dim = None
            decoder_num_layers = None
            if exp_type == "dim":
                latent_dim = parse_dim_config(config)
            elif exp_type == "capacity":
                decoder_hidden_dim, decoder_num_layers = parse_capacity_config(config)
            elif exp_type == "num_dec":
                num_decoders = parse_num_dec_config(config)
            elif exp_type == "cov":
                _ = parse_cov_config(config)

            models_by_shape[shape].append({
                "name": os.path.splitext(os.path.basename(fname))[0],
                "path": os.path.join(models_dir, fname),
                "latent_dim": latent_dim,
                "num_decoders": num_decoders,
                "decoder_hidden_dim": decoder_hidden_dim,
                "decoder_num_layers": decoder_num_layers,
            })

    all_results = {"configuration": {}, "metrics": {}, "raw_results": {}}
    filtered_shapes = shapes_filter if shapes_filter else shapes
    all_results["configuration"] = {
        "constraint_families": filtered_shapes,
        "objective_types": objective_types,
        "num_seeds": num_seeds,
        "problems_per_seed": num_problems_per_seed,
        "train_epochs": train_epochs,
        "train_batch_size": train_batch_size,
        "models_dir": models_dir,
        "csv_path": csv_path,
    }

    two_moons_oracle = data_generation.build_two_moons_oracle()
    print("Starting Optimal Ablation Experiments")
    print(f"Constraint families: {', '.join(filtered_shapes)}")
    print(f"Objective types: {', '.join(objective_types)}")
    print(f"Seeds: {num_seeds}")
    print(f"Problems per seed: {num_problems_per_seed}")
    print(f"Train epochs: {train_epochs}, batch size: {train_batch_size}")

    for shape_name in filtered_shapes:
        print(f"\n{'='*60}")
        print(f"Testing constraint family: {shape_name}")
        print(f"{'='*60}")

        sol_dim = get_sol_dim(shape_name)
        n_samples_shape, _ = get_n_samples_and_phase1_epochs(shape_name, 500)
        norm_mean, norm_std, _ = compute_norm_params(shape_name, n_samples_shape)

        baseline_model_infos = []

        for obj_type in objective_types:
            print(f"\nObjective type: {obj_type}")
            methods = []
            nn_models = {}

            if obj_type == "qp":
                nn_input_dim = sol_dim * (sol_dim + 3) // 2
            else:
                nn_input_dim = sol_dim
            
            # Train penalty_nn baseline
            print("Training penalty_nn baseline...")
            penalty_nn = ProblemSolverNN(nn_input_dim, output_dim=sol_dim, hidden_dim=128).to(device)
            penalty_nn = train_penalty_nn_model(
                penalty_nn,
                shape_name,
                obj_type,
                num_vars=sol_dim,
                num_epochs=train_epochs,
                batch_size=train_batch_size,
                penalty_weight=10.0,
                two_moons_oracle=two_moons_oracle,
            )
            nn_models["penalty_nn"] = (penalty_nn, None)
            methods.append("penalty_nn")
            
            all_model_infos = list(models_by_shape[shape_name]) + baseline_model_infos
            for info in all_model_infos:
                autoencoder = load_phase2_model(
                    info["path"],
                    input_dim=sol_dim,
                    latent_dim=info["latent_dim"],
                    num_decoders=info["num_decoders"],
                    decoder_hidden_dim=info["decoder_hidden_dim"],
                    decoder_num_layers=info["decoder_num_layers"],
                )
                if autoencoder is None:
                    print(f"SKIP: failed to load AE from {info['path']}")
                    continue

                print(f"Training NN (post-hoc) for {info['name']}...")
                nn_model = ProblemSolverNN(nn_input_dim, output_dim=sol_dim, hidden_dim=128).to(device)
                nn_model = train_nn_model_posthoc(
                    nn_model,
                    shape_name,
                    obj_type,
                    num_vars=sol_dim,
                    num_epochs=train_epochs,
                    batch_size=train_batch_size,
                )
                method_name = info["name"]
                nn_models[method_name] = (nn_model, autoencoder)
                methods.append(method_name)

                print(f"Training NN (end-to-end) for {info['name']}...")
                nn_model_e2e = ProblemSolverNN(nn_input_dim, output_dim=sol_dim, hidden_dim=128).to(device)
                nn_model_e2e = train_nn_model_e2e(
                    nn_model_e2e,
                    autoencoder,
                    norm_mean,
                    norm_std,
                    shape_name,
                    obj_type,
                    num_vars=sol_dim,
                    num_epochs=train_epochs,
                    batch_size=train_batch_size,
                )
                e2e_method_name = info["name"] + "_e2e"
                nn_models[e2e_method_name] = (nn_model_e2e, autoencoder)
                methods.append(e2e_method_name)

            if not methods:
                print(f"No models loaded for {shape_name} {obj_type}, skipping.")
                continue

            method_results = {method: {"objectives": [], "violations": [], "times": [], "optimality_gaps": []} for method in methods}

            for seed in range(num_seeds):
                np.random.seed(seed)
                torch.manual_seed(seed)
                print(f"Seed {seed + 1}/{num_seeds}")

                problems = []
                ground_truth_objectives = []

                for _ in tqdm(range(num_problems_per_seed), desc="      Generating problems"):
                    x_init = torch.randn(1, sol_dim, device=device)
                    if obj_type == "qp":
                        Q, p = generate_qp_problem(n_vars=sol_dim, batch_size=1)
                        problems.append({"Q": Q, "p": p})
                        objective_fn_gt = lambda x: 0.5 * (x * torch.matmul(Q.to(x.dtype), x.unsqueeze(-1)).squeeze(-1)).sum(dim=1) + (p.to(x.dtype) * x).sum(dim=1)
                        x_true = solve_qp_with_projection(Q, p, x_init, shape_name, max_iter=100)
                    elif obj_type == "lp":
                        c = generate_lp_problem(n_vars=sol_dim, batch_size=1)
                        problems.append({"c": c})
                        objective_fn_gt = lambda x: (c.to(x.dtype) * x).sum(dim=1)
                        x_true = solve_lp_with_projection(c, x_init, shape_name, max_iter=100)
                    else:
                        target = generate_distance_problem(n_vars=sol_dim, batch_size=1)
                        problems.append({"target": target})
                        objective_fn_gt = lambda x: ((x - target.to(x.dtype)) ** 2).sum(dim=1)
                        x_true = solve_distance_with_projection(target, x_init, shape_name, max_iter=100)

                    ground_truth_objectives.append(objective_fn_gt(x_true).item())

                for method in methods:
                    print(f"Testing {method}...")
                    for i, problem in enumerate(tqdm(problems, desc=f"      {method}")):
                        start_time = time.time()
                        if obj_type == "qp":
                            Q, p = problem["Q"], problem["p"]
                            objective_fn = lambda x: 0.5 * (x * torch.matmul(Q.to(x.dtype), x.unsqueeze(-1)).squeeze(-1)).sum(dim=1) + (p.to(x.dtype) * x).sum(dim=1)
                            problem_params = prepare_qp_input(Q, p)
                        elif obj_type == "lp":
                            c = problem["c"]
                            objective_fn = lambda x: (c.to(x.dtype) * x).sum(dim=1)
                            problem_params = prepare_lp_input(c)
                        else:
                            target = problem["target"]
                            objective_fn = lambda x: ((x - target.to(x.dtype)) ** 2).sum(dim=1)
                            problem_params = prepare_distance_input(target)

                        nn_model, autoencoder = nn_models[method]
                        
                        if autoencoder is None:
                            x_sol = nn_model(problem_params)
                        else:
                            x_pred = nn_model(problem_params)
                            with torch.no_grad():
                                x_sol = ae_project(x_pred, autoencoder, norm_mean, norm_std)

                        end_time = time.time()
                        obj_value = objective_fn(x_sol).item()
                        gt_obj_value = ground_truth_objectives[i]
                        optimality_gap = abs(obj_value - gt_obj_value)
                        x_np = x_sol.detach().cpu().numpy()
                        if shape_name == "two_moons":
                            is_feasible = two_moons_oracle(x_np)[0]
                        else:
                            is_feasible = data_generation.check_feasibility(x_np, shape_name)[0]

                        method_results[method]["objectives"].append(obj_value)
                        method_results[method]["optimality_gaps"].append(optimality_gap)
                        method_results[method]["violations"].append(0 if is_feasible else 1)
                        method_results[method]["times"].append(end_time - start_time)

            config_key = f"{shape_name}_{obj_type}"
            all_results["metrics"][config_key] = {}
            for method in methods:
                objectives = np.array(method_results[method]["objectives"])
                violations = np.array(method_results[method]["violations"])
                times = np.array(method_results[method]["times"])
                optimality_gaps = np.array(method_results[method]["optimality_gaps"])

                percentiles = [10, 25, 50, 75, 90]
                obj_percentiles = {p: np.percentile(objectives, p) for p in percentiles}
                time_percentiles = {p: np.percentile(times, p) for p in percentiles}
                gap_percentiles = {p: np.percentile(optimality_gaps, p) for p in percentiles}

                all_results["metrics"][config_key][method] = {
                    "objective_percentiles": obj_percentiles,
                    "violation_rate": np.mean(violations),
                    "mean_time": np.mean(times),
                    "time_percentiles": time_percentiles,
                    "mean_optimality_gap": np.mean(optimality_gaps),
                    "optimality_gap_percentiles": gap_percentiles,
                    "total_problems": len(objectives),
                    "std_time": np.std(times),
                    "std_violation_rate": np.std(violations),
                    "std_optimality_gap": np.std(optimality_gaps),
                }
            all_results["raw_results"][config_key] = method_results

            print(f"\nConfiguration Summary ({shape_name} + {obj_type}):")
            print(f"{'Method':<40} {'Feasibility (±std)':>20} {'Time (ms) (±std)':>20} {'Opt. Gap (±std)':>22}")
            print(f"{'-'*100}")
            for method in methods:
                metrics = all_results["metrics"][config_key][method]
                feasibility_rate = 1.0 - metrics["violation_rate"]
                avg_time_ms = metrics["mean_time"] * 1000
                avg_optimality_gap = metrics["mean_optimality_gap"]
                std_feasibility = metrics["std_violation_rate"]
                std_time_ms = metrics["std_time"] * 1000
                std_optimality_gap = metrics["std_optimality_gap"]
                feasibility_str = f"{feasibility_rate:.1%} ± {std_feasibility:.1%}"
                time_str = f"{avg_time_ms:.2f} ± {std_time_ms:.2f}"
                gap_str = f"{avg_optimality_gap:.4f} ± {std_optimality_gap:.4f}"
                method_display = method if len(method) <= 37 else method[:34] + "..."
                print(f"{method_display:<40} {feasibility_str:>20} {time_str:>20} {gap_str:>22}")
            print(f" {'-'*100}")

    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = build_run_tag(shapes_filter, exp_types_filter, configs_filter)
    results_filename = os.path.join(
        results_dir, f"testing_results_optimal_{run_tag}_{timestamp}.txt"
    )
    with open(results_filename, "w") as f:
        f.write("\nOPTIMAL ABLATION TESTING RESULTS\n" + f"Timestamp: {timestamp}\n\n")
        f.write(
            "CONFIGURATION:\n"
            f"Constraint families: {filtered_shapes}\n"
            f"Objective types: {objective_types}\n"
            f"Number of seeds: {num_seeds}\n"
            f"Problems per seed: {num_problems_per_seed}\n"
            f"Train epochs: {train_epochs}\n"
            f"Train batch size: {train_batch_size}\n\n"
            + "=" * 80 + "\n\n"
        )
        for shape_name in filtered_shapes:
            for obj_type in objective_types:
                config_key = f"{shape_name}_{obj_type}"
                if config_key not in all_results["metrics"]:
                    continue
                methods_tested_config = list(all_results["metrics"][config_key].keys())
                f.write(f"\nCONFIGURATION: {shape_name} constraints, {obj_type} objective\n" + "-" * 60 + "\n")
                for method in methods_tested_config:
                    metrics = all_results["metrics"][config_key][method]
                    f.write(f"\n  Method: {method}\n")
                    f.write(f"Total problems solved: {metrics['total_problems']}\n")
                    f.write(f"Constraint violation rate: {metrics['violation_rate']:.2%}\n")
                    f.write(f"Std dev of violation rate: {metrics['std_violation_rate']:.4f}\n")
                    f.write(f"Mean runtime: {metrics['mean_time']:.6f} seconds\n")
                    f.write(f"Std dev of runtime: {metrics['std_time']:.6f} seconds\n")
                    f.write(f"Mean optimality gap: {metrics['mean_optimality_gap']:.6f}\n")
                    f.write(f"Std dev of optimality gap: {metrics['std_optimality_gap']:.6f}\n")
                    f.write("Objective value percentiles:\n")
                    for p, val in metrics["objective_percentiles"].items():
                        f.write(f"{p}th percentile: {val:.6f}\n")
                    f.write("Optimality gap percentiles:\n")
                    for p, val in metrics["optimality_gap_percentiles"].items():
                        f.write(f"{p}th percentile: {val:.6f}\n")
                    f.write("Runtime percentiles (seconds):\n")
                    for p, val in metrics["time_percentiles"].items():
                        f.write(f"{p}th percentile: {val:.6f}\n")
                f.write("\n" + "-" * 60 + "\n")

    with open(
        os.path.join(results_dir, f"testing_results_optimal_{run_tag}_{timestamp}.pkl"),
        "wb",
    ) as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to {results_filename}")
    return results_filename


def main():
    shapes_2d = ["blob_with_bite", "star_shaped", "two_moons", "concentric_circles"]
    shapes_multidim = ["hyperspherical_shell_3d", "hyperspherical_shell_5d", "hyperspherical_shell_10d"]
    shapes = shapes_2d + shapes_multidim
    dim_exp = ["3D", "5D", "10D"]
    cov_exp = ["Cov_10", "Cov_25", "Cov_50", "Cov_75"]
    capacity_exp = ["W32_D2", "W32_D4", "W32_D6", "W64_D2", "W64_D4", "W64_D6", "W128_D2", "W128_D4", "W128_D6"]
    num_dec_exp = ["2_decoders"]
    exp_type_options = ["dim", "cov", "capacity", "num_dec"]
    lambda_recon_options = [1.5, 2.0]
    lambda_feas_options = [1.0, 1.5, 2.0]
    lambda_latent_options = [1.0, 1.5]
    lambda_geom_options = [0.025]
    lambda_hinge_options = [0.5, 1.0]

    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=shapes, default=None)
    parser.add_argument("--shapes_2d", action="store_true",
                        help="Run all 2D shapes (blob_with_bite, star_shaped, two_moons, concentric_circles)")
    parser.add_argument("--shapes_multidim", action="store_true",
                        help="Run all multidimensional shapes (hyperspherical_shell_3d, 5d, 10d)")
    parser.add_argument("--exp_type", nargs="+", choices=exp_type_options, default=None)
    parser.add_argument("--config", nargs="+", default=None)
    parser.add_argument("--models_dir", default="ablations_trained_models")
    parser.add_argument("--n_latent_samples", type=int, default=500)
    parser.add_argument("--latent_radius", type=float, default=0.5)
    parser.add_argument("--output_csv", default="optimal_ablation_params.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_latent_eval", action="store_true")
    parser.add_argument("--skip_experiments", action="store_true")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--plot_sampling", action="store_true")
    parser.add_argument("--plot_models", nargs="+", default=None)
    parser.add_argument("--plot_show", action="store_true")
    parser.add_argument("--plot_dir", default="sampling_plots")
    parser.add_argument("--penalty_nn_only", action="store_true",
                        help="Run only the penalty_nn baseline (no autoencoder models needed)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Determine which shapes to run based on arguments
    if args.shape:
        shapes_to_run = [args.shape]
    elif args.shapes_2d:
        shapes_to_run = shapes_2d
    elif args.shapes_multidim:
        shapes_to_run = shapes_multidim
    else:
        # Default: run all 2D shapes only (to maintain backward compatibility)
        shapes_to_run = shapes_2d
    
    exp_types_to_run = args.exp_type if args.exp_type else exp_type_options
    config_map = {
        "dim": dim_exp,
        "cov": cov_exp,
        "capacity": capacity_exp,
        "num_dec": num_dec_exp,
    }

    results = []
    two_moons_oracle = data_generation.build_two_moons_oracle()

    run_tag = build_run_tag(shapes_to_run, exp_types_to_run, args.config)
    output_path = args.output_csv
    if args.output_csv == "optimal_ablation_params.csv":
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_{run_tag}{ext or '.csv'}"
    if not args.skip_latent_eval and not args.penalty_nn_only:
        for shape in shapes_to_run:
            n_samples, _ = get_n_samples_and_phase1_epochs(shape, 500)
            norm_mean, norm_std, input_dim = compute_norm_params(shape, n_samples)
            for exp_type in exp_types_to_run:
                configs = config_map[exp_type]
                if args.config:
                    configs = [c for c in configs if c in args.config]
                for config in configs:
                    latent_dim = input_dim
                    num_decoders = 1
                    decoder_hidden_dim = None
                    decoder_num_layers = None
                    if exp_type == "dim":
                        latent_dim = parse_dim_config(config)
                    elif exp_type == "capacity":
                        decoder_hidden_dim, decoder_num_layers = parse_capacity_config(config)
                    elif exp_type == "num_dec":
                        num_decoders = parse_num_dec_config(config)
                    elif exp_type == "cov":
                        _ = parse_cov_config(config)

                    best_rate = None
                    best_phase2_name = None
                    lambda_grid = itertools.product(
                        lambda_recon_options,
                        lambda_feas_options,
                        lambda_latent_options,
                        lambda_geom_options,
                        lambda_hinge_options,
                    )
                    for lambda_recon, lambda_feas, lambda_latent, lambda_geom, lambda_hinge in lambda_grid:
                        lr_s = format_lambda(lambda_recon)
                        lf_s = format_lambda(lambda_feas)
                        ll_s = format_lambda(lambda_latent)
                        lg_s = format_lambda(lambda_geom)
                        lh_s = format_lambda(lambda_hinge)
                        phase2_name = f"phase2_{shape}_{exp_type}_{config}_{lr_s}_{lf_s}_{ll_s}_{lg_s}_{lh_s}"
                        phase2_path = os.path.join(args.models_dir, f"{phase2_name}.pt")

                        model = load_phase2_model(
                            phase2_path,
                            input_dim=input_dim,
                            latent_dim=latent_dim,
                            num_decoders=num_decoders,
                            decoder_hidden_dim=decoder_hidden_dim,
                            decoder_num_layers=decoder_num_layers,
                        )
                        if model is None:
                            continue
                        feas_rate = evaluate_model_feasibility(
                            model,
                            shape=shape,
                            norm_mean=norm_mean,
                            norm_std=norm_std,
                            n_latent_samples=args.n_latent_samples,
                            latent_radius=args.latent_radius,
                            two_moons_oracle=two_moons_oracle,
                        )
                        if best_rate is None or feas_rate > best_rate:
                            best_rate = feas_rate
                            best_phase2_name = f"{phase2_name}.pt"
                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    if best_phase2_name is None:
                        print(f"Missing models for {shape} {exp_type} {config}; skipping.")
                        continue
                    results.append((best_phase2_name, best_rate))
                    print(f"{shape} {exp_type} {config}: {best_phase2_name} -> {best_rate:.4f}")

                    if args.plot_sampling:
                        if args.plot_models:
                            plot_ok = False
                            plot_targets = set(args.plot_models)
                            if best_phase2_name in plot_targets:
                                plot_ok = True
                            elif os.path.splitext(best_phase2_name)[0] in plot_targets:
                                plot_ok = True
                            elif f"{shape}_{exp_type}_{config}" in plot_targets:
                                plot_ok = True
                            elif f"{shape}:{exp_type}:{config}" in plot_targets:
                                plot_ok = True
                            if not plot_ok:
                                continue
                        best_model_path = os.path.join(args.models_dir, best_phase2_name)
                        best_model = load_phase2_model(
                            best_model_path,
                            input_dim=input_dim,
                            latent_dim=latent_dim,
                            num_decoders=num_decoders,
                            decoder_hidden_dim=decoder_hidden_dim,
                            decoder_num_layers=decoder_num_layers,
                        )
                        if best_model is None:
                            print(f"SKIP: failed to load AE for plotting: {best_model_path}")
                        else:
                            plot_name = f"sampling_{shape}_{exp_type}_{config}.png"
                            plot_path = os.path.join(args.plot_dir, plot_name)
                            title_prefix = f"{shape} | {exp_type} | {config}"
                            plot_sampling_and_decoding(
                                best_model,
                                shape=shape,
                                norm_mean=norm_mean,
                                norm_std=norm_std,
                                n_latent_samples=args.n_latent_samples,
                                latent_radius=args.latent_radius,
                                two_moons_oracle=two_moons_oracle,
                                output_path=plot_path,
                                show=args.plot_show,
                                title_prefix=title_prefix,
                            )
                        del best_model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            for phase2_name, feas_rate in results:
                writer.writerow([phase2_name, f"{feas_rate:.6f}"])

        print(f"Saved {len(results)} optimal configs to {output_path}")

    if not args.skip_experiments:
        if not args.penalty_nn_only and not os.path.exists(output_path):
            raise FileNotFoundError(f"Expected {output_path} but it does not exist.")
        run_optimal_ablation_experiments(
            csv_path=output_path,
            models_dir=args.models_dir,
            results_dir=args.results_dir,
            shapes_filter=shapes_to_run,
            exp_types_filter=exp_types_to_run,
            configs_filter=args.config,
            penalty_nn_only=args.penalty_nn_only,
        )
if __name__ == "__main__":
    main()