import os
import sys
import argparse
import itertools
import time
import pickle
import re
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

from autoencoder import (
    ConstraintAwareAutoencoder, geometric_regularization_loss,
    ConditionalConstraintAwareAutoencoder, conditional_geometric_regularization_loss,
)
import data_generation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_phase1(
    model,
    X_feasible,
    batch_size,
    epochs,
    lr,
    save_path=None,
    load_path=None,
    val_split=0.2,
    wandb_run=None,
    conditional=False,
    state_dim=None,
):
    """Train the autoencoder on feasible-only data using reconstruction loss (Phase 1).

    Args:
        model: ConstraintAwareAutoencoder or ConditionalConstraintAwareAutoencoder.
        X_feasible: Array of feasible samples used for reconstruction training.
            For conditional models, each row is (state || action).
        batch_size: Number of samples per training batch.
        epochs: Number of training epochs.
        lr: Learning rate for the Adam optimizer.
        save_path: Path to save the trained model weights and loss history. Defaults to None.
        load_path: Path to load existing model weights before training. Defaults to None.
        val_split: Fraction of data reserved for validation. Defaults to 0.2.
        wandb_run: Active W&B run for logging metrics. Defaults to None.
        conditional: If True, use the conditional AE API (split inputs into state/action).
        state_dim: Number of leading columns that represent the state (required when conditional=True).

    Returns:
        Tuple of (model, history, duration_s, n_train_samples).
    """
    if conditional and state_dim is None:
        raise ValueError("state_dim is required when conditional=True")
    if load_path and os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path, map_location=device))
    X_f_train, X_f_val = train_test_split(X_feasible, test_size=val_split, random_state=42)
    X_f_train_tensor = torch.FloatTensor(X_f_train).to(device)
    X_f_val_tensor = torch.FloatTensor(X_f_val).to(device)
    train_dataset = TensorDataset(X_f_train_tensor, X_f_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_f_val_tensor, X_f_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    reconstruction_criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {'train_recon_loss': [], 'val_recon_loss': []}
    start_time = time.time()
    for epoch in tqdm(range(epochs)):
        model.train()
        train_recon_loss = 0.0
        for batch_x, _ in train_loader:
            if conditional:
                batch_state = batch_x[:, :state_dim]
                batch_action = batch_x[:, state_dim:]
                x_recon, z = model(batch_action, batch_state)
                recon_loss = reconstruction_criterion(x_recon, batch_action)
            else:
                x_recon, z = model(batch_x, classify=False)
                recon_loss = reconstruction_criterion(x_recon, batch_x)
            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()
            train_recon_loss += recon_loss.item()
        model.eval()
        val_recon_loss = 0.0
        with torch.no_grad():
            for batch_x_val, _ in val_loader:
                if conditional:
                    s_val = batch_x_val[:, :state_dim]
                    a_val = batch_x_val[:, state_dim:]
                    x_recon_val, _ = model(a_val, s_val)
                    recon_loss_val = reconstruction_criterion(x_recon_val, a_val)
                else:
                    x_recon_val, _ = model(batch_x_val, classify=False)
                    recon_loss_val = reconstruction_criterion(x_recon_val, batch_x_val)
                val_recon_loss += recon_loss_val.item()
        epoch_metrics = {
            'train_recon_loss': train_recon_loss / len(train_loader),
            'val_recon_loss': val_recon_loss / len(val_loader),
        }
        history['train_recon_loss'].append(epoch_metrics['train_recon_loss'])
        history['val_recon_loss'].append(epoch_metrics['val_recon_loss'])
        if wandb_run is not None:
            wandb_run.log({**epoch_metrics, "epoch": epoch + 1}, step=epoch + 1)
    duration_s = time.time() - start_time
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        history_path = save_path.replace('.pt', '_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
    return model, history, duration_s, len(train_loader.dataset)


def train_phase2(
    model,
    X_all,
    feasible_mask,
    shape_name,
    batch_size,
    epochs,
    lambda_recon,
    lambda_feasibility,
    lambda_latent,
    lambda_hinge,
    lambda_geometric,
    lr_ae,
    lr_d,
    discriminator,
    save_path=None,
    load_path=None,
    k_critic_steps=3,
    normalize_inputs=True,
    force_mask_labels=False,
    wandb_run=None,
    conditional=False,
    state_dim=None,
):
    """Phase 2 training.

    Alternates between updating the feasibility predictor (k_critic_steps) and the
    autoencoder, incorporating reconstruction, classification, latent coverage, hinge,
    and geometric regularization losses.

    Args:
        model: ConstraintAwareAutoencoder or ConditionalConstraintAwareAutoencoder.
        X_all: Array of all samples (feasible and infeasible).
            For conditional models, each row is (state || action).
        feasible_mask: Boolean or float array indicating feasibility of each sample in X_all.
        shape_name: Name of the constraint shape; controls oracle feasibility checks and loss masking.
        batch_size: Number of samples per training batch.
        epochs: Number of training epochs.
        lambda_recon: Weight for the reconstruction loss.
        lambda_feasibility: Weight for the feasibility classification loss on reconstructions.
        lambda_latent: Weight for the feasibility loss on latent-space samples.
        lambda_hinge: Weight for the hinge loss separating feasible/infeasible latent norms.
        lambda_geometric: Weight for the geometric regularization loss.
        lr_ae: Learning rate for the autoencoder optimizer.
        lr_d: Learning rate for the feasibility predictor optimizer.
        discriminator: Strategy for the feasibility predictor labels ('absolute' uses oracle labels).
        save_path: Path to save the trained model weights and loss history. Defaults to None.
        load_path: Path to load Phase 1 model weights before training. Defaults to None.
        k_critic_steps: Number of predictor update steps per AE update step. Defaults to 3.
        normalize_inputs: Whether to z-score normalize inputs before training. Defaults to True.
        force_mask_labels: Force use of mask-derived labels instead of oracle queries. Defaults to False.
        wandb_run: Active W&B run for logging metrics. Defaults to None.
        conditional: If True, use the conditional AE API (split inputs into state/action).
        state_dim: Number of leading columns that represent the state (required when conditional=True).

    Returns:
        Tuple of (model, history, duration_s, n_train_samples).
    """
    if conditional and state_dim is None:
        raise ValueError("state_dim is required when conditional=True")
    if load_path and os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path, map_location=device))
    X_all_train, X_all_val, mask_train, mask_val = train_test_split(
        X_all, feasible_mask.astype(float), test_size=0.2, random_state=42
    )
    train_tensor = torch.FloatTensor(X_all_train).to(device)
    val_tensor = torch.FloatTensor(X_all_val).to(device)
    train_mask_tensor = torch.FloatTensor(mask_train).unsqueeze(1).to(device)
    val_mask_tensor = torch.FloatTensor(mask_val).unsqueeze(1).to(device)
    if normalize_inputs and shape_name != 'ieee37bus':
        norm_mean = train_tensor.mean(dim=0, keepdim=True)
        norm_std = train_tensor.std(dim=0, keepdim=True)
        eps = torch.tensor(1e-8, device=device)
        norm_std = torch.where(norm_std < eps, eps, norm_std)
        train_tensor = (train_tensor - norm_mean) / norm_std
        val_tensor = (val_tensor - norm_mean) / norm_std
    else:
        norm_mean = torch.zeros((1, train_tensor.size(1)), device=device)
        norm_std = torch.ones((1, train_tensor.size(1)), device=device)

    def denorm(x):
        return x * norm_std + norm_mean

    # --- helpers that abstract conditional vs unconditional API ----------
    def _split(bx):
        if conditional:
            return bx[:, :state_dim], bx[:, state_dim:]
        return None, bx

    def _forward(bx):
        if conditional:
            s, a = _split(bx)
            return model(a, s)
        return model(bx, classify=False)

    def _decode(z, bx):
        if conditional:
            s = bx[:, :state_dim]
            return model.decode(z, s)
        return model.decode(z)

    def _predict_feas(x_or_action, bx):
        if conditional:
            s = bx[:, :state_dim]
            return model.predict_feasibility_with_nn(x_or_action, s)
        return model.predict_feasibility_with_nn(x_or_action)

    def _predict_feas_raw(bx):
        if conditional:
            s, a = _split(bx)
            return model.predict_feasibility_with_nn(a, s)
        return model.feasibility_predictor_nn(bx)

    def _recon_target(bx):
        if conditional:
            return bx[:, state_dim:]
        return bx

    def _geom_loss(z, bx):
        if conditional:
            s = bx[:, :state_dim]
            return conditional_geometric_regularization_loss(model, z, s, alpha=1.0)
        return geometric_regularization_loss(model, z, alpha=1.0)

    has_oracle = shape_name not in ('safety_gym', 'safety_gym_traj')

    def _verify_feas(bx_denormed):
        """Oracle feasibility; only valid for CO shapes."""
        if conditional:
            s, a = bx_denormed[:, :state_dim], bx_denormed[:, state_dim:]
            return model.verify_feasibility(a, s, shape_name)
        return model.verify_feasibility(bx_denormed, shape_name)
    # --------------------------------------------------------------------

    train_dataset = TensorDataset(train_tensor, train_mask_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_tensor, val_mask_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    reconstruction_criterion = torch.nn.MSELoss()
    num_pos = float(train_mask_tensor.sum().item())
    num_total = float(train_mask_tensor.numel())
    num_neg = max(num_total - num_pos, 1.0)
    pos_weight_value = torch.tensor(num_neg / max(num_pos, 1.0), device=device, dtype=torch.float32)
    classification_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_value)
    decoder_params = []
    for decoder in model.decoders:
        decoder_params.extend(list(decoder.parameters()))
    optimizer_AE = optim.Adam(
        list(model.encoder.parameters()) +
        decoder_params +
        list(model.gating_network.parameters()),
        lr=lr_ae
    )
    optimizer_D = optim.Adam(model.feasibility_predictor_nn.parameters(), lr=lr_d)
    history = {
        'train_recon_loss': [], 'val_recon_loss': [],
        'train_ae_class_loss': [], 'val_ae_class_loss': [],
        'train_ae_latent_loss': [], 'val_ae_latent_loss': [],
        'train_hinge_loss': [], 'val_hinge_loss': [],
        'train_total_ae_loss': [], 'val_total_ae_loss': [],
        'train_predictor_loss': [], 'val_predictor_loss': [],
        'train_geometric_loss': [], 'val_geometric_loss': [],
        'val_predictor_accuracy': []
    }
    start_time = time.time()
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_losses = {k: 0.0 for k in history if k.startswith('train')}
        for batch_x, batch_y_original in train_loader:
            batch_x = batch_x.to(device)
            batch_y_original = batch_y_original.to(device)
            # --- Discriminator / feasibility-predictor update -------
            model.feasibility_predictor_nn.train()
            for _ in range(k_critic_steps):
                optimizer_D.zero_grad()
                if (shape_name in ['safety_gym'] or force_mask_labels) and discriminator == "absolute":
                    pred_real = _predict_feas_raw(batch_x)
                    loss_D = classification_criterion(pred_real, batch_y_original)
                else:
                    with torch.no_grad():
                        x_recon, _ = _forward(batch_x)
                        z_sampled = torch.randn(batch_x.size(0), model.latent_dim, device=device).tanh()
                        x_from_latent = _decode(z_sampled, batch_x)
                    if discriminator == "absolute":
                        pred_real = _predict_feas_raw(batch_x)
                        pred_fake_recon = _predict_feas(x_recon, batch_x)
                        pred_fake_latent = _predict_feas(x_from_latent, batch_x)
                        pred_fake = torch.cat([pred_fake_recon, pred_fake_latent], dim=0)
                        oracle_real = _verify_feas(denorm(batch_x))
                        x_recon_full = torch.cat([batch_x[:, :state_dim], x_recon], dim=1) if conditional else x_recon
                        x_latent_full = torch.cat([batch_x[:, :state_dim], x_from_latent], dim=1) if conditional else x_from_latent
                        oracle_fake = _verify_feas(denorm(torch.cat([x_recon_full, x_latent_full], dim=0)))
                        loss_D = classification_criterion(pred_real, oracle_real) + classification_criterion(pred_fake, oracle_fake)
                    else:
                        loss_D = classification_criterion(_predict_feas_raw(batch_x), batch_y_original)
                loss_D.backward()
                optimizer_D.step()
                epoch_losses['train_predictor_loss'] += loss_D.item()
            # --- Autoencoder update ---------------------------------
            model.encoder.train()
            model.decoders.train()
            model.feasibility_predictor_nn.eval()
            optimizer_AE.zero_grad()
            x_recon, z = _forward(batch_x)
            target = _recon_target(batch_x)
            if shape_name in ['safety_gym', 'safety_gym_traj'] or force_mask_labels:
                feas_mask_b = (batch_y_original > 0.5).squeeze(1)
                if feas_mask_b.any():
                    recon_loss = reconstruction_criterion(x_recon[feas_mask_b], target[feas_mask_b])
                else:
                    recon_loss = torch.tensor(0.0, device=device)
            else:
                recon_loss = reconstruction_criterion(x_recon, target)
            logits_recon = _predict_feas(x_recon, batch_x)
            if shape_name == 'safety_gym' or force_mask_labels:
                ae_class_loss = classification_criterion(logits_recon, torch.ones_like(logits_recon))
            else:
                ae_class_loss = classification_criterion(logits_recon, batch_y_original)
            z_sampled = torch.randn(batch_x.size(0), model.latent_dim, device=device)
            z_norm_sample = torch.norm(z_sampled, p=2, dim=1, keepdim=True)
            z_unit_sphere = z_sampled / (z_norm_sample + 1e-8)
            u = torch.rand(batch_x.size(0), 1, device=device) ** (1.0 / model.latent_dim)
            z_sampled = z_unit_sphere * u * 0.5
            x_from_latent = _decode(z_sampled, batch_x)
            logits_latent = _predict_feas(x_from_latent, batch_x)
            logits_latent = torch.nan_to_num(logits_latent, nan=0.0, posinf=50.0, neginf=-50.0)
            ae_latent_loss = classification_criterion(logits_latent, torch.ones_like(logits_latent))
            z_norm = torch.norm(z, p=2, dim=1, keepdim=True)
            hinge_feasible = batch_y_original * torch.clamp(z_norm - 0.5, min=0)
            hinge_infeasible = (1 - batch_y_original) * torch.clamp(0.5 - z_norm, min=0)
            hinge_loss = torch.mean(hinge_feasible + hinge_infeasible)
            geometric_loss = _geom_loss(z, batch_x)
            total_ae_loss = (
                lambda_recon * recon_loss +
                lambda_feasibility * ae_class_loss +
                lambda_latent * ae_latent_loss +
                lambda_hinge * hinge_loss +
                lambda_geometric * geometric_loss
            )
            total_ae_loss.backward()
            optimizer_AE.step()
            epoch_losses['train_recon_loss'] += recon_loss.item()
            epoch_losses['train_ae_class_loss'] += ae_class_loss.item()
            epoch_losses['train_ae_latent_loss'] += ae_latent_loss.item()
            epoch_losses['train_hinge_loss'] += hinge_loss.item()
            epoch_losses['train_total_ae_loss'] += total_ae_loss.item()
            epoch_losses['train_geometric_loss'] += geometric_loss.item()
        # --- Validation --------------------------------------------------
        model.eval()
        val_losses = {k: 0.0 for k in history if k.startswith('val')}
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch_x_val, batch_y_original_val in val_loader:
                x_recon_val, z_val = _forward(batch_x_val)
                target_val = _recon_target(batch_x_val)
                recon_b = reconstruction_criterion(x_recon_val, target_val).item()
                val_losses['val_recon_loss'] += recon_b
                pred_val_recon = _predict_feas(x_recon_val, batch_x_val)
                class_recon_b = classification_criterion(pred_val_recon, batch_y_original_val).item()
                val_losses['val_ae_class_loss'] += class_recon_b
                z_sampled_val = torch.randn(batch_x_val.size(0), model.latent_dim, device=device)
                z_norm_sample_val = torch.norm(z_sampled_val, p=2, dim=1, keepdim=True)
                z_unit_sphere_val = z_sampled_val / (z_norm_sample_val + 1e-8)
                u_val = torch.rand(batch_x_val.size(0), 1, device=device) ** (1.0 / model.latent_dim)
                z_sampled_val = z_unit_sphere_val * u_val * 0.5
                x_from_latent_val = _decode(z_sampled_val, batch_x_val)
                logits_latent_val = _predict_feas(x_from_latent_val, batch_x_val)
                logits_latent_val = torch.nan_to_num(logits_latent_val, nan=0.0, posinf=50.0, neginf=-50.0)
                latent_b = classification_criterion(logits_latent_val, torch.ones_like(logits_latent_val)).item()
                val_losses['val_ae_latent_loss'] += latent_b
                z_norm_val = torch.norm(z_val, p=2, dim=1, keepdim=True)
                hinge_feasible_val = batch_y_original_val * torch.clamp(z_norm_val - 0.5, min=0)
                hinge_infeasible_val = (1 - batch_y_original_val) * torch.clamp(0.5 - z_norm_val, min=0)
                hinge_b = torch.mean(hinge_feasible_val + hinge_infeasible_val).item()
                val_losses['val_hinge_loss'] += hinge_b
                val_losses['val_total_ae_loss'] += (
                    lambda_recon * recon_b +
                    lambda_feasibility * class_recon_b +
                    lambda_latent * latent_b +
                    lambda_hinge * hinge_b
                )
                if has_oracle:
                    oracle_labels_val = _verify_feas(denorm(batch_x_val))
                else:
                    oracle_labels_val = batch_y_original_val
                predictor_logits_val = _predict_feas_raw(batch_x_val)
                val_losses['val_predictor_loss'] += classification_criterion(predictor_logits_val, oracle_labels_val).item()
                predicted_labels = (predictor_logits_val > 0).float()
                total_correct += (predicted_labels == oracle_labels_val).sum().item()
                total_samples += batch_y_original_val.size(0)
        epoch_metrics = {}
        for key in history:
            if key.startswith('train'):
                epoch_metrics[key] = epoch_losses[key] / len(train_loader)
                history[key].append(epoch_metrics[key])
            elif key.startswith('val') and 'accuracy' not in key:
                epoch_metrics[key] = val_losses[key] / len(val_loader)
                history[key].append(epoch_metrics[key])
        epoch_metrics['val_predictor_accuracy'] = total_correct / total_samples
        history['val_predictor_accuracy'].append(epoch_metrics['val_predictor_accuracy'])
        if wandb_run is not None:
            wandb_run.log({**epoch_metrics, "epoch": epoch + 1}, step=epoch + 1)
    duration_s = time.time() - start_time
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        history_path = save_path.replace('.pt', '_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
    return model, history, duration_s, len(train_loader.dataset)


def format_lambda(value):
    """Format a float lambda value as a compact string with trailing zeros removed."""
    s = f"{value:.2f}".rstrip('0').rstrip('.')
    return s if s else "0"


def parse_capacity_config(config):
    """Parse a capacity config string (e.g. 'W64_D4') into (hidden_dim, num_layers)."""
    match = re.match(r'^W(\d+)_D(\d+)$', config)
    if not match:
        raise ValueError(f"Invalid capacity config: {config}")
    return int(match.group(1)), int(match.group(2))


def parse_dim_config(config):
    """Parse a latent dimension config string (e.g. '3D') into an integer."""
    return int(config.replace('D', '').strip())


def parse_cov_config(config):
    """Parse a coverage config string (e.g. 'Cov_50') into an integer percentage."""
    return int(config.replace('Cov_', '').strip())


def parse_num_dec_config(config):
    """Parse a decoder count config string (e.g. '2_decoders') into an integer."""
    return int(config.split('_')[0])


def get_n_samples_and_phase1_epochs(shape, default_epochs):
    """Return the recommended (n_samples, phase1_epochs) for a given constraint shape.

    Higher-dimensional shells require more samples and, in some cases, more epochs
    to adequately cover the feasible manifold.
    """
    if shape == "hyperspherical_shell_3d":
        return 90000, default_epochs
    if shape == "hyperspherical_shell_5d":
        return 150000, default_epochs
    if shape == "hyperspherical_shell_10d":
        return 350000, default_epochs
    # if shape == "hyperspherical_shell_50d":
    #     return 1200000, 300
    return 60000, default_epochs


def main():
    """Parse arguments, generate data, and run Phase 1 + Phase 2 training sweeps.

    Iterates over selected shapes, experiment types, and lambda hyperparameter grids,
    skipping runs whose output checkpoints already exist. Logs all metrics to W&B.
    """
    shapes_2d = ['blob_with_bite', 'star_shaped', 'two_moons', 'concentric_circles']
    shapes_multidim = [
        'hyperspherical_shell_3d', 'hyperspherical_shell_5d',
        'hyperspherical_shell_10d',
    ]
    shapes_rl = ['safety_gym']
    shapes = shapes_2d + shapes_multidim + shapes_rl
    dim_exp = ['3D', '5D', '10D']
    cov_exp = ['Cov_10', 'Cov_25', 'Cov_50', 'Cov_75']
    capacity_exp = ['W32_D2', 'W32_D4', 'W32_D6', 'W64_D2', 'W64_D4', 'W64_D6', 'W128_D2', 'W128_D4', 'W128_D6']
    num_dec_exp = ['2_decoders']
    exp_type_options = ['dim', 'cov', 'capacity', 'num_dec']
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
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to .npz dataset (used with --shape safety_gym)")
    parser.add_argument("--state_dim", type=int, default=None,
                        help="Observation/state dimension for conditional AE (safety_gym). "
                             "Auto-detected from .npz if not provided.")
    parser.add_argument("--exp_type", nargs="+", choices=exp_type_options, default=None)
    parser.add_argument("--config", nargs="+", default=None)
    parser.add_argument("--lambda_recon", type=float, nargs="+", default=None,
                        help="Override lambda_recon values (default: grid of [1.5, 2.0])")
    parser.add_argument("--lambda_feas", type=float, nargs="+", default=None,
                        help="Override lambda_feas values (default: grid of [1.0, 1.5, 2.0])")
    parser.add_argument("--lambda_latent", type=float, nargs="+", default=None,
                        help="Override lambda_latent values (default: grid of [1.0, 1.5])")
    parser.add_argument("--lambda_geom", type=float, nargs="+", default=None,
                        help="Override lambda_geom values (default: grid of [0.025])")
    parser.add_argument("--lambda_hinge", type=float, nargs="+", default=None,
                        help="Override lambda_hinge values (default: grid of [0.5, 1.0])")
    args = parser.parse_args()
    if args.lambda_recon is not None:
        lambda_recon_options = args.lambda_recon
    if args.lambda_feas is not None:
        lambda_feas_options = args.lambda_feas
    if args.lambda_latent is not None:
        lambda_latent_options = args.lambda_latent
    if args.lambda_geom is not None:
        lambda_geom_options = args.lambda_geom
    if args.lambda_hinge is not None:
        lambda_hinge_options = args.lambda_hinge
    if args.shape:
        shapes_to_run = [args.shape]
    elif args.shapes_2d:
        shapes_to_run = shapes_2d
    elif args.shapes_multidim:
        shapes_to_run = shapes_multidim
    else:
        shapes_to_run = shapes_2d
    if 'safety_gym' in shapes_to_run and len(shapes_to_run) > 1:
        print("Warning: safety_gym uses a conditional AE and will be trained separately.")
    exp_types_to_run = args.exp_type if args.exp_type else exp_type_options
    config_map = {
        "dim": dim_exp,
        "cov": cov_exp,
        "capacity": capacity_exp,
        "num_dec": num_dec_exp
    }
    output_dir = "ablations_trained_models"
    os.makedirs(output_dir, exist_ok=True)
    for shape in shapes_to_run:
        is_conditional = (shape == 'safety_gym')
        n_samples, phase1_epochs = get_n_samples_and_phase1_epochs(shape, 500)

        if is_conditional:
            dataset_path = args.dataset_path or "safe_rl/dataset_pointgoal2.npz"
            X_feasible, X_infeasible, X_all, feasible_mask = data_generation.generate_nonconvex_data(
                shape_name=shape, n_samples=n_samples,
            )
            # Determine state_dim: prefer CLI arg, else infer from .npz metadata
            sg_state_dim = args.state_dim
            if sg_state_dim is None:
                D = np.load(dataset_path)
                if 'obs_dim' in D.files:
                    sg_state_dim = int(D['obs_dim'])
                else:
                    raise ValueError(
                        "Cannot infer state_dim from dataset. "
                        "Provide --state_dim explicitly."
                    )
            action_dim = X_feasible.shape[1] - sg_state_dim
        else:
            sg_state_dim = None
            X_feasible, X_infeasible, X_all, feasible_mask = data_generation.generate_nonconvex_data(
                shape_name=shape, n_samples=n_samples
            )

        for exp_type in exp_types_to_run:
            configs = config_map[exp_type]
            if args.config:
                configs = [c for c in configs if c in args.config]
            for config in configs:
                input_dim = X_feasible.shape[1]
                latent_dim = input_dim if not is_conditional else action_dim
                num_decoders = 1
                decoder_hidden_dim = None
                decoder_num_layers = None
                X_feasible_phase1 = X_feasible
                feasible_mask_phase2 = feasible_mask.astype(float)
                force_mask_labels = is_conditional
                if exp_type == "dim":
                    latent_dim = parse_dim_config(config)
                elif exp_type == "capacity":
                    decoder_hidden_dim, decoder_num_layers = parse_capacity_config(config)
                elif exp_type == "num_dec":
                    num_decoders = parse_num_dec_config(config)
                elif exp_type == "cov":
                    cov_pct = parse_cov_config(config)
                    frac = cov_pct / 100.0
                    num_feas = X_feasible.shape[0]
                    rng = np.random.default_rng(42)
                    k = max(1, int(np.ceil(frac * num_feas)))
                    indices = rng.permutation(num_feas)
                    seen_idx = indices[:k]
                    seen_mask = np.zeros(num_feas, dtype=bool)
                    seen_mask[seen_idx] = True
                    X_feasible_phase1 = X_feasible[seen_mask]
                    feas_indices = np.where(feasible_mask)[0]
                    reduced_mask = np.zeros_like(feasible_mask, dtype=bool)
                    reduced_mask[feas_indices[seen_idx]] = True
                    feasible_mask_phase2 = reduced_mask.astype(float)
                    force_mask_labels = True

                def _make_model():
                    if is_conditional:
                        return ConditionalConstraintAwareAutoencoder(
                            action_dim=action_dim,
                            state_dim=sg_state_dim,
                            latent_dim=latent_dim,
                            hidden_dim=64,
                            num_decoders=num_decoders,
                        ).to(device)
                    return ConstraintAwareAutoencoder(
                        input_dim=input_dim,
                        latent_dim=latent_dim,
                        hidden_dim=64,
                        num_decoders=num_decoders,
                        decoder_hidden_dim=decoder_hidden_dim,
                        decoder_num_layers=decoder_num_layers,
                    ).to(device)

                phase1_name = f"phase1_{shape}_{exp_type}_{config}"
                phase1_path = os.path.join(output_dir, f"{phase1_name}.pt")
                if not os.path.exists(phase1_path):
                    phase1_run = wandb.init(
                        project="ablations_training",
                        name=phase1_name,
                        config={
                            "shape": shape,
                            "exp_type": exp_type,
                            "config": config,
                            "batch_size": 256,
                            "discriminator": "absolute",
                            "phase": 1,
                            "epochs": phase1_epochs,
                            "lr": 0.001,
                            "input_dim": input_dim,
                            "latent_dim": latent_dim,
                            "num_decoders": num_decoders,
                            "decoder_hidden_dim": decoder_hidden_dim,
                            "decoder_num_layers": decoder_num_layers,
                            "conditional": is_conditional,
                            "state_dim": sg_state_dim,
                        }
                    )
                    model = _make_model()
                    model, history, duration_s, train_samples = train_phase1(
                        model,
                        X_feasible_phase1,
                        batch_size=256,
                        epochs=phase1_epochs,
                        lr=0.001,
                        save_path=phase1_path,
                        wandb_run=phase1_run,
                        conditional=is_conditional,
                        state_dim=sg_state_dim,
                    )
                    throughput = (train_samples * phase1_epochs) / max(duration_s, 1e-8)
                    phase1_run.log({"throughput": throughput, "training_time": duration_s})
                    phase1_run.finish()
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                lambda_grid = itertools.product(
                    lambda_recon_options,
                    lambda_feas_options,
                    lambda_latent_options,
                    lambda_geom_options,
                    lambda_hinge_options
                )
                for lambda_recon, lambda_feas, lambda_latent, lambda_geom, lambda_hinge in lambda_grid:
                    lr_s = format_lambda(lambda_recon)
                    lf_s = format_lambda(lambda_feas)
                    ll_s = format_lambda(lambda_latent)
                    lg_s = format_lambda(lambda_geom)
                    lh_s = format_lambda(lambda_hinge)
                    phase2_name = f"phase2_{shape}_{exp_type}_{config}_{lr_s}_{lf_s}_{ll_s}_{lg_s}_{lh_s}"
                    phase2_path = os.path.join(output_dir, f"{phase2_name}.pt")
                    if os.path.exists(phase2_path):
                        continue
                    phase2_run = wandb.init(
                        project="ablations_training",
                        name=phase2_name,
                        config={
                            "shape": shape,
                            "exp_type": exp_type,
                            "config": config,
                            "batch_size": 256,
                            "discriminator": "absolute",
                            "phase": 2,
                            "epochs": 100,
                            "lr_ae": 0.001,
                            "lr_d": 0.001,
                            "lambda_recon": lambda_recon,
                            "lambda_feas": lambda_feas,
                            "lambda_latent": lambda_latent,
                            "lambda_geom": lambda_geom,
                            "lambda_hinge": lambda_hinge,
                            "input_dim": input_dim,
                            "latent_dim": latent_dim,
                            "num_decoders": num_decoders,
                            "decoder_hidden_dim": decoder_hidden_dim,
                            "decoder_num_layers": decoder_num_layers,
                            "conditional": is_conditional,
                            "state_dim": sg_state_dim,
                        }
                    )
                    model = _make_model()
                    model, history, duration_s, train_samples = train_phase2(
                        model,
                        X_all,
                        feasible_mask_phase2,
                        shape_name=shape,
                        batch_size=256,
                        epochs=100,
                        lambda_recon=lambda_recon,
                        lambda_feasibility=lambda_feas,
                        lambda_latent=lambda_latent,
                        lambda_hinge=lambda_hinge,
                        lambda_geometric=lambda_geom,
                        lr_ae=0.001,
                        lr_d=0.001,
                        discriminator="absolute",
                        save_path=phase2_path,
                        load_path=phase1_path,
                        normalize_inputs=True,
                        force_mask_labels=force_mask_labels,
                        wandb_run=phase2_run,
                        conditional=is_conditional,
                        state_dim=sg_state_dim,
                    )
                    throughput = (train_samples * 100) / max(duration_s, 1e-8)
                    phase2_run.log({"throughput": throughput, "training_time": duration_s})
                    phase2_run.finish()
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()