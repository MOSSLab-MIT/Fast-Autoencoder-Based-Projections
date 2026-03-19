"""
    Autoencoder implementations.

    - ConstraintAwareAutoencoder: unconditional AE for constrained optimization (CO) datasets.
    - ConditionalConstraintAwareAutoencoder: state-conditioned AE for safe RL (encodes/decodes
      actions conditioned on observations).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

import data_generation

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConstraintAwareAutoencoder(nn.Module):
    def __init__(self, input_dim=2, latent_dim=2, hidden_dim=64,
                 num_decoders=1, latent_geom="hypersphere",
                 decoder_hidden_dim=None, decoder_num_layers=None):
        super(ConstraintAwareAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_decoders = num_decoders
        self.latent_geom = latent_geom
        self.decoder_hidden_dim = decoder_hidden_dim if decoder_hidden_dim is not None else hidden_dim
        self.decoder_num_layers = decoder_num_layers if decoder_num_layers is not None else 4

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()  # Constrain latent space to [-1, 1] hypercube
        )

        # Build decoder(s) with configurable width and depth
        def build_decoder():
            layers = []
            in_dim = latent_dim
            for _ in range(self.decoder_num_layers):
                layers.append(nn.Linear(in_dim, self.decoder_hidden_dim))
                layers.append(nn.ReLU())
                in_dim = self.decoder_hidden_dim
            layers.append(nn.Linear(in_dim, input_dim))
            return nn.Sequential(*layers)

        self.decoders = nn.ModuleList([build_decoder() for _ in range(num_decoders)])
        
        # Gating network: determines weights for each decoder
        self.gating_network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_decoders),
            nn.Softmax(dim=-1)  # Ensure weights sum to 1
        )
        
        self.feasibility_predictor_nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # Take normalized x as input
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        """
        Decode latent representations using mixture of experts.
        z: Latent representations (batch_size, latent_dim)
        Return decoded outputs (batch_size, input_dim)
        """
        gate_weights = self.gating_network(z)
        decoder_outputs = []
        for decoder in self.decoders:
            output = decoder(z)
            decoder_outputs.append(output)
        decoder_outputs = torch.stack(decoder_outputs, dim=0)
        gate_weights = gate_weights.t().unsqueeze(-1)
        # weighted sum of decoder outputs
        mixed_output = (decoder_outputs * gate_weights).sum(dim=0) 
        return mixed_output

    def forward(self, x, classify=True):
        """
        Forward pass through the autoencoder
        x: Input points in original space
        # classify: Whether to run the classifier (set to False during initial reconstruction training)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def predict_feasibility_with_nn(self, x_normalized_batch):
        """
        Predicts feasibility using the trained neural network predictor (Critic D).
        Input should be a normalized batch of points.
        Output is a tensor of probabilities.
        """
        return self.feasibility_predictor_nn(x_normalized_batch)
    
    def verify_feasibility(self, x_batch, shape_name):
        """
        Verifies feasibility of points using the ground-truth geometric function.
        For CO datasets only.
        x_batch (torch.Tensor or np.ndarray): Batch of points to check.
        shape_name (str): The name of the shape to check against.
        Returns a float tensor of shape (batch_size, 1) with 1.0 for feasible and 0.0 for infeasible.
        """
        if isinstance(x_batch, torch.Tensor):
            x_np = x_batch.detach().cpu().numpy()
        else:
            x_np = x_batch
        feasible_mask = data_generation.check_feasibility(x_np, shape_name)
        feasibility_scores = torch.tensor(feasible_mask, dtype=torch.float32, device=device).unsqueeze(1)
        return feasibility_scores
    
    def project_to_feasible(self, x, beta: float = 20.0):
        """
        Differentiable projection of points to the feasible set via the latent space.
        Softplus-based clamping: near-exact identity for feasible points,
        smooth saturation for infeasible points. Beta controls sharpness.
        """
        z = self.encode(x)
        if self.latent_geom == "hypersphere":
            z_norm = torch.norm(z, dim=1, keepdim=True)
            projected_norm = z_norm - (1.0 / beta) * F.softplus(beta * (z_norm - 0.5))
            z_projected = z * projected_norm / (z_norm + 1e-8)
        # elif self.latent_geom == "hypercube":
        #     z_projected = z - (1.0 / beta) * F.softplus(beta * (z - 0.5)) \
        #                     + (1.0 / beta) * F.softplus(beta * (-z - 0.5))
        return self.decode(z_projected)

def geometric_regularization_loss(model, z_batch, alpha=1.0):
    """
    Encourage uniform Jacobian determinants across latent space.
    """
    if not z_batch.requires_grad:
        z_batch.requires_grad_(True)
    batch_size = z_batch.size(0)
    log_det_values = []
    num_samples_to_process = min(batch_size, 32)

    for i in range(num_samples_to_process):
        z_sample_i = z_batch[i:i+1]
        x_decoded_sample_i = model.decode(z_sample_i)

        jacobian_rows = []
        for j in range(x_decoded_sample_i.size(1)):
            grad_j = torch.autograd.grad(
                outputs=x_decoded_sample_i[0, j],
                inputs=z_sample_i,
                grad_outputs=torch.ones_like(x_decoded_sample_i[0, j]),
                retain_graph=True,
                create_graph=True
            )[0]
            jacobian_rows.append(grad_j)
        jacobian_i = torch.stack(jacobian_rows, dim=0)
        jacobian_i = jacobian_i.squeeze(1)

        matrix_for_det = jacobian_i @ jacobian_i.T + 1e-6 * torch.eye(jacobian_i.size(0), device=jacobian_i.device)
        det = torch.det(matrix_for_det)
        if torch.isfinite(det) and det > 0:
            det_clamped = torch.clamp(det, min=1e-12, max=1e12)
            log_det_values.append(torch.log(det_clamped))

    if len(log_det_values) == 0:
        return torch.tensor(0.0, device=z_batch.device, dtype=z_batch.dtype)

    log_jacobian_dets = torch.stack(log_det_values)
    log_jacobian_dets = torch.nan_to_num(log_jacobian_dets, nan=0.0, posinf=0.0, neginf=0.0)
    variance_penalty = torch.var(log_jacobian_dets)

    return alpha * variance_penalty

# ---------------------------------------------------------------------------
# Conditional (state-conditioned) autoencoder for safe RL
# ---------------------------------------------------------------------------

class ConditionalConstraintAwareAutoencoder(nn.Module):
    """Autoencoder that encodes/decodes *actions* conditioned on *states*.

    Used for safe RL: the encoder maps (action, state) → latent, and the
    decoder maps (latent, state) → action.  Projection to the feasible set
    is done by clamping the latent representation to a convex set (ball /
    hypercube) then decoding.
    """

    def __init__(self, action_dim, state_dim, latent_dim=None, hidden_dim=64,
                 num_decoders=1, latent_geom="hypersphere"):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.latent_dim = latent_dim if latent_dim is not None else action_dim
        self.hidden_dim = hidden_dim
        self.num_decoders = num_decoders
        self.latent_geom = latent_geom
        self.latent_radius = 0.5

        self.encoder = nn.Sequential(
            nn.Linear(action_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.latent_dim),
            nn.Tanh(),
        )

        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_dim + state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )
            for _ in range(num_decoders)
        ])

        self.gating_network = nn.Sequential(
            nn.Linear(self.latent_dim + state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_decoders),
            nn.Softmax(dim=-1),
        )

        self.feasibility_predictor_nn = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    # ----- core forward path -------------------------------------------

    def encode(self, action, state):
        return self.encoder(torch.cat([action, state], dim=-1))

    def decode(self, z, state):
        z_state = torch.cat([z, state], dim=-1)
        gate_weights = self.gating_network(z_state)
        decoder_outputs = torch.stack([dec(z_state) for dec in self.decoders], dim=0)
        gate_weights = gate_weights.t().unsqueeze(-1)
        return (decoder_outputs * gate_weights).sum(dim=0)

    def forward(self, action, state):
        z = self.encode(action, state)
        action_recon = self.decode(z, state)
        return action_recon, z

    # ----- feasibility helpers -----------------------------------------

    def predict_feasibility_with_nn(self, action, state):
        return self.feasibility_predictor_nn(torch.cat([state, action], dim=-1))

    def verify_feasibility(self, action_batch, state_batch, shape_name):
        """Ground-truth feasibility check on (state, action) pairs.

        Falls back to data_generation.check_feasibility on the concatenated
        (state, action) vector.
        """
        if isinstance(action_batch, torch.Tensor):
            x_np = torch.cat([state_batch, action_batch], dim=-1).detach().cpu().numpy()
        else:
            x_np = np.concatenate([state_batch, action_batch], axis=-1)
        feasible_mask = data_generation.check_feasibility(x_np, shape_name)
        return torch.tensor(
            feasible_mask, dtype=torch.float32, device=device,
        ).unsqueeze(1)

    # ----- projection --------------------------------------------------

    def project_action(self, action_batch, state_batch, beta: float = 20.0):
        """Differentiable projection of actions to the feasible set via the
        latent space.  Softplus-based clamping: near-exact identity for
        feasible points, smooth saturation for infeasible points.  Beta
        controls sharpness."""
        z = self.encode(action_batch, state_batch)
        if self.latent_geom == "hypersphere":
            z_norm = torch.norm(z, dim=-1, keepdim=True)
            projected_norm = z_norm - (1.0 / beta) * F.softplus(
                beta * (z_norm - self.latent_radius)
            )
            z_projected = z * projected_norm / (z_norm + 1e-8)
        else:
            z_projected = z
        return self.decode(z_projected, state_batch)


def conditional_geometric_regularization_loss(model, z_batch, state_batch, alpha=1.0):
    """Jacobian-determinant variance penalty for the conditional decoder."""
    if not z_batch.requires_grad:
        z_batch.requires_grad_(True)
    log_det_values = []
    n = min(z_batch.size(0), 32)

    for i in range(n):
        z_i = z_batch[i:i+1]
        s_i = state_batch[i:i+1]
        decoded_i = model.decode(z_i, s_i)

        jacobian_rows = []
        for j in range(decoded_i.size(1)):
            grad_j = torch.autograd.grad(
                outputs=decoded_i[0, j],
                inputs=z_i,
                grad_outputs=torch.ones_like(decoded_i[0, j]),
                retain_graph=True,
                create_graph=True,
            )[0]
            jacobian_rows.append(grad_j)
        J = torch.stack(jacobian_rows, dim=0).squeeze(1)
        gram = J @ J.T + 1e-6 * torch.eye(J.size(0), device=J.device)
        det = torch.det(gram)
        if torch.isfinite(det) and det > 0:
            log_det_values.append(torch.log(torch.clamp(det, min=1e-12, max=1e12)))

    if len(log_det_values) == 0:
        return torch.tensor(0.0, device=z_batch.device, dtype=z_batch.dtype)

    log_dets = torch.stack(log_det_values)
    log_dets = torch.nan_to_num(log_dets, nan=0.0, posinf=0.0, neginf=0.0)
    return alpha * torch.var(log_dets)