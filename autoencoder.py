"""
    Autoencoder implementation.
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

# def compute_density_loss(z_batch, eps=1e-6):
#     """
#     Compute density loss to encourage uniform distribution in latent space
#     """
#     batch_size = z_batch.size(0)
#     z_expanded = z_batch.unsqueeze(1).repeat(1, batch_size, 1)
#     z_expanded_t = z_batch.unsqueeze(0).repeat(batch_size, 1, 1)
#     dist_matrix = torch.norm(z_expanded - z_expanded_t, dim=2)

#     inf_mask = torch.eye(batch_size, device=device) * 10**12
#     dist_matrix = dist_matrix + inf_mask
#     min_dists = torch.min(dist_matrix, dim=1)[0]

#     # add epsilon for numerical stability
#     density_loss = -torch.mean(torch.log(min_dists + eps))

#     return density_loss