import torch
import torch.nn as nn
import torch.nn.functional as F
import geotorch

class GeometricEncoder(nn.Module):
    def __init__(self, input_dim, subspace_dim):
        """
        Args:
            input_dim (D): Number of spectral bands.
            subspace_dim (k): The dimension of the target subspace (rank).
        """
        super().__init__()
        self.D = input_dim
        self.k = subspace_dim
        
        # The learnable basis matrix W (D x k)
        self.W = nn.Parameter(torch.empty(input_dim, subspace_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize W as an orthogonal matrix
        nn.init.orthogonal_(self.W)

    def forward(self, x):
        """
        Args:
            x: Input tensor shape (Batch, D)
        Returns:
            z: Latent coordinates shape (Batch, k)
            x_recon: Reconstructed input shape (Batch, D)
        """
        # 1. Project onto subspace (Encoder)
        # z = x @ W
        z = x @ self.W
        
        # 2. Reconstruct from subspace (Decoder)
        # x_hat = z @ W.T
        x_recon = z @ self.W.T
        
        return z, x_recon

    def get_orthogonality_loss(self):
        """
        Computes L_orth = || W.T @ W - I ||^2_F
        This forces the columns of W to be orthonormal.
        """
        # W.T @ W should be Identity(k)
        gram_matrix = self.W.T @ self.W
        identity = torch.eye(self.k, device=self.W.device)
        
        # Frobenius norm squared
        loss = torch.norm(gram_matrix - identity, p='fro') ** 2
        return loss


class GeometricEncoderRiemannian(nn.Module):
    def __init__(self, input_dim, subspace_dim):
        super().__init__()
        self.D = input_dim
        self.k = subspace_dim
        
        # create a standard Linear layer
        self.linear = nn.Linear(input_dim, subspace_dim, bias=False)
        
        # constrain the weight matrix to be on the Grassmannian
        geotorch.orthogonal(self.linear, "weight", columns=subspace_dim)

    def forward(self, x):
        # The weight is guaranteed to be orthogonal (W.T @ W = I)
        
        # 1. Project onto subspace (Encoder)
        # z = x @ W
        z = self.linear(x)
        
        # 2. Reconstruct (Decoder)
        # the reconstruction is simply z @ W.T since W is orthogonal
        W = self.linear.weight
        x_recon = z @ W.T
        
        return z, x_recon
