import torch
import torch.nn as nn
import geoopt

class StiefelGovernor(nn.Module):
    """
    Manifold Governor that projects Euclidean outputs onto the Stiefel Manifold.
    Manifold: Stiefel(n, p) = {X in R^{n x p} : X^T X = I}
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # Standard Linear layer for the mapping
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        
        # Define the Manifold for optimization context or downstream usage
        self.manifold = geoopt.manifolds.Stiefel()

    def forward(self, x):
        # x: Input (Batch, In_Dim)
        z = self.linear(x)
        
        # To project onto Stiefel(n, p), we need a matrix.
        # Treat output as set of column vectors.
        p = 3
        n = z.size(-1) // p
        if n * p != z.size(-1):
             n = z.size(-1)
             p = 1
        
        z_mat = z.view(z.size(0), n, p)
        
        # Constraint: QR Decomposition
        Q, R = torch.linalg.qr(z_mat)
        
        # Q is on Stiefel(n, p) (columns are orthonormal)
        return Q.view(z.size(0), -1)

class FeatureExtractionModule(nn.Module):
    def __init__(self, in_bands, out_features=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_bands, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(64, 64),
            nn.Conv2d(64, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features)
        )
    def forward(self, x):
        return self.net(x)

class RiemannianTangentBridge(nn.Module):
    def __init__(self, input_dim, subspace_dim):
        """
        Args:
            input_dim (D): Spectral dimension.
            subspace_dim (k): Subspace dimension from Step 1.
        """
        super().__init__()
        self.D = input_dim
        self.k = subspace_dim
        
        # The learnable Mean Subspace 'M'.
        # We initialize it randomly, but we must ensure it stays orthonormal.
        # We store 'M_raw' and orthonormalize it on the fly.
        self.M_raw = nn.Parameter(torch.randn(input_dim, subspace_dim))
        
    def get_mean_subspace(self):
        # Enforce orthogonality using QR decomposition
        M_orth, _ = torch.linalg.qr(self.M_raw)
        return M_orth

    def log_map(self, X, Y):
        """
        Computes the Riemannian Log_X(Y) on the Grassmannian.
        X: Base point (The Mean M), shape (D, k)
        Y: Target point (Input W), shape (Batch, D, k)
        """
        # 1. Compute M^T @ W (Shape: Batch, k, k)
        # We need to broadcast X to match Y's batch dimension
        XtY = torch.matmul(X.T, Y) 
        
        # 2. Check for singularity (if subspaces are orthogonal, inverse explodes)
        # In practice, adding a tiny jitter to the diagonal helps stability.
        jitter = 1e-6 * torch.eye(self.k, device=X.device).unsqueeze(0)
        XtY_inv = torch.linalg.inv(XtY + jitter)
        
        # 3. Compute the "Projected" Y that aligns with X
        # Y_proj = Y @ (X^T Y)^-1
        Y_proj = torch.matmul(Y, XtY_inv)
        
        # 4. Compute the difference matrix E (Tangent direction pre-scaling)
        # Note: X needs to broadcast to (Batch, D, k)
        E = Y_proj - X.unsqueeze(0)
        
        # 5. Compute SVD of E to get principal angles
        # U: (Batch, D, k), S: (Batch, k), Vh: (Batch, k, k)
        U, S, Vh = torch.linalg.svd(E, full_matrices=False)
        
        # 6. Apply Arctan to singular values (Geodesic scaling)
        # reconstruct U @ diag(arctan(S)) @ Vh
        
        Theta = torch.arctan(S)
        
        # Reconstruct the tangent vector: Delta = U @ diag(Theta) @ Vh
        # We construct a diagonal matrix from Theta
        Theta_diag = torch.zeros_like(Vh)
        Theta_diag.diagonal(dim1=-2, dim2=-1).copy_(Theta)
        
        Delta = U @ Theta_diag @ Vh
        
        return Delta

    def forward(self, W):
        """
        Args:
            W: Batch of orthonormal matrices from Step 1. Shape (Batch, D, k)
        Returns:
            v: Flattened tangent vectors. Shape (Batch, D*k)
        """
        # 1. Get the current orthonormal Mean
        M = self.get_mean_subspace()
        
        # 2. Compute Log_M(W)
        tangent_matrix = self.log_map(M, W)
        
        # 3. Flatten for the next stage (Quantum Network or MLP)
        v = tangent_matrix.reshape(W.shape[0], -1)
        
        return v
