import torch
import torch.nn as nn
from .registry import ModelFactory
from .geometric import GeometricEncoder, GeometricEncoderRiemannian
from .manifold import StiefelGovernor, RiemannianTangentBridge
from .quantum import get_quantum_layer, QuantumVariationalLayer

@ModelFactory.register("stiefel_quantum")
class HybridModel(nn.Module):
    """
    Hybrid Quantum-Classical Model for Hyperspectral Image Classification.
    
    Architecture:
        Input (SVD Basis) -> StiefelGovernor (Manifold Layer) -> Bridge -> Quantum Circuit -> Classifier
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Calculate input dimension from flattened patch
        self.input_dim = (cfg.dataset.patch_size ** 2) * cfg.dataset.components
        
        # Stiefel Governor setup
        self.governor_dim = cfg.model.get('governor_dim', 16)
        self.governor_layer = StiefelGovernor(self.input_dim, self.governor_dim)
        
        # Bridge layer: Projects manifold features to quantum circuit inputs
        # We output 3 scalar values to control the quantum circuit parameters
        self.bridge = nn.Linear(self.governor_dim, 3)
        
        # Quantum Layer
        self.quantum = get_quantum_layer(
            n_wires=cfg.model.get('quantum_wires', 3), 
            n_layers=cfg.model.get('quantum_layers', 1), 
            force_cpu=cfg.model.get('force_cpu', (cfg.device == 'cpu'))
        )
        
        # Classifier
        # Input: 3 quantum measurements (expectations)
        # Output: Class logits
        num_classes = 16
        self.classifier = nn.Sequential(
            nn.Linear(3, num_classes),
        )

    def forward(self, x):
        """
        Forward pass of the hybrid model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Pixels, Components)
            
        Returns:
            torch.Tensor: Classification logits
        """
        b = x.size(0)
        x_flat = x.view(b, -1)
        
        # 1. Manifold Governance: Project onto Stiefel Manifold
        q_stiefel = self.governor_layer(x_flat)
        
        # 2. Bridge: Map to quantum parameters
        q_params = self.bridge(q_stiefel)
        
        # 3. Quantum Circuit: Process through parameterized quantum layer
        q_out = self.quantum(q_params)
        
        # 4. Classification
        logits = self.classifier(q_out)
        
        return logits


@ModelFactory.register("grassmann_vqe")
class GrassmannVQEModel(nn.Module):
    """
    Grassmannian VQE Model for Hyperspectral Image Application.
    
    Architecture:
        1. Input Patch -> StiefelGovernor (Mapping to Manifold)
        2. Manifold Point -> RiemannianTangentBridge (Log Map to Tangent Space)
        3. Tangent Vector -> QuantumVariationalLayer (VQE Circuit)
        4. Quantum State -> Classifier
    """
    def __init__(self, input_dim, subspace_dim, n_qubits, n_q_layers, num_classes):
        super().__init__()
        
        # Geometric Encoder (Subspace Extraction)
        # Acts as a "Local SVD" extractor returning a subspace U (D x k)
        self.geo_encoder = GeometricEncoderRiemannian(input_dim, subspace_dim)
        
        # Riemannian Tangent Bridge
        # Maps the curved manifold point U onto the flat tangent space of Mean M
        self.bridge = RiemannianTangentBridge(input_dim, subspace_dim)
        
        # Quantum VQC
        # Input dim is D*k because we flattened the tangent matrix
        self.q_layer = QuantumVariationalLayer(input_dim=input_dim * subspace_dim, 
                                               n_qubits=n_qubits, 
                                               n_layers=n_q_layers)
        
        # Classifier Head
        self.classifier = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        """
        Forward pass for GrassmannVQEModel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, D) or (Batch, D, N_neighbors).
                              Representation depends on whether using 1D line or k-dim subspace.

        Returns:
            tuple: (logits, reconstruction)
        """
        # 1. Extract Geometry
        # z: coordinates, x_rec: reconstruction, 
        # W_local: The subspace basis for this input
        z, x_recon, W_local = self.geo_encoder(x)
        
        # 2. Apply Riemannian Tangent Bridge
        # Input: (Batch, D, k) -> Output: (Batch, D*k) flat vector
        tangent_vector = self.bridge(W_local) 
        
        # 3. Quantum Processing
        q_features = self.q_layer(tangent_vector)
        
        # 4. Classification
        logits = self.classifier(q_features)
        
        return logits, x_recon