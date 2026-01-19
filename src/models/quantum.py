import torch
import torch.nn as nn
import pennylane as qml
import logging

log = logging.getLogger(__name__)

def get_device(n_wires, force_cpu=False):
    """
    Selects the best available device.
    Prioritizes lightning.gpu if CUDA is available and not forced to CPU.
    """
    if not force_cpu and torch.cuda.is_available():
        try:
            # Try loading lightning.gpu
            dev = qml.device("lightning.gpu", wires=n_wires)
            log.info(f"Initialized PennyLane device: lightning.gpu (wires={n_wires})")
            return dev
        except Exception as e:
            log.warning(f"Failed to load lightning.gpu: {e}. Falling back to default.qubit.")
    
    # Fallback
    dev = qml.device("default.qubit", wires=n_wires)
    log.info(f"Initialized PennyLane device: default.qubit (wires={n_wires})")
    return dev


class QuantumVariationalLayer(nn.Module):
    def __init__(self, input_dim, n_qubits, n_layers, force_cpu=False):
        """
        Args:
            input_dim: Dimension of the tangent vector v.
            n_qubits: Number of qubits to simulate (e.g., 4, 8, or 10).
            n_layers: Depth of the Strongly Entangling Layers.
            force_cpu: If True, force usage of CPU (default.qubit) instead of GPU.
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # 1. Classical Compression Layer
        # Compresses the high-dim tangent vector to match the number of qubits.
        self.compressor = nn.Linear(input_dim, n_qubits)
        
        # 2. Define Quantum Device
        # Use existing helper to pick best device
        self.dev = get_device(n_qubits, force_cpu=force_cpu)
        
        # 3. Create the QNode (The Quantum Circuit)
        # Note: qml.QNode needs to wrap the method
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")
        
        # 4. Initialize Learnable Quantum Weights
        # Shape for StronglyEntanglingLayers: (n_layers, n_qubits, 3)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(self.qnode, weight_shapes)

    def _circuit(self, inputs, weights):
        """
        The internal quantum circuit function.
        inputs: The compressed vector (batch_size, n_qubits)
        weights: Learnable parameters for the ansatz
        """
        # A. Angle Encoding
        qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='Y')
        
        # B. Variational Ansatz (Strongly Entangling Layers)
        qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
        
        # C. Measurement
        # Return expectation value of Pauli-Z on all qubits
        # Output range: [-1, 1] per qubit
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

    def forward(self, v):
        """
        v: Tangent vector. Shape (Batch, input_dim)
        """
        # 1. Compress
        v_compressed = self.compressor(v)
        
        # 2. Squash to [-pi, pi] for optimal rotation encoding
        v_scaled = torch.tanh(v_compressed) * torch.pi 
        
        # 3. Pass through Quantum Circuit
        # q_out shape: (Batch, n_qubits)
        q_out = self.q_layer(v_scaled)
        
        return q_out
