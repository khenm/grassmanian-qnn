import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def get_rgb_from_hsi(hsi_image, bands=(29, 19, 9)):
    """
    Creates a False Color Composite (RGB) from Hyperspectral Data.
    Args:
        hsi_image: Numpy array (H, W, D)
        bands: Tuple of 3 indices for (R, G, B). Defaults are typical for Indian Pines.
    """
    if hsi_image.ndim == 3:
        r = hsi_image[:, :, bands[0]]
        g = hsi_image[:, :, bands[1]]
        b = hsi_image[:, :, bands[2]]
        rgb = np.dstack((r, g, b))
    else:
        # If flattened or single pixel
        return hsi_image
        
    # Normalize to [0, 1] for display
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    return rgb

def plot_spectral_reconstruction(model, x_input, band_wavelengths=None):
    """
    Compares the original spectral signature vs. the Geometric Encoder's reconstruction.
    Useful to verify if the Grassmannian projection preserves signal integrity.
    
    Args:
        model: Trained GrassmannVQEModel
        x_input: A batch of pixels (Batch, D)
    """
    model.eval()
    with torch.no_grad():
        # Forward pass to get reconstruction
        _, x_recon = model(x_input)
        
    x_in_np = x_input.cpu().numpy()
    x_rec_np = x_recon.cpu().numpy()
    
    # Plot first 3 samples in the batch
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i in range(3):
        ax = axes[i]
        if i >= len(x_in_np): break
        
        ax.plot(x_in_np[i], label='Original Input', color='black', alpha=0.7)
        ax.plot(x_rec_np[i], label='Manifold Reconstruction', color='red', linestyle='--')
        
        ax.set_title(f"Pixel Sample {i+1}")
        ax.set_xlabel("Spectral Band Index")
        ax.set_ylabel("Reflectance / Intensity")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("Geometric Encoder: Spectral Reconstruction Quality", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_latent_manifold(model, data_loader, device='cpu', max_samples=1000):
    """
    Visualizes the learned Riemannian Tangent Space using t-SNE.
    This shows if the manifold learning actually separates classes geometrically.
    
    Args:
        model: The full pipeline.
        data_loader: Loader returning (x, label).
    """
    model.eval()
    latents = []
    labels = []
    
    print("Extracting latent features for t-SNE...")
    
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            x = x.to(device)
            
            # 1. Get Geometric Features (Step 1 output)
            # z is the coordinate on the subspace
            z, _, _ = model.geo_encoder(x)
            
            # Optionally: Get Quantum Features (Step 3 output)
            # q_features = model.q_layer(z)
            
            latents.append(z.cpu().numpy())
            labels.append(y.numpy())
            
            if len(latents) * x.shape[0] > max_samples:
                break
    
    latents = np.concatenate(latents, axis=0)[:max_samples]
    labels = np.concatenate(labels, axis=0)[:max_samples]
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    z_embedded = tsne.fit_transform(latents)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(
        x=z_embedded[:, 0], 
        y=z_embedded[:, 1], 
        hue=labels, 
        palette="viridis", 
        s=60, 
        alpha=0.8,
        legend="full"
    )
    plt.title("Latent Space Manifold (t-SNE Projection)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_classification_maps(ground_truth_map, prediction_map, hsi_image=None):
    """
    Plots the Ground Truth vs. Model Prediction side-by-side.
    
    Args:
        ground_truth_map: (H, W) array of class labels.
        prediction_map: (H, W) array of predicted labels.
        hsi_image: Optional (H, W, D) to show the FCC image alongside.
    """
    num_plots = 3 if hsi_image is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    
    # 1. False Color Composite (if provided)
    if hsi_image is not None:
        rgb = get_rgb_from_hsi(hsi_image)
        axes[0].imshow(rgb)
        axes[0].set_title("False Color Composite (Input)")
        axes[0].axis('off')
        idx_offset = 1
    else:
        idx_offset = 0

    # 2. Ground Truth
    # Mask background (usually label 0)
    masked_gt = np.ma.masked_where(ground_truth_map == 0, ground_truth_map)
    cmap = plt.get_cmap('tab20')
    
    axes[idx_offset].imshow(masked_gt, cmap=cmap, interpolation='none')
    axes[idx_offset].set_title("Ground Truth Map")
    axes[idx_offset].axis('off')
    
    # 3. Prediction
    masked_pred = np.ma.masked_where(prediction_map == 0, prediction_map)
    
    axes[idx_offset + 1].imshow(masked_pred, cmap=cmap, interpolation='none')
    axes[idx_offset + 1].set_title("Quantum-Riemannian Prediction")
    axes[idx_offset + 1].axis('off')
    
    plt.tight_layout()
    plt.show()