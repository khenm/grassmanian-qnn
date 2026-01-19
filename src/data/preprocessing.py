import torch
from einops import rearrange
import glob
import multiprocessing
import numpy as np
import rasterio
import os
import tarfile

# Constants for HySpecNet-11k processing
INVALID_CHANNELS = [126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 160, 161, 162, 163, 164, 165, 166]
VALID_CHANNELS_IDS = [c + 1 for c in range(224) if c not in INVALID_CHANNELS]
MINIMUM_VALUE = 0
MAXIMUM_VALUE = 10000

def process_patch(patch_3d: torch.Tensor, n_components: int = 3) -> torch.Tensor:
    """
    Process a 3D patch to extract its principal subspace using SVD.

    Args:
        patch_3d (torch.Tensor): Input patch of shape (H, W, D), typically (Height, Width, Spectral).
        n_components (int): Number of principal components to keep (subspace dimension).

    Returns:
        torch.Tensor: Orthonormal basis of the subspace (U_init) with shape (H*W, n_components).
                      This represents the spatial subspace.
    """
    X = rearrange(patch_3d, 'h w d -> (h w) d')
    
    # We use full_matrices=False for efficiency
    U, S, V = torch.linalg.svd(X, full_matrices=False)
    
    if U.shape[1] < n_components:
        raise ValueError(f"Patch pixels ({U.shape[1]}) fewer than requested components ({n_components}). Increase patch size.")
        
    U_init = U[:, :n_components] 
    

    return U_init

def convert_patch(patch_path: str) -> None:
    """
    Convert a single hyperspectral patch from TIF to npy format.
    
    Args:
        patch_path (str): Path to the SPECTRAL_IMAGE.TIF file.
    """
    try:
        with rasterio.open(patch_path) as dataset:
            src = dataset.read(VALID_CHANNELS_IDS)
            
        clipped = np.clip(src, a_min=MINIMUM_VALUE, a_max=MAXIMUM_VALUE)
        
        # Normalize to [0, 1]
        out_data = (clipped - MINIMUM_VALUE) / (MAXIMUM_VALUE - MINIMUM_VALUE)
        out_data = out_data.astype(np.float32)
        
        out_path = patch_path.replace("SPECTRAL_IMAGE", "DATA").replace("TIF", "npy")
        np.save(out_path, out_data)
        
    except Exception as e:
        print(f"Error processing {patch_path}: {e}")


def generate_hyspecnet_data(in_directory: str = "./hyspecnet-11k/patches/", num_workers: int = 64) -> None:
    """
    Generate .npy data files for the HySpecNet-11k dataset.
    
    Args:
        in_directory (str): Root directory of the patches.
        num_workers (int): Number of multiprocessing workers.
    """
    if not in_directory.endswith("/"):
        in_directory += "/"
        
    print(f"Scanning for TIF files in {in_directory}...")
    in_patches = glob.glob(f"{in_directory}**/**/*SPECTRAL_IMAGE.TIF", recursive=True)
    
    if not in_patches:
        print(f"No TIF files found in {in_directory}")
        return

    print(f"Found {len(in_patches)} files. Processing with {num_workers} workers...")
    
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(convert_patch, in_patches)
        
    print("Processing complete.")


def unzip_all(tar_dir: str, out_dir: str, file_name: str = "SPECTRAL_IMAGE.TIF", num_workers: int = 64) -> None:
    """
    Unzip a specific file from all tar files in a directory using multiprocessing.
    
    Args:
        tar_dir (str): Directory containing tar files.
        out_dir (str): Directory to extract to.
        file_name (str): Name of the file to extract inside the tar.
        num_workers (int): Number of workers for multiprocessing.
    """
    # Ensure directory path ends with slash for glob
    if not tar_dir.endswith("/"):
        tar_dir += "/"
        
    print(f"Scanning for tar files in {tar_dir}...")
    tar_files = glob.glob(f"{tar_dir}**/*.tar", recursive=True)
    
    if not tar_files:
        print(f"No tar files found in {tar_dir}")
        return

    print(f"Found {len(tar_files)} tar files. Processing with {num_workers} workers...")
    
    args_list = [(tar_path, file_name, out_dir) for tar_path in tar_files]
    
    with multiprocessing.Pool(num_workers) as pool:
        pool.starmap(unzip_single_file, args_list)
        
    print("Unzip processing complete.")


def unzip_single_file(tar_path: str, file_name: str, out_dir: str) -> None:
    """
    Extract a single file from a tar archive.
    
    Args:
        tar_path (str): Path to the tar file.
        file_name (str): Name of the file to extract inside the tar.
        out_dir (str): Directory to extract the file to.
    """
    try:
        with tarfile.open(tar_path, "r") as tar:
            target_member = None
            
            try:
                target_member = tar.getmember(file_name)
            except KeyError:
                # Search recursively if not found at root
                for member in tar.getmembers():
                    if member.name.endswith(file_name):
                        target_member = member
                        break
            
            if target_member:
                tar.extract(target_member, path=out_dir)
                print(f"Successfully extracted {target_member.name} from {tar_path}")
            else:
                print(f"Warning: {file_name} not found in {tar_path}")

    except tarfile.ReadError:
        print(f"CRITICAL ERROR: Corrupted tar file {tar_path}. Please re-download this file.")
    except Exception as e:
        print(f"Error extracting {file_name} from {tar_path}: {e}")

def prepare_hyspecnet_data(tar_dir: str, root_dir: str, num_workers: int = 64) -> None:
    """
    End-to-end data preparation for HySpecNet-1k.
    1. Extracts TIF files from tars.
    2. Converts TIF -> NPY.
    
    Args:
        tar_dir: Directory containing source .tar files.
        root_dir: Target directory for extraction (dataset root).
        num_workers: Number of parallel workers.
    """
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)
        
    print(f"=== Starting HySpecNet-1k Preparation ===")
    print(f"Source Tars: {tar_dir}")
    print(f"Target Root: {root_dir}")
    
    # 1. Unzip relevant files
    print("\n[Step 1/2] Extracting TIF files...")
    unzip_all(tar_dir, root_dir, file_name="SPECTRAL_IMAGE.TIF", num_workers=num_workers)
    unzip_all(tar_dir, root_dir, file_name="QL_QUALITY_CLASSES.TIF", num_workers=num_workers)
    
    # 2. Convert Data
    print("\n[Step 2/2] Converting Spectral Images to NPY...")
    generate_hyspecnet_data(in_directory=root_dir, num_workers=num_workers)
    
    print("\n=== Data Preparation Complete ===")