import torch
import numpy as np
import scipy.io
import pandas as pd
import os
import rasterio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .registry import DataFactory, HSIDatasetConfig
from .preprocessing import process_patch
import logging

log = logging.getLogger(__name__)

@DataFactory.register("IndianPines")
class IndianPinesDataset(Dataset):
    def __init__(self, cfg: HSIDatasetConfig, test_mode=False):
        self.cfg = cfg
        self.test_mode = test_mode
        self.data, self.labels = self._load_data()
        
        # Split logic: Standard random split of pixels with labels.
        
        # Filter out background (label 0 usually)
        valid_indices = np.argwhere(self.labels > 0)
        
        X_idx = valid_indices
        y = self.labels[valid_indices[:, 0], valid_indices[:, 1]]
        
        # Stratified Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_idx, y, 
            train_size=cfg.train_split, 
            stratify=y, 
            random_state=42
        )
        
        self.indices = X_test if test_mode else X_train
        self.labels_subset = y_test if test_mode else y_train
        
    def _load_data(self):
        # Load .mat files
        
        try:
            data_mat = scipy.io.loadmat(self.cfg.data_path)
            # Keys usually 'indian_pines_corrected' or similar
            key_data = [k for k in data_mat if not k.startswith('_') and data_mat[k].ndim == 3][0]
            data = data_mat[key_data]
            
            # For GT
            gt_path = self.cfg.data_path.replace("corrected", "gt")
            gt_mat = scipy.io.loadmat(gt_path)
            key_gt = [k for k in gt_mat if not k.startswith('_') and gt_mat[k].ndim == 2][0]
            labels = gt_mat[key_gt]
            
            return data, labels
            
        except Exception as e:
            log.error(f"Failed to load Indian Pines data: {e}")
            # Return dummy for dev if file missing
            log.warning("Generating DUMMY data for verification.")
            return np.random.rand(145, 145, 200), np.random.randint(0, 17, (145, 145))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get pixel coordinate
        r, c = self.indices[idx]
        
        # Extract patch with boundary checks
        pad = self.cfg.patch_size // 2
        
        img_H, img_W, _ = self.data.shape
        
        r_start = max(0, r - pad)
        r_end = min(img_H, r + pad + 1)
        c_start = max(0, c - pad)
        c_end = min(img_W, c + pad + 1)
        
        patch = self.data[r_start:r_end, c_start:c_end, :]
        
        # Pad if near edge to ensure fixed size
        if patch.shape[0] != self.cfg.patch_size or patch.shape[1] != self.cfg.patch_size:
            # simple padding (reflection or zero)
            pass
            
        # If strict size needed:
        full_patch = np.zeros((self.cfg.patch_size, self.cfg.patch_size, self.data.shape[2]), dtype=self.data.dtype)
        
        # Centering logic
        h_len = patch.shape[0]
        w_len = patch.shape[1]
        full_patch[:h_len, :w_len, :] = patch
        
        # Convert to Tensor
        patch_tensor = torch.from_numpy(full_patch).float()
        
        # Preprocessing (SVD)
        # Output: (Pixels, Components)
        u_init = process_patch(patch_tensor, n_components=self.cfg.components)
        
        # Label (0-indexed for CrossEntropy usually, original 1-16)
        label = self.labels_subset[idx] - 1 
        # Note: labels in Indian Pines are 1-16 usually. 0 is background (filtered out).
        
        return u_init, torch.tensor(label, dtype=torch.long)

@DataFactory.register("HyspecNet1k")
class HySpecNetDataset(Dataset):
    """
    Dataset class for HySpecNet-11k.
    
    Args:
        cfg (HSIDatasetConfig): Configuration object containing dataset parameters.
        test_mode (bool): If True, loads the test split. Defaults to False.
    """
    def __init__(self, cfg: HSIDatasetConfig, test_mode=False):
        self.cfg = cfg
        self.test_mode = test_mode
        self.root = cfg.root_dir if cfg.root_dir else cfg.data_path
        
        # Determine split file
        split_name = "test.csv" if test_mode else "train.csv"
        
        # Navigate to the correct split directory
        # Priority: configured root -> parent of root -> assumed standard structure
        possible_paths = [
            os.path.join(self.root, "splits", cfg.split_mode, split_name),
            os.path.join(self.root, "..", "splits", cfg.split_mode, split_name)
        ]
        
        split_path = None
        for path in possible_paths:
            if os.path.exists(path):
                split_path = path
                break
        
        if split_path is None:
             # Fallback to standard structure assumption
            split_path = os.path.join(self.root, "splits", cfg.split_mode, split_name)
            
        try:
            self.df = pd.read_csv(split_path)
        except Exception as e:
            log.error(f"Failed to load split {split_path}: {e}")
            self.df = pd.DataFrame() 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the item to load.
            
        Returns:
            tuple: (u_init, label_tensor)
                - u_init (torch.Tensor): Processed hyperspectral data (subspace basis).
                - label_tensor (torch.Tensor): Label/Segmentation map.
        """
        row = self.df.iloc[idx]
        
        # Assumption: First column contains the relative path to the patch
        patch_rel_path = row.iloc[0] 
        
        # Construct full path
        if "patches" not in self.root:
             patch_dir = os.path.join(self.root, "patches", patch_rel_path)
        else:
             patch_dir = os.path.join(self.root, patch_rel_path)
        
        patch_name = os.path.basename(patch_rel_path) 
        
        # Define file paths
        data_path = os.path.join(patch_dir, f"{patch_name}-DATA.npy")
        label_path = os.path.join(patch_dir, f"{patch_name}-QL_QUALITY_CLASSES.TIF")
        
        # Load Data
        if os.path.exists(data_path):
            try:
                data = np.load(data_path) 
                data_tensor = torch.from_numpy(data).float()
                u_init = process_patch(data_tensor, n_components=self.cfg.components)
            except Exception as e:
                log.error(f"Error loading data at {data_path}: {e}")
                u_init = torch.zeros((self.cfg.patch_size ** 2, self.cfg.components))
        else:
            log.warning(f"Data file not found: {data_path}")
            u_init = torch.zeros((self.cfg.patch_size ** 2, self.cfg.components)) 
            
        # Load Label
        if os.path.exists(label_path):
            try:
                with rasterio.open(label_path) as src:
                    label_map = src.read(1) 
                label_tensor = torch.from_numpy(label_map.flatten()).long()
            except Exception as e:
                log.error(f"Error loading label at {label_path}: {e}")
                label_tensor = torch.zeros(u_init.shape[0], dtype=torch.long)
        else:
            label_tensor = torch.zeros(u_init.shape[0], dtype=torch.long)

        return u_init, label_tensor

