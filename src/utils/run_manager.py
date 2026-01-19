import os
import torch
import yaml
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf, DictConfig
import logging

log = logging.getLogger(__name__)

class RunManager:
    def __init__(self, cfg: DictConfig, mode='train', resume_path=None):
        self.cfg = cfg
        self.mode = mode
        self.best_score = float('-inf')
        
        if resume_path:
            self._resume(resume_path)
        else:
            self._create_new_run()

    def _create_new_run(self):
        """Create a new run directory with a timestamp and save the configuration."""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.run_dir = Path(f"runs/{timestamp}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration for reproducibility
        with open(self.run_dir / "config.yaml", "w") as f:
            OmegaConf.save(self.cfg, f)

    def _resume(self, path):
        """Resume from an existing run directory."""
        self.run_dir = Path(path)
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Resume path {path} does not exist.")
            
        # Load previous configuration to ensure consistency
        with open(self.run_dir / "config.yaml", "r") as f:
            self.cfg = OmegaConf.load(f)

    def save_ckpt(self, model, optimizer, epoch, loss=None):
        ckpt = {
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'best_score': self.best_score
        }
        torch.save(ckpt, self.run_dir / "last.pt")
        # Optional: Save best or specific epochs if needed in the future

    def save_best_ckpt(self, model, optimizer, epoch, score, mode='max'):
        is_best = False
        if mode == 'max':
            if score > self.best_score:
                is_best = True
        elif mode == 'min':
            if score < self.best_score:
                is_best = True
                
        if is_best:
            self.best_score = score
            log.info(f"New best score: {score:.4f}. Saving best.pt...")
            ckpt = {
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'epoch': epoch,
                'score': score
            }
            torch.save(ckpt, self.run_dir / "best.pt")

    def load_last_ckpt(self, model, optimizer):
        ckpt_path = self.run_dir / "last.pt"
        if ckpt_path.exists():
            print(f"Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optim'])
            self.best_score = ckpt.get('best_score', float('-inf'))
            return ckpt['epoch'] + 1 # Start from next epoch
        return 0

    @staticmethod
    def get_latest_run_dir(runs_root="runs"):
        """Find the latest run directory in the runs_root folder."""
        root = Path(runs_root)
        if not root.exists():
            return None
        
        # Get all subdirectories in runs/
        subdirs = [d for d in root.iterdir() if d.is_dir()]
        if not subdirs:
            return None
            
        # Sort by creation time (or name which is timestamp)
        # Since names are YYYY-MM-DD_HH-MM-SS, sorting by name is robust
        sorted_subdirs = sorted(subdirs, key=lambda x: x.name, reverse=True)
        
        for d in sorted_subdirs:
            if (d / "last.pt").exists():
                return d
                
        return None
