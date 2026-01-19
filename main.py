import argparse
import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
import logging
from torch.utils.data import DataLoader
import os

from src.data.registry import HSIDatasetConfig, DataFactory
from src.data.preprocessing import prepare_hyspecnet_data
from src.utils.run_manager import RunManager
from src.models.registry import ModelFactory
from src.train import train

log = logging.getLogger(__name__)

def load_config(config_path: str, cli_options: list = None) -> DictConfig:
    """
    Load main config, process 'defaults' list to load sub-configs, 
    merge everything, and apply CLI overrides.
    """
    base_conf = OmegaConf.load(config_path)

    defaults = base_conf.get('defaults', [])
    final_cfg = OmegaConf.create()
    
    conf_dir = os.path.dirname(config_path)

    # 1. Identify valid config groups from defaults
    valid_groups = set()
    if defaults:
        for item in defaults:
            if isinstance(item, dict):
                valid_groups.update(item.keys())

    # 2. Separate CLI overrides into group switches vs normal overrides
    group_overrides = {}
    remaining_cli_opts = []
    
    if cli_options:
        for opt in cli_options:
            if '=' in opt:
                key, val = opt.split('=', 1)
                if key in valid_groups:
                    group_overrides[key] = val
                    continue
            remaining_cli_opts.append(opt)
    
    # 3. Load defaults (respecting group overrides)
    if not defaults:
        final_cfg = base_conf
    else:
        for item in defaults:
            if item == "_self_":
                # Merge the content of the main config (excluding 'defaults')
                main_content = OmegaConf.merge(base_conf)
                if 'defaults' in main_content:
                    del main_content['defaults']
                final_cfg = OmegaConf.merge(final_cfg, main_content)
            elif isinstance(item, dict):
                for group, default_name in item.items():
                    # Construct path: conf/group/name.yaml
                    # Use override if present, else default
                    name = group_overrides.get(group, default_name)
                    
                    sub_config_path = os.path.join(conf_dir, group, f"{name}.yaml")
                    if os.path.exists(sub_config_path):
                        sub_conf = OmegaConf.load(sub_config_path)
                        # Merge sub_conf into final_cfg[group]
                        wrapper = OmegaConf.create({group: sub_conf})
                        final_cfg = OmegaConf.merge(final_cfg, wrapper)
                    else:
                        print(f"Warning: Config file not found: {sub_config_path}")
    
    # 4. Apply remaining CLI overrides (dotlist format e.g. ["dataset.batch_size=32"])
    if remaining_cli_opts:
        cli_conf = OmegaConf.from_dotlist(remaining_cli_opts)
        final_cfg = OmegaConf.merge(final_cfg, cli_conf)
        
    return final_cfg

def parse_args():
    parser = argparse.ArgumentParser(description="Grassmanian QNN Training")
    parser.add_argument('--config', type=str, default="conf/config.yaml", help='Path to main config file')
    parser.add_argument('opts', nargs='*', help="Modify config options using the command-line (e.g., dataset.batch_size=16)")
    parser.add_argument('--resume', action='store_true', help='Resume from the latest checkpoint')
    parser.add_argument('--preprocess', action='store_true', help='Run data preparation (unzip/convert) for HyspecNet')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load and compose config
    cfg = load_config(args.config, args.opts)
    
    # Setup Run Manager
    resume_path = None
    if args.resume:
        resume_path = RunManager.get_latest_run_dir()
        if resume_path:
            log.info(f"Resuming from latest run: {resume_path}")
        else:
            log.warning("No previous run found to resume. Starting a new run.")

    run_manager = RunManager(cfg, resume_path=resume_path)
    logging.basicConfig(level=logging.INFO)
    log.info(f"Run initialized at {run_manager.run_dir}")
    
    # Data Module
    if 'dataset' not in cfg:
        raise ValueError("Config missing 'dataset' section. Ensure defaults are loaded correctly.")
        
    dataset_cfg_dict = dict(cfg.dataset)
    ds_config = HSIDatasetConfig(**dataset_cfg_dict)
    
    # Preprocessing Check
    if ds_config.name == "HyspecNet1k":
        if args.preprocess:
             prepare_hyspecnet_data(
                 tar_dir=ds_config.tar_dir, # Config must have this!
                 root_dir=ds_config.root_dir,
                 num_workers=64 # Could param this
             )
        else:
            # Check if root_dir is empty or non-existent
            if not os.path.exists(ds_config.root_dir) or not os.listdir(ds_config.root_dir):
                 log.error(f"Root dir {ds_config.root_dir} is empty or missing. "
                           "Please run with --preprocess to extract and prepare data.")
                 return # Exit safely
    
    # Instantiate Dataset via Factory
    DatasetClass = DataFactory.get_dataset_class(ds_config.name)
    
    train_ds = DatasetClass(ds_config, test_mode=False)
    test_ds = DatasetClass(ds_config, test_mode=True)
    
    train_loader = DataLoader(train_ds, batch_size=ds_config.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=ds_config.batch_size, shuffle=False)
    
    log.info(f"Data Loaded: {ds_config.name} | Train={len(train_ds)}, Test={len(test_ds)}")
    
    # Model
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = ModelFactory.get_model(cfg).to(device)
    log.info(f"Model initialized on {device}")
    
    # Optimizer 
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.lr, 
        weight_decay=cfg.get("weight_decay", 1e-4)
    )
    
    # Resume logic
    start_epoch = run_manager.load_last_ckpt(model, optimizer)
    
    # Loss
    criterion = nn.CrossEntropyLoss()

    # Start Training
    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        run_manager=run_manager,
        start_epoch=start_epoch,
        epochs=cfg.epochs,
        device=device
    )

if __name__ == "__main__":
    main()
