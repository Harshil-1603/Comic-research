"""
Reproducibility utilities: seed setting and config logging
"""
import random
import numpy as np
import torch
import json
import os
from datetime import datetime


def seed_all(s=42):
    """Fix random seeds for reproducibility"""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    # Make PyTorch deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_config(config_dict, log_dir="logs"):
    """Log configuration to JSON file"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join(log_dir, f"config_{timestamp}.json")
    
    # Add timestamp to config
    config_dict['timestamp'] = timestamp
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Config logged to {config_path}")
    return config_path


def get_default_config():
    """Get default configuration for experiments"""
    return {
        "seed": 42,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "epochs": 10,
        "model": {
            "d_img": 512,
            "d_txt": 768,
            "d_col": 48,
            "n_cls": 5
        },
        "encoders": {
            "image": "openai/clip-vit-base-patch32",
            "text": "bert-base-uncased"
        },
        "color_features": {
            "bins": 16,
            "use_background_aware": False
        },
        "data": {
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15
        }
    }


if __name__ == "__main__":
    # Test seed setting
    seed_all(42)
    print("Random seeds set for reproducibility")
    
    # Test config logging
    config = get_default_config()
    log_config(config)
