# File: src/utils/config_loader.py

import yaml
import os
from typing import Dict

def load_config(config_path: str = "configs/main_config.yaml") -> Dict:
    """
    Loads the main project configuration file.

    Args:
        config_path: Path to the main YAML configuration file.

    Returns:
        A dictionary containing the project configuration.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration loaded successfully from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ CONFIGURATION ERROR: The file '{config_path}' was not found.")
        print("   Please ensure you have a 'configs/main_config.yaml' file in your project root.")
        raise
    except Exception as e:
        print(f"❌ CONFIGURATION ERROR: An error occurred while parsing '{config_path}': {e}")
        raise