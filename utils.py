"""
Utility Module for Hybrid Valorant DFS Micro Engine (v6 - Phase 5).

Provides centralized, error-handled loading functions for YAML configuration and JSON slate payloads.
"""

import os
import json
import logging
from typing import Dict, List, Any
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Safely load engine parameters from central YAML configuration file.
    
    Args:
        config_path (str): Path to config.yaml (default 'config.yaml').
        
    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    from pathlib import Path
    root_dir = Path(__file__).resolve().parent
    config_p = Path(config_path)
    if not config_p.is_absolute():
        config_p = root_dir / config_p
        
    if not config_p.exists():
        logger.error("Config file not found at %s. Returning empty config.", config_p)
        raise FileNotFoundError(f"Configuration file not found: {config_p}")
        
    try:
        with open(config_p, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.debug("Successfully loaded config from %s", config_p)
        return config
    except Exception as e:
        logger.error("Failed to parse config file %s: %s", config_p, e)
        raise e


def load_slate_payload(slate_path: str = "data/processed/current_slate.json") -> List[Dict[str, Any]]:
    """
    Safely load active DFS slate player metadata from JSON payload.
    
    Args:
        slate_path (str): Path to current_slate.json.
        
    Returns:
        List[Dict[str, Any]]: List of player metadata dictionaries.
    """
    from pathlib import Path
    root_dir = Path(__file__).resolve().parent
    slate_p = Path(slate_path)
    if not slate_p.is_absolute():
        slate_p = root_dir / slate_p
        
    if not slate_p.exists():
        logger.error("Slate payload file not found at %s.", slate_p)
        raise FileNotFoundError(f"Slate payload file not found: {slate_p}")
        
    try:
        with open(slate_p, "r", encoding="utf-8") as f:
            slate = json.load(f)
        logger.debug("Successfully loaded %d player records from %s", len(slate), slate_p)
        return slate
    except Exception as e:
        logger.error("Failed to parse slate payload file %s: %s", slate_p, e)
        raise e
