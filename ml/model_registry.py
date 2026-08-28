import os
import sys
sys.path.insert(0, ".")

import json
import shutil
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger("ml.model_registry")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

MODELS_DIR = "models"
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")
CHAMPION_DIR = os.path.join(MODELS_DIR, "production", "champion")
CHALLENGER_DIR = os.path.join(MODELS_DIR, "production", "challenger")

def load_registry() -> Dict[str, Any]:
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load registry ({e}). Re-initializing.")
    return {"models": {}, "champion": None, "last_updated": datetime.utcnow().isoformat()}


def save_registry(registry: Dict[str, Any]):
    os.makedirs(MODELS_DIR, exist_ok=True)
    registry["last_updated"] = datetime.utcnow().isoformat()
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    logger.info(f"Saved model registry to {REGISTRY_PATH}")


def register_model(model_name: str, file_path: str, metrics: Dict[str, Any],
                   version: str = "10.0.0", status: str = "challenger") -> Dict[str, Any]:
    """
    Registers a trained model bundle into models/registry.json and copies to champion/challenger directory.
    """
    os.makedirs(CHAMPION_DIR, exist_ok=True)
    os.makedirs(CHALLENGER_DIR, exist_ok=True)
    
    registry = load_registry()
    
    target_dir = CHAMPION_DIR if status == "champion" else CHALLENGER_DIR
    target_file = os.path.join(target_dir, f"{model_name}.pkl")
    
    if os.path.exists(file_path):
        shutil.copy(file_path, target_file)
        
    model_entry = {
        "model_name": model_name,
        "version": version,
        "status": status,
        "metrics": metrics,
        "dataset_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "registered_at": datetime.utcnow().isoformat(),
        "bundle_path": target_file
    }
    
    registry["models"][model_name] = model_entry
    if status == "champion":
        registry["champion"] = model_name
        
    save_registry(registry)
    logger.info(f"Registered model '{model_name}' with status '{status}'")
    return model_entry


def promote_challenger_to_champion(model_name: str, min_acc_gain: float = 0.005) -> bool:
    """
    Promotes challenger model to champion if its accuracy exceeds current champion by min_acc_gain.
    """
    registry = load_registry()
    models = registry.get("models", {})
    
    if model_name not in models:
        logger.error(f"Model '{model_name}' not found in registry.")
        return False
        
    challenger = models[model_name]
    curr_champ_name = registry.get("champion")
    
    if curr_champ_name and curr_champ_name != model_name and curr_champ_name in models:
        curr_champ = models[curr_champ_name]
        c_acc = challenger["metrics"].get("accuracy", 0.0)
        p_acc = curr_champ["metrics"].get("accuracy", 0.0)
        
        if c_acc < p_acc + min_acc_gain:
            logger.info(f"Challenger acc ({c_acc:.4f}) does not exceed Champion ({p_acc:.4f}) by threshold {min_acc_gain}. Promotion skipped.")
            return False
            
        curr_champ["status"] = "archived"
        
    challenger["status"] = "champion"
    registry["champion"] = model_name
    
    # Copy bundle to champion directory
    if os.path.exists(challenger["bundle_path"]):
        champ_bundle_path = os.path.join(CHAMPION_DIR, f"{model_name}.pkl")
        shutil.copy(challenger["bundle_path"], champ_bundle_path)
        challenger["bundle_path"] = champ_bundle_path
        
    save_registry(registry)
    logger.info(f"Successfully promoted model '{model_name}' to CHAMPION!")
    return True

if __name__ == "__main__":
    register_model("match_winner_v1", "models/checkpoints/match_winner_v1.pkl", {"accuracy": 0.74, "ece": 0.05}, status="champion")
