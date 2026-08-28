"""
v8_patch_analyzer.py
--------------------
Central Pipeline Driver for the v8 Differentiable Patch Analyzer.

Pipes data sequentially from Phase 1 through Phase 4:
1. Phase 1 (v8_patch_parser.py): Schema-driven wikitext NLP extraction.
2. Phase 2 (v8_differentiable_base.py): Category embeddings & dynamic attention gating.
3. Phase 3 (v8_breakpoint_thresholds.py): Straight-Through Estimator breakpoint thresholding.
4. Phase 4 (v8_copula_aggregation.py): Archimedean Gumbel Copula synergistic aggregation.

Outputs the final Concept Drift scores to data/processed/automated_patch_nerf_registry.json.
"""

import os
import json
import logging
from typing import Dict, List, Any
import torch

from v8_patch_parser import V8PatchParser, PatchExtractionPayload
from v8_differentiable_base import PatchEmbeddingBase, PatchTensorBuilder
from v8_breakpoint_thresholds import BreakpointShockEvaluator
from v8_copula_aggregation import AgentGroupedCopulaAggregator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("v8_patch_analyzer")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
CACHE_PATCHES_DIR = os.path.join(DATA_DIR, "patches")
RAW_WIKI_DIR = os.path.join(DATA_DIR, "raw", "wiki_patches")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
REGISTRY_PATH = os.path.join(PROCESSED_DIR, "automated_patch_nerf_registry.json")
BACKUP_REGISTRY_PATH = os.path.join(PROCESSED_DIR, "patch_nerf_registry.json")


def safe_float(val: Any, default: float = 1.0) -> float:
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
        # Strip trailing units like 's', '%', 'hp'
        s = str(val).strip().rstrip("%sS")
        return float(s)
    except (ValueError, TypeError):
        return default


def run_v8_patch_pipeline() -> Dict[str, Dict[str, float]]:
    """
    Executes the full Phase 1-4 pipeline across all cached wikitext patches and
    writes the automated_patch_nerf_registry.json output.
    """
    logger.info("=== STARTING V8 DIFFERENTIABLE PATCH ANALYZER PIPELINE ===")

    # Initialize Phase 1 - 4 Modules
    v8_parser = V8PatchParser()
    phase2_model = PatchEmbeddingBase(embed_dim=8, context_dim=16)
    breakpoint_evaluator = BreakpointShockEvaluator(threshold=150.0, mode="ste")
    copula_aggregator = AgentGroupedCopulaAggregator(init_theta=1.5)

    # Collect patch files (.wiki in data/patches/ or .txt in data/raw/wiki_patches/)
    patch_sources: Dict[str, str] = {}

    if os.path.exists(CACHE_PATCHES_DIR):
        for fname in os.listdir(CACHE_PATCHES_DIR):
            if fname.endswith(".wiki"):
                version = fname.replace(".wiki", "")
                fpath = os.path.join(CACHE_PATCHES_DIR, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    patch_sources[version] = f.read()

    if os.path.exists(RAW_WIKI_DIR):
        for fname in os.listdir(RAW_WIKI_DIR):
            if fname.endswith(".txt") or fname.endswith(".wiki"):
                version = fname.replace(".txt", "").replace(".wiki", "").replace("patch_", "")
                fpath = os.path.join(RAW_WIKI_DIR, fname)
                if version not in patch_sources:
                    with open(fpath, "r", encoding="utf-8") as f:
                        patch_sources[version] = f.read()

    # Fallback mock patches if no local files exist
    if not patch_sources:
        logger.warning("No patch files found in local cache. Loading default V8 benchmark patch notes.")
        patch_sources["9.0"] = """== Agent Updates ==
=== Iso ===
* Double Tap: duration decreased from 20 >>> 12
"""
        patch_sources["9.02"] = """== Agent Updates ==
=== Neon ===
* High Gear: speed multiplier increased from 1.0 >>> 1.1
=== Jett ===
* Cloudburst: duration decreased from 4.5 >>> 2.5
* Tailwind: dash windup delay increased from 0.0 >>> 0.75
"""

    automated_nerf_registry: Dict[str, Dict[str, float]] = {}

    for version in sorted(patch_sources.keys()):
        wikitext = patch_sources[version]
        logger.info(f"Processing Patch Version {version} through V8 Pipeline...")

        # Phase 1: Schema-Driven NLP Extraction
        payload: PatchExtractionPayload = v8_parser.parse_wikitext(wikitext, version=version)
        changes = [item.model_dump() for item in payload.changes]

        if not changes:
            logger.info(f"Patch {version}: No agent balance changes detected.")
            automated_nerf_registry[version] = {}
            continue

        # Convert changes to Phase 2 tensor format
        tensors = PatchTensorBuilder.payload_to_tensors(changes)

        # Phase 2: Differentiable Base & Dynamic Attention Gating
        phase2_out = phase2_model(tensors)
        gated_shocks = phase2_out["gated_shocks"]

        # Phase 3: Straight-Through Estimator Breakpoint Thresholding
        x_old_list = []
        x_new_list = []
        agent_names = []

        for c in changes:
            agent_names.append(c.get("agent", "Unknown"))
            old_val = c.get("old_value")
            new_val = c.get("new_value")

            # Fallback for non-numeric strings or mechanical removals
            x_old_list.append(safe_float(old_val, default=1.0))
            x_new_list.append(safe_float(new_val, default=0.5))

        x_old = torch.tensor(x_old_list, dtype=torch.float32).unsqueeze(1)
        x_new = torch.tensor(x_new_list, dtype=torch.float32).unsqueeze(1)

        phase3_out = breakpoint_evaluator(x_old, x_new, gated_shocks)
        fused_shocks = phase3_out["fused_shocks"]

        # Phase 4: Copula-Based Synergistic Aggregation
        drift_by_agent = copula_aggregator(agent_names, fused_shocks)

        # Store rounded scalar Concept Drift scores
        automated_nerf_registry[version] = {}
        for agent, drift_tensor in drift_by_agent.items():
            drift_score = round(float(drift_tensor.item()), 4)
            automated_nerf_registry[version][agent] = drift_score
            logger.info(f"  -> Patch {version} | Agent {agent}: Concept Drift = {drift_score:.4f}")

    # Write output to registry paths
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(automated_nerf_registry, f, indent=4)
    logger.info(f"Successfully saved V8 Concept Drift Registry to {REGISTRY_PATH}")

    with open(BACKUP_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(automated_nerf_registry, f, indent=4)
    logger.info(f"Successfully updated backup registry at {BACKUP_REGISTRY_PATH}")

    logger.info("=== V8 DIFFERENTIABLE PATCH ANALYZER COMPLETED SUCCESSFULLY ===")
    return automated_nerf_registry


if __name__ == "__main__":
    run_v8_patch_pipeline()
