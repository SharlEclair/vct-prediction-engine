import os
import re
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("feature_builder")

PATCHES_PROCESSED_DIR = "./data/processed/patches"
FEATURES_DIR = "./data/processed/features"
REPORTS_DIR = "./data/reports"

# Regex patterns for numeric transitions
# Matches: X >>> Y, X -> Y, X% >>> Y%, X.Y -> A.B
transition_pattern1 = re.compile(
    r'((?:\d+(?:\.\d+)?|\.\d+))\s*(?:s|%|m|cd)?\s*(?:>>>|->)\s*((?:\d+(?:\.\d+)?|\.\d+))\s*(?:s|%|m|cd)?',
    re.IGNORECASE
)
# Matches: increased/decreased from X to Y
transition_pattern2 = re.compile(
    r'(increased|decreased)\s+(?:from\s+)?((?:\d+(?:\.\d+)?|\.\d+))\s*(?:s|%|m|cd)?\s+(?:to|>>>|->)\s*((?:\d+(?:\.\d+)?|\.\d+))\s*(?:s|%|m|cd)?',
    re.IGNORECASE
)

def parse_numeric_transition(description):
    """Extracts (old_val, new_val) from description text using transition patterns."""
    m1 = transition_pattern1.search(description)
    if m1:
        try:
            return float(m1.group(1)), float(m1.group(2))
        except ValueError:
            pass
            
    m2 = transition_pattern2.search(description)
    if m2:
        try:
            return float(m2.group(2)), float(m2.group(3))
        except ValueError:
            pass
            
    return None

def infer_change_type(feature_name, old_val, new_val, initial_type=None):
    """Infers nerf, buff, or adjustment based on feature semantics and direction."""
    if initial_type and initial_type.lower() in ["nerf", "buff", "adjustment", "bugfix"]:
        t = initial_type.lower()
        if t == "bugfix":
            return "adjustment"
        return t
        
    lower_is_better = ["cost", "cooldown", "reload", "windup", "delay", "time", "spread", "drain", "charge_time"]
    name_lower = feature_name.lower()
    is_lower_better = any(x in name_lower for x in lower_is_better)
    
    if old_val is not None and new_val is not None:
        if new_val == old_val:
            return "adjustment"
        if new_val < old_val:
            return "buff" if is_lower_better else "nerf"
        else:
            return "nerf" if is_lower_better else "buff"
            
    return "adjustment"

def get_weight_for_type(change_type):
    t = change_type.lower()
    if t == "nerf":
        return 0.8
    elif t == "adjustment":
        return 0.2
    return 0.0

def map_semantic_feature(extracted_name, description=""):
    """Maps arbitrary text to strict schema categories and features using a priority-aware regex/token mapper."""
    combined = (extracted_name + " " + description).lower()
    
    # Priority 1: Exact explicit phrases (multi-word)
    if "ultimate cost" in combined or "ult points" in combined:
        return "economy", "ultimate_cost"
    if "cast speed" in combined or "windup" in combined or "equip time" in combined:
        return "ability", "cast_time"
    if "weapon reload" in combined or "reload speed" in combined or "reload time" in combined:
        return "combat", "reload"
    if "projectile speed" in combined or "projectile velocity" in combined:
        return "projectile", "velocity"
    if "damage falloff" in combined or "falloff" in combined:
        return "combat", "damage_falloff"
    if "fire rate" in combined or "firerate" in combined:
        return "combat", "fire_rate"
    
    # Priority 2: High-specificity single tokens
    if "duration" in combined or "time" in combined and not ("reload" in combined or "cast" in combined or "charge" in combined):
        return "ability", "duration"
    if "cooldown" in combined or "cd" in combined:
        return "ability", "cooldown"
    if "charge" in combined:
        return "ability", "charges"
    if "width" in combined or "size" in combined or "radius" in combined:
        return "ability", "size"
    if "health" in combined or "hp" in combined:
        return "ability", "health"
    if "slide" in combined:
        return "movement", "slide_count"
    
    # Priority 3: Broad tokens
    if "damage" in combined:
        return "combat", "damage"
    if "magazine" in combined or "ammo" in combined:
        return "combat", "ammo"
    if "cost" in combined or "credits" in combined:
        return "economy", "cost"
    if "movement" in combined or "speed" in combined or "velocity" in combined:
        return "movement", "movement_speed"
        
    # Priority 4: Qualitative changes
    if "rework" in combined or "remove" in combined or "mechanic" in combined or "logic" in combined:
        clean_name = extracted_name.strip() if extracted_name else "mechanic_change"
        return "general", clean_name
        
    # Fallback: Unknown feature
    clean_name = extracted_name.strip() if extracted_name else "raw_text"
    logger.warning(f"Unknown feature extracted: '{clean_name}'. Falling back to general category.")
    return "general", clean_name

def extract_feature_name(description, ability=None):
    """Extracts a candidate feature name from description string."""
    if ":" in description:
        prefix = description.split(":")[0].strip()
        clean_prefix = re.sub(r'[^a-zA-Z0-9_\s-]', '', prefix).strip()
        if clean_prefix and len(clean_prefix.split()) <= 2:
            return clean_prefix.lower().replace(" ", "_")
            
    m1 = transition_pattern1.search(description)
    m2 = transition_pattern2.search(description)
    start_idx = None
    if m1:
        start_idx = m1.start()
    elif m2:
        start_idx = m2.start()
        
    if start_idx is not None and start_idx > 0:
        preceding_text = description[:start_idx].strip()
        preceding_text = re.sub(r'\b(increased|decreased|from|to|reduced|changed|adjusted|by|of)\b.*$', '', preceding_text, flags=re.IGNORECASE).strip()
        words = re.findall(r'[a-zA-Z0-9_-]+', preceding_text)
        if words:
            return words[-1].lower()
            
    return "general" if not ability else ability.lower().replace(" ", "_")

def build_features(version):
    """Translates a processed JSON patch into the structured feature schema."""
    patch_path = os.path.join(PATCHES_PROCESSED_DIR, f"{version}.json")
    if not os.path.exists(patch_path):
        logger.warning(f"Flat patch JSON not found: {patch_path}")
        return None
        
    with open(patch_path, "r", encoding="utf-8") as f:
        patch_data = json.load(f)
        
    feature_data = {
        "Agent Updates": {},
        "Weapon Updates": {}
    }
    
    stats = {
        "total_extracted": 0,
        "numeric_features": 0,
        "text_only_features": 0,
        "ignored_changes": 0,
        "failures": 0
    }
    
    # 1. Process agent changes
    for change in patch_data.get("agent_changes", []):
        agent = change.get("agent", "General")
        ability = change.get("ability", "General")
        desc = change.get("description", "")
        change_type_init = change.get("change_type", "Adjustment")
        
        # Avoid processing general metadata or platform notes as updates
        if agent.lower() in ["all platforms", "general updates"]:
            continue
            
        trans = parse_numeric_transition(desc)
        if trans:
            old_val, new_val = trans
            feat_raw = extract_feature_name(desc, ability)
            category, feat_name = map_semantic_feature(feat_raw, desc)
            change_type = infer_change_type(feat_name, old_val, new_val, change_type_init)
            stats["numeric_features"] += 1
            stats["total_extracted"] += 1
            
            payload = {
                "agent": agent,
                "ability": ability,
                "category": category,
                "feature_name": feat_name,
                "type": change_type,
                "weight": get_weight_for_type(change_type),
                "values": {"old": old_val, "new": new_val}
            }
        else:
            # Non-numeric change
            feat_raw = extract_feature_name(desc, ability)
            category, feat_name = map_semantic_feature(feat_raw, desc)
            change_type = infer_change_type(feat_name, None, None, change_type_init)
            stats["text_only_features"] += 1
            stats["total_extracted"] += 1
            
            payload = {
                "agent": agent,
                "ability": ability,
                "category": category,
                "feature_name": feat_name,
                "type": change_type,
                "weight": get_weight_for_type(change_type),
                "values": None
            }
            
        if agent not in feature_data["Agent Updates"]:
            feature_data["Agent Updates"][agent] = []
        feature_data["Agent Updates"][agent].append(payload)
        
    # 2. Process weapon changes
    for change in patch_data.get("weapon_changes", []):
        weapon = change.get("weapon", "General")
        stat = change.get("stat", "General")
        desc = change.get("description", "")
        old_val = change.get("old_value")
        new_val = change.get("new_value")
        change_type_init = change.get("change_type", "Adjustment")
        
        feat_raw = extract_feature_name(desc, stat)
        category, feat_name = map_semantic_feature(feat_raw, desc)
        
        if old_val is not None and new_val is not None:
            change_type = infer_change_type(feat_name, float(old_val), float(new_val), change_type_init)
            stats["numeric_features"] += 1
            stats["total_extracted"] += 1
            
            payload = {
                "weapon": weapon,
                "category": category,
                "feature_name": feat_name,
                "type": change_type,
                "weight": get_weight_for_type(change_type),
                "values": {"old": float(old_val), "new": float(new_val)}
            }
        else:
            # Attempt to parse from description if old/new is missing
            trans = parse_numeric_transition(desc)
            if trans:
                old_v, new_v = trans
                change_type = infer_change_type(feat_name, old_v, new_v, change_type_init)
                stats["numeric_features"] += 1
                stats["total_extracted"] += 1
                payload = {
                    "weapon": weapon,
                    "category": category,
                    "feature_name": feat_name,
                    "type": change_type,
                    "weight": get_weight_for_type(change_type),
                    "values": {"old": old_v, "new": new_v}
                }
            else:
                change_type = infer_change_type(feat_name, None, None, change_type_init)
                stats["text_only_features"] += 1
                stats["total_extracted"] += 1
                payload = {
                    "weapon": weapon,
                    "category": category,
                    "feature_name": feat_name,
                    "type": change_type,
                    "weight": get_weight_for_type(change_type),
                    "values": None
                }
                
        if weapon not in feature_data["Weapon Updates"]:
            feature_data["Weapon Updates"][weapon] = []
        feature_data["Weapon Updates"][weapon].append(payload)
        
    # 3. Compile ignored lists counts
    for category in ["competitive_changes", "performance_changes", "bug_fixes", "player_behavior_changes"]:
        stats["ignored_changes"] += len(patch_data.get(category, []))
        
    # Ensure directory exists and write output
    os.makedirs(FEATURES_DIR, exist_ok=True)
    features_path = os.path.join(FEATURES_DIR, f"{version}.json")
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feature_data, f, indent=4)
        
    logger.info(f"Built and saved features for version {version} to {features_path}")
    return stats

def run_validation():
    validation_versions = ["9.0", "9.01", "9.02", "9.03", "9.04", "10.04", "12.09"]
    logger.info(f"Running validation bridge across: {validation_versions}")
    
    report = {
        "summary": {
            "total_extracted_features": 0,
            "numeric_features": 0,
            "text_only_features": 0,
            "ignored_changes": 0,
            "extraction_failures": 0
        },
        "versions": {}
    }
    
    for version in validation_versions:
        try:
            stats = build_features(version)
            if stats:
                report["versions"][version] = stats
                report["summary"]["total_extracted_features"] += stats["total_extracted"]
                report["summary"]["numeric_features"] += stats["numeric_features"]
                report["summary"]["text_only_features"] += stats["text_only_features"]
                report["summary"]["ignored_changes"] += stats["ignored_changes"]
                report["summary"]["extraction_failures"] += stats["failures"]
            else:
                report["versions"][version] = {"error": "Missing patch JSON"}
                report["summary"]["extraction_failures"] += 1
        except Exception as e:
            logger.error(f"Error bridging version {version}: {e}")
            report["versions"][version] = {"error": str(e)}
            report["summary"]["extraction_failures"] += 1
            
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "feature_builder_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    logger.info(f"Saved feature builder report to {report_path}")
    
    print("\n" + "="*80)
    print("FEATURE BUILDER REPORT")
    print(f"Total Extracted Features: {report['summary']['total_extracted_features']}")
    print(f"Numeric Features: {report['summary']['numeric_features']}")
    print(f"Text-Only Features: {report['summary']['text_only_features']}")
    print(f"Ignored Non-Balance Changes: {report['summary']['ignored_changes']}")
    print(f"Extraction Failures: {report['summary']['extraction_failures']}")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_validation()
