import os
import re
import json
import logging
from patch_parser import PatchParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("quality_reporter")

PATCHES_DIR = "./data/patches"
PROCESSED_DIR = "./data/processed/patches"
REPORTS_DIR = "./data/reports"

# Templates currently supported by PatchParser
SUPPORTED_TEMPLATES = {"ai", "wi", "ui", "abi text"}

def extract_templates(wikitext: str) -> list[str]:
    """Finds all MediaWiki templates like {{ai|Neon}} or {{Infobox patch}} in the wikitext."""
    # Matches {{template_name|...}} or {{template_name}}
    return re.findall(r'\{\{([^|}\n]+)(?:\||})', wikitext)

def is_meaningful_bullet(line_str: str) -> bool:
    """Checks if a bullet line contains actual content changes rather than structure or nav."""
    if not line_str.startswith("*"):
        return False
    # Clean it up
    clean = line_str.replace("*", "").strip()
    if not clean:
        return False
    # Ignore pure headers/nav/toc templates
    if re.match(r'^\{\{PatchNav', clean, re.IGNORECASE) or re.match(r'^\{\{TOC', clean, re.IGNORECASE):
        return False
    # Ignore standalone agent/weapon/ability structure bullets (e.g. *{{ai|Neon}}, *{{abi text|High Gear}} with no other text)
    # If a line is just the template and has no other words, it's a structure bullet
    clean_stripped = re.sub(r'\{\{[^}]+\}\}', '', clean).strip()
    # Strip links [[...]]
    clean_stripped = re.sub(r'\[\[[^\]]+\]\]', '', clean_stripped).strip()
    if not clean_stripped or len(clean_stripped) <= 2:
        return False
        
    # Ignore cosmetic/store bundles and capsules
    if re.search(r'\{\{(collection|capsule|bundle)', clean, re.IGNORECASE):
        return False
    if "minor patch" in clean.lower():
        return False
        
    # Ignore purely structural layout headings in bullets
    clean_lower = clean_stripped.lower()
    structural_headings = {
        "console changes:", "pc changes:", "maps", "bug fixes", "pings", 
        "servers", "agents", "gameplay systems", "engine update", "competitive"
    }
    if clean_lower in structural_headings or clean_lower.rstrip(":") in structural_headings:
        return False
        
    return True

def run_quality_check():
    logger.info("Initializing parser quality analysis...")
    
    if not os.path.exists(PATCHES_DIR) or not os.listdir(PATCHES_DIR):
        logger.error(f"No patches cache found at {PATCHES_DIR}. Please run backfill first.")
        return
        
    version_stats = {}
    global_template_counts = {}
    unrecognized_template_counts = {}
    
    total_bullet_lines = 0
    total_uncategorized_lines = 0
    all_uncategorized_lines_details = []
    
    parser = PatchParser()
    
    # Sort files
    files = sorted(os.listdir(PATCHES_DIR))
    
    for filename in files:
        if not filename.endswith(".wiki"):
            continue
            
        version = filename.replace(".wiki", "")
        wiki_path = os.path.join(PATCHES_DIR, filename)
        json_path = os.path.join(PROCESSED_DIR, f"{version}.json")
        
        # Read raw wikitext
        with open(wiki_path, "r", encoding="utf-8") as f:
            wikitext = f.read()
            
        # 1. Scan templates
        templates = extract_templates(wikitext)
        ver_unrecognized_count = 0
        
        for t in templates:
            t_clean = t.strip().lower()
            global_template_counts[t_clean] = global_template_counts.get(t_clean, 0) + 1
            
            if t_clean not in SUPPORTED_TEMPLATES:
                unrecognized_template_counts[t_clean] = unrecognized_template_counts.get(t_clean, 0) + 1
                ver_unrecognized_count += 1
                
        # 2. Match parsed changes
        if not os.path.exists(json_path):
            logger.warning(f"Structured JSON missing for version: {version}")
            continue
            
        with open(json_path, "r", encoding="utf-8") as f:
            parsed_data = json.load(f)
            
        # Read lines to evaluate categorisation
        lines = wikitext.split("\n")
        ver_bullet_lines = 0
        ver_uncategorized = 0
        ver_uncategorized_lines_list = []
        
        # Compile lists of descriptions and titles in JSON for quick lookup
        parsed_descriptions = set()
        
        # Add agent changes descriptions
        for ac in parsed_data.get("agent_changes", []):
            desc = parser.normalize_text(ac.get("description", ""))
            if desc:
                parsed_descriptions.add(desc.lower())
                
        # Add weapon changes descriptions and stats
        for wc in parsed_data.get("weapon_changes", []):
            desc = wc.get("description", "")
            if desc:
                parsed_descriptions.add(parser.normalize_text(desc).lower())
            stat = wc.get("stat", "")
            if stat:
                parsed_descriptions.add(parser.normalize_text(stat).lower())
                
        # Add competitive, performance, bug fixes, player behavior lists
        for key in ["competitive_changes", "performance_changes", "bug_fixes", "player_behavior_changes"]:
            for item in parsed_data.get(key, []):
                item_cleaned = parser.normalize_text(item)
                if item_cleaned:
                    parsed_descriptions.add(item_cleaned.lower())
                    
        has_headings_started = False
        current_category = None
        for line in lines:
            line_str = line.strip()
            if line_str.startswith("=="):
                has_headings_started = True
                m = re.match(r'^(==+)(.+?)(==+)$', line_str)
                if m:
                    title_clean = parser.normalize_text(m.group(2)).strip().lower()
                    # Deduplicate adjacent identical words
                    words = title_clean.split()
                    dedup_words = []
                    for w in words:
                        if not dedup_words or w != dedup_words[-1]:
                            dedup_words.append(w)
                    title_clean = " ".join(dedup_words)
                    
                    if "agent update" in title_clean:
                        current_category = "agent_changes"
                    elif "weapon update" in title_clean:
                        current_category = "weapon_changes"
                    elif "competitive update" in title_clean:
                        current_category = "competitive_changes"
                    elif "performance update" in title_clean:
                        current_category = "performance_changes"
                    elif "bug fix" in title_clean:
                        current_category = "bug_fixes"
                    elif "player behavior" in title_clean:
                        current_category = "player_behavior_changes"
                    else:
                        level = len(m.group(1))
                        if level == 2:
                            current_category = None
            if not has_headings_started or not current_category:
                continue
                
            if is_meaningful_bullet(line_str):
                ver_bullet_lines += 1
                total_bullet_lines += 1
                
                # Check if this bullet is represented in parsed JSON
                normalized_bullet = parser.normalize_text(line_str).lower()
                # Strip leading bullet and tag prefix (e.g. "nerf head 40 >>> 34" -> "head 40 >>> 34")
                normalized_bullet_stripped = re.sub(r'^\*+\s*', '', normalized_bullet).strip()
                for prefix in ["nerf", "buff", "adjustment", "bugfix"]:
                    if normalized_bullet_stripped.startswith(prefix):
                        normalized_bullet_stripped = normalized_bullet_stripped[len(prefix):].strip()
                
                # Check if normalized bullet is a substring or close match to any parsed item
                matched = False
                for parsed_desc in parsed_descriptions:
                    if (normalized_bullet_stripped in parsed_desc) or (parsed_desc in normalized_bullet_stripped):
                        matched = True
                        break
                        
                if not matched:
                    ver_uncategorized += 1
                    total_uncategorized_lines += 1
                    ver_uncategorized_lines_list.append(line_str)
                    all_uncategorized_lines_details.append({
                        "version": version,
                        "line": line_str
                    })
                    
        version_stats[version] = {
            "version": version,
            "agent_changes": len(parsed_data.get("agent_changes", [])),
            "weapon_changes": len(parsed_data.get("weapon_changes", [])),
            "competitive_changes": len(parsed_data.get("competitive_changes", [])),
            "bug_fixes": len(parsed_data.get("bug_fixes", [])),
            "unparsed_lines": ver_uncategorized,
            "unrecognized_templates": ver_unrecognized_count,
            "uncategorized_examples": ver_uncategorized_lines_list[:5]
        }
        
    # Calculate uncategorized rate
    uncategorized_rate = (total_uncategorized_lines / total_bullet_lines) if total_bullet_lines > 0 else 0.0
    passed_success_criterion = uncategorized_rate < 0.05
    
    # Sort templates frequency table
    sorted_templates = {f"{{{{{k}}}}}" if not k.startswith("{{") else k: v for k, v in sorted(global_template_counts.items(), key=lambda x: x[1], reverse=True)}
    unhandled_templates = [f"{{{{{k}}}}}" for k in unrecognized_template_counts.keys()]
    
    quality_report = {
        "uncategorized_rate": round(uncategorized_rate * 100, 2),
        "total_meaningful_bullet_lines": total_bullet_lines,
        "total_uncategorized_lines": total_uncategorized_lines,
        "passed_success_criterion": passed_success_criterion,
        "template_frequency": sorted_templates,
        "unhandled_templates": unhandled_templates,
        "uncategorized_lines": all_uncategorized_lines_details,
        "version_statistics": version_stats
    }
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "parser_quality_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=4)
        
    logger.info(f"Saved parser quality report to: {report_path}")
    
    print("\n" + "="*80)
    print("PARSER QUALITY REPORT SUMMARY")
    print(f"Total Meaningful Bullet Lines: {total_bullet_lines}")
    print(f"Total Uncategorized Lines: {total_uncategorized_lines}")
    print(f"Uncategorized Rate: {uncategorized_rate:.2%}")
    print(f"Success Criterion Passed (<5%): {passed_success_criterion}")
    print(f"Unhandled Templates: {unhandled_templates[:10]}")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_quality_check()
