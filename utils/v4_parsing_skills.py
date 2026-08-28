import re

def parse_mediawiki_tree(raw_wiki_text):
    """
    Flexible state machine parser mapping MediaWiki headings to a structured dictionary tree.
    Extracts platforms, update categories, subjects, and semantic change metrics.
    Robust to heading levels (2, 3, or 4) and handles value deltas even if templates are missing.
    """
    lines = raw_wiki_text.split('\n')
    patch_data = {"Agent Updates": {}, "Weapon Updates": {}}
    
    current_category = None
    current_subject = None
    current_platform = None
    
    # Regular expressions for token extraction
    ui_pattern = re.compile(r'\{\{ui\|(Nerf|Buff|Adjustment|Bugfix)\}\}')
    val_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*[a-zA-Z%]*\s*>>>\s*(\d+(?:\.\d+)?)\s*[a-zA-Z%]*')
    entity_pattern = re.compile(r'\{\{[aw]i\|([^}]+)\}\}')

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
            
        # Detect headings (e.g. ==Heading==, ===Heading===, ====Heading====)
        heading_match = re.match(r'^(==+)(.+?)(==+)$', line_str)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            
            # Check if title is category
            if "agent update" in title.lower():
                current_category = "Agent Updates"
                current_subject = None
                continue
            elif "weapon update" in title.lower():
                current_category = "Weapon Updates"
                current_subject = None
                continue
                
            # Check if title contains entity (e.g. {{ai|Neon}} or {{wi|Operator}})
            entity_match = entity_pattern.search(title)
            if entity_match:
                current_subject = entity_match.group(1).strip()
                continue
                
            # If we are already under a category and see a heading at level 3 or 4:
            clean_title = re.sub(r'[^a-zA-Z0-9\s/]', '', title).strip()
            if current_category and level >= 3:
                # Filter out platform keywords
                if clean_title.upper() in ["PC", "CONSOLE", "ALL PLATFORMS", "GENERAL", "SYSTEMS", "BUG FIXES", "MAPS"]:
                    current_platform = clean_title
                else:
                    # Treat as subject (e.g. Neon, Iso, Operator)
                    current_subject = clean_title
            else:
                # Heading level 2 or other: check if it's a known agent or weapon if we can guess
                if clean_title.upper() in ["PC", "CONSOLE", "ALL PLATFORMS"]:
                    current_platform = clean_title
            continue
            
        # Detect inline subject declarations (e.g. *{{ai|Neon}} or '''{{ai|Iso}}''')
        if (line_str.startswith("*") or line_str.startswith("'")) and ("{{ai|" in line_str or "{{wi|" in line_str):
            entity_match = entity_pattern.search(line_str)
            if entity_match:
                current_subject = entity_match.group(1).strip()
                continue
            
        # Parse change lines starting with *
        if line_str.startswith("*") and current_category in ["Agent Updates", "Weapon Updates"] and current_subject:
            ui_match = ui_pattern.search(line_str)
            val_match = val_pattern.search(line_str)
            
            if val_match:
                old_v = float(val_match.group(1))
                new_v = float(val_match.group(2))
                
                # Determine feature name
                # Extract the word right before the value
                line_before_val = line_str[:val_match.start()].strip()
                parts = line_before_val.split()
                feature_name = "unknown"
                if parts:
                    feature_name = parts[-1].replace('}}', '').replace(':', '').replace('*', '').strip()
                
                # Clean up feature name
                feature_name = re.sub(r'[^a-zA-Z0-9_]', '', feature_name)
                if not feature_name:
                    feature_name = "unknown"
                
                # Determine change type
                if ui_match:
                    change_type = ui_match.group(1).lower()
                else:
                    # Guess type based on value delta and feature name
                    # Decreasing standard values is a nerf
                    is_nerf = new_v < old_v
                    # For cost or cooldown, increasing is a nerf
                    if any(x in feature_name.lower() for x in ["cost", "cooldown", "reload", "windup", "delay", "time"]):
                        is_nerf = new_v > old_v
                    change_type = "nerf" if is_nerf else "buff"
                    
                penalty_weight = 0.8 if change_type == "nerf" else (0.2 if change_type == "adjustment" else 0.0)
                
                if current_subject not in patch_data[current_category]:
                    patch_data[current_category][current_subject] = []
                    
                patch_data[current_category][current_subject].append({
                    "feature_name": feature_name,
                    "type": change_type,
                    "weight": penalty_weight,
                    "values": {"old": old_v, "new": new_v}
                })
                
    return patch_data