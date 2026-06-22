import re

def parse_mediawiki_tree(raw_wiki_text):
    """
    State machine parser mapping MediaWiki headings to a structured dictionary tree.
    Extracts platforms, update categories, subjects, and semantic change metrics.
    """
    lines = raw_wiki_text.split('\n')
    patch_data = {"Agent Updates": {}, "Weapon Updates": {}}
    
    current_platform = None
    current_category = None
    current_subject = None
    
    # Regular expressions for token extraction
    ui_pattern = re.compile(r'\{\{ui\|(Nerf|Buff|Adjustment|Bugfix)\}\}')
    val_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*>>>\s*(\d+(?:\.\d+)?)')
    entity_pattern = re.compile(r'\{\{[aw]i\|([^}]+)\}\}')

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
            
        # 1. Platform Sections (Level 2)
        if line_str.startswith("==") and not line_str.startswith("==="):
            current_platform = line_str.replace("==", "").strip()
            continue
            
        # 2. Update Categories (Level 3)
        if line_str.startswith("===") and not line_str.startswith("===="):
            current_category = line_str.replace("===", "").strip()
            continue
            
        # 3. Subject Sections (Level 4)
        if line_str.startswith("===="):
            subject_raw = line_str.replace("====", "").strip()
            entity_match = entity_pattern.search(subject_raw)
            current_subject = entity_match.group(1) if entity_match else subject_raw
            continue
            
        # 4. Change Lists parsing (Bullet levels)
        if line_str.startswith("*") and current_category in ["Agent Updates", "Weapon Updates"] and current_subject:
            ui_match = ui_pattern.search(line_str)
            val_match = val_pattern.search(line_str)
            
            if ui_match:
                change_type = ui_match.group(1).lower()
                if current_subject not in patch_data[current_category]:
                    patch_data[current_category][current_subject] = []
                    
                penalty_weight = 0.8 if change_type == "nerf" else (0.2 if change_type == "adjustment" else 0.0)
                
                old_v, new_v = 0.0, 0.0
                if val_match:
                    old_v, new_v = float(val_match.group(1)), float(val_match.group(2))
                
                patch_data[current_category][current_subject].append({
                    "type": change_type,
                    "weight": penalty_weight,
                    "values": {"old": old_v, "new": new_v}
                })
                
    return patch_data