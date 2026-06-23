import os
import re
import json
from datetime import datetime

class PatchParser:
    def __init__(self):
        # Mappings of template patterns
        self.ai_pattern = re.compile(r'\{\{ai\|([^}]+)\}\}')
        self.wi_pattern = re.compile(r'\{\{wi\|([^}]+)\}\}')
        self.ui_pattern = re.compile(r'\{\{ui\|([^}]+)\}\}')
        self.abi_pattern = re.compile(r'\{\{abi text\|([^}]+)\}\}')
        self.link_pattern = re.compile(r'\[\[(?:[^|\]]+\|)?([^\]]+)\]\]')
        self.general_template_pattern = re.compile(r'\{\{[^|}]+\|([^}]+)\}\}')
        
        # Numeric transitions patterns
        # Matches: X >>> Y, X -> Y, X >>> Y%, X% -> Y% (supports leading dot decimals like .075)
        self.transition_pattern1 = re.compile(
            r'((?:\d+(?:\.\d+)?|\.\d+))\s*%?\s*(?:>>>|->)\s*((?:\d+(?:\.\d+)?|\.\d+))\s*%?'
        )
        # Matches: increased/decreased from X to Y
        self.transition_pattern2 = re.compile(
            r'(increased|decreased)\s+from\s+((?:\d+(?:\.\d+)?|\.\d+))\s*%?\s+(?:to|>>>|->)\s*((?:\d+(?:\.\d+)?|\.\d+))\s*%?',
            re.IGNORECASE
        )

    def normalize_text(self, text: str) -> str:
        """Removes MediaWiki templates and link syntax to return clean text."""
        # 1. Normalize specific templates
        text = self.ai_pattern.sub(r'\1', text)
        text = self.wi_pattern.sub(r'\1', text)
        text = self.ui_pattern.sub(r'\1', text)
        text = self.abi_pattern.sub(r'\1', text)
        text = self.link_pattern.sub(r'\1', text)
        text = self.general_template_pattern.sub(r'\1', text)
        # 2. Strip bold/italic quotes
        text = text.replace("'''", "").replace("''", "")
        return text.strip()

    def normalize_date(self, date_str: str) -> str:
        """Converts human-readable date formats (e.g. 'May 12th, 2026') to YYYY-MM-DD."""
        date_str = date_str.strip()
        # Remove ordinals (1st, 2nd, 3rd, 4th...)
        date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
        # Replace special spaces
        date_str = date_str.replace('\xa0', ' ')
        for fmt in ("%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return date_str

    def parse_infobox_date(self, raw_text: str) -> str:
        """Attempts to extract the patch date from the MediaWiki Infobox template."""
        # Look for |date = ... or | date = ... inside Infobox
        date_match = re.search(r'\|\s*date\s*=\s*([^|\n}]+)', raw_text)
        if date_match:
            return self.normalize_date(date_match.group(1))
        return ""

    def parse_patch(self, version: str, date_str: str, raw_text: str) -> dict:
        """
        Parses raw patch wikitext into a structured patch notes dictionary.
        """
        date = date_str
        if not date:
            date = self.parse_infobox_date(raw_text)
            
        patch_json = {
            "version": version,
            "date": date,
            "agent_changes": [],
            "weapon_changes": [],
            "competitive_changes": [],
            "performance_changes": [],
            "bug_fixes": [],
            "player_behavior_changes": []
        }
        
        current_category = None  # e.g. 'Agent Updates', 'Weapon Updates', 'Bug Fixes'
        current_subject = None   # e.g. 'Neon', 'Bucky'
        current_ability = "General"
        
        lines = raw_text.split('\n')
        
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
                
            # --- 1. Detect Heading Categories ---
            heading_match = re.match(r'^(==+)(.+?)(==+)$', line_str)
            if heading_match:
                title = heading_match.group(2).strip()
                title_clean = self.normalize_text(title)
                # Deduplicate adjacent identical words (e.g. "Bucky Bucky" -> "Bucky")
                words = title_clean.split()
                dedup_words = []
                for w in words:
                    if not dedup_words or w.lower() != dedup_words[-1].lower():
                        dedup_words.append(w)
                title_clean = " ".join(dedup_words)
                level = len(heading_match.group(1))
                
                # Check category shifts
                if "agent update" in title_clean.lower():
                    current_category = "agent_changes"
                    current_subject = None
                    current_ability = "General"
                    continue
                elif "weapon update" in title_clean.lower():
                    current_category = "weapon_changes"
                    current_subject = None
                    continue
                elif "competitive update" in title_clean.lower():
                    current_category = "competitive_changes"
                    current_subject = None
                    continue
                elif "performance update" in title_clean.lower():
                    current_category = "performance_changes"
                    current_subject = None
                    continue
                elif "bug fix" in title_clean.lower():
                    current_category = "bug_fixes"
                    current_subject = None
                    continue
                elif "player behavior" in title_clean.lower():
                    current_category = "player_behavior_changes"
                    current_subject = None
                    continue
                
                # Check subjects under categories
                if current_category == "agent_changes" and level >= 3:
                    current_subject = title_clean
                    current_ability = "General"
                    continue
                elif current_category == "weapon_changes" and level >= 3:
                    current_subject = title_clean
                    continue
                    
                continue
                
            # --- 2. Parse Bullets ---
            if line_str.startswith("*"):
                # Normalize line content
                clean_line = self.normalize_text(line_str)
                bullet_level = len(re.match(r'^(\*+)', line_str).group(1))
                
                # A. Handle Agent Changes
                if current_category == "agent_changes":
                    subject = current_subject or "General"
                    # Clean double/triple stars
                    clean_content = re.sub(r'^\*+\s*', '', clean_line)
                    
                    # Detect if bullet is declaring an ability (starts with {{abi text|...}} or fits ability style)
                    # Wikitext checks:
                    is_abi_declaration = False
                    abi_match = self.abi_pattern.search(line_str)
                    if abi_match and bullet_level == 1:
                        # Check if it has actual change details on the same line
                        line_stripped = self.abi_pattern.sub('', line_str).strip('* ').strip()
                        if len(line_stripped) > 3 or any(kw in line_stripped.lower() for kw in [" >>> ", " -> ", "increase", "decrease", "reduce", "buff", "nerf", "cost"]):
                            current_ability = abi_match.group(1).strip()
                            is_abi_declaration = False
                        else:
                            is_abi_declaration = True
                            current_ability = abi_match.group(1).strip()
                    elif bullet_level == 1 and not any(x in clean_content for x in ["Nerf", "Buff", "Adjustment", "Bugfix"]):
                        # Fallback: if it's a short first-level bullet and has no nerf/buff tags, treat it as ability context
                        if len(clean_content) < 30:
                            is_abi_declaration = True
                            current_ability = clean_content
                        else:
                            is_abi_declaration = False
                            
                    if is_abi_declaration:
                        continue
                        
                    # Extract change details from sub-bullet or direct bullet
                    change_type = "Adjustment"
                    for t in ["Nerf", "Buff", "Adjustment", "Bugfix"]:
                        if clean_content.startswith(t):
                            change_type = t
                            clean_content = clean_content[len(t):].strip()
                            break
                            
                    # Clean up descriptions
                    clean_content = re.sub(r'^[:\-\s\>]+', '', clean_content).strip()
                    
                    if clean_content:
                        # Append to existing changes
                        patch_json["agent_changes"].append({
                            "agent": subject,
                            "ability": current_ability,
                            "change_type": change_type,
                            "description": clean_content
                        })
                        
                # B. Handle Weapon Changes
                elif current_category == "weapon_changes":
                    subject = current_subject or "General"
                    clean_content = re.sub(r'^\*+\s*', '', clean_line)
                    
                    # Check for numeric transitions
                    transition = self.transition_pattern1.search(clean_content)
                    transition2 = self.transition_pattern2.search(clean_content)
                    
                    if transition or transition2:
                        old_val = None
                        new_val = None
                        change_type = "Adjustment"
                        
                        if transition:
                            old_val = float(transition.group(1))
                            new_val = float(transition.group(2))
                            # Determine start index of transition in string to isolate stat
                            stat_text_raw = clean_content[:transition.start()].strip()
                        else:
                            direction = transition2.group(1).lower()
                            old_val = float(transition2.group(2))
                            new_val = float(transition2.group(3))
                            change_type = "Nerf" if direction == "decreased" else "Buff"
                            stat_text_raw = clean_content[:transition2.start()].strip()
                            
                        # Extract explicit nerf/buff indicators
                        for t in ["Nerf", "Buff", "Adjustment"]:
                            if stat_text_raw.startswith(t):
                                change_type = t
                                stat_text_raw = stat_text_raw[len(t):].strip()
                                break
                                
                        # Guess change type if not explicitly set
                        if change_type == "Adjustment":
                            is_decrease = new_val < old_val
                            # For costs/spread/cooldowns, decrease is a Buff
                            is_cost_or_spread = any(x in stat_text_raw.lower() for x in ["cost", "spread", "reload", "multiplier", "decrease"])
                            if is_cost_or_spread:
                                change_type = "Buff" if is_decrease else "Nerf"
                            else:
                                change_type = "Nerf" if is_decrease else "Buff"
                                
                        # Clean stat text
                        stat_text = re.sub(r'^[:\-\s\>]+', '', stat_text_raw).strip()
                        stat_text = re.sub(r'\b(increased|decreased|from|to)\b', '', stat_text, flags=re.IGNORECASE).strip()
                        stat_text = stat_text.replace(":", "").strip()
                        if not stat_text:
                            stat_text = "General"
                            
                        # Strip current subject name from stat if present (e.g. Bucky Head -> Head)
                        if stat_text.startswith(subject):
                            stat_text = stat_text[len(subject):].strip()
                            
                        patch_json["weapon_changes"].append({
                            "weapon": subject,
                            "stat": stat_text,
                            "old_value": old_val,
                            "new_value": new_val,
                            "change_type": change_type
                        })
                    else:
                        # Non-numeric change description
                        change_type = "Adjustment"
                        for t in ["Nerf", "Buff", "Adjustment"]:
                            if clean_content.startswith(t):
                                change_type = t
                                clean_content = clean_content[len(t):].strip()
                                break
                        clean_content = re.sub(r'^[:\-\s\>]+', '', clean_content).strip()
                        
                        patch_json["weapon_changes"].append({
                            "weapon": subject,
                            "stat": "General",
                            "old_value": None,
                            "new_value": None,
                            "change_type": change_type,
                            "description": clean_content
                        })
                        
                # C. Handle Other Categories
                elif current_category in patch_json and current_category not in ["agent_changes", "weapon_changes"]:
                    clean_content = re.sub(r'^\*+\s*', '', clean_line)
                    # Clean leading symbols
                    clean_content = re.sub(r'^[:\-\s\>]+', '', clean_content).strip()
                    if clean_content:
                        # Append description or structured note
                        patch_json[current_category].append(clean_content)
                        
        return patch_json
