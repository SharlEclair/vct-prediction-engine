import os, json, re
from patch_parser import PatchParser
from feature_builder import build_features
from patch_ingestor import get_patch_versions, load_version_dates
from patch_analyzer import generate_patch_distances

def load_registry():
    path = os.path.join('data', 'processed', 'automated_patch_nerf_registry.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def count_raw_balance_changes(wiki_path):
    count = 0
    current_category = None
    with open(wiki_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            # detect headings
            heading_match = re.match(r'^(==+)(.+?)(==+)$', stripped)
            if heading_match:
                title = heading_match.group(2).strip().lower()
                if 'agent' in title and 'update' in title:
                    current_category = 'agent'
                elif 'weapon' in title and 'update' in title:
                    current_category = 'weapon'
                else:
                    current_category = None
                continue
            if stripped.startswith('*') and current_category:
                count += 1
    return count

def audit():
    # Get versions 9.0 to 12.09 inclusive
    versions = []
    csv_path = os.path.join('data','raw','patch_notes.csv')
    # use get_patch_versions without limit to get all
    from patch_ingestor import get_patch_versions
    all_versions = get_patch_versions(csv_path=csv_path, limit=None)
    def vtuple(v):
        parts = v.split('.')
        cleaned_parts = []
        for p in parts:
            # Use regex to find the first continuous digits in the string segment
            match = re.match(r'\d+', p)
            if match:
                cleaned_parts.append(int(match.group()))
        else:
            cleaned_parts.append(0) # Fallback if a segment has no digits
        return tuple(cleaned_parts)

    start, end = vtuple('9.0'), vtuple('12.09')
    for v in all_versions:
        vt = vtuple(v)
        if start <= vt <= end:
            versions.append(v)
    versions = sorted(versions, key=vtuple)
    parser = PatchParser()
    registry = load_registry()
    # Ensure registry exists by running analyzer if missing
    if not registry:
        generate_patch_distances()
        registry = load_registry()
    rows = []
    for v in versions:
        wiki_path = os.path.join('data','patches', f'{v}.wiki')
        
        # Skip this version completely if the wiki file doesn't exist
        if not os.path.exists(wiki_path):
            continue
            
        raw_changes = count_raw_balance_changes(wiki_path)
        with open(wiki_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            
        parsed = parser.parse_patch(v, '', raw_text)
        parsed_changes = len(parsed.get('agent_changes', [])) + len(parsed.get('weapon_changes', []))
        
        # build features (gets stats)
        stats = build_features(v)
        features_generated = stats.get('total_extracted', 0)
        numeric_features = stats.get('numeric_features', 0)
        features_consumed = features_generated  # analyzer consumes all
        
        registry_entries = len(registry.get(v, {})) if isinstance(registry, dict) else 0
        rows.append([v, raw_changes, parsed_changes, features_generated, numeric_features, features_consumed, registry_entries])    # Print markdown table
    print('# Survival Analysis Audit Table')
    print('| Version | Raw Balance Changes | Parsed Changes | Features Generated | Numeric Features | Features Consumed | Registry Entries |')
    print('|---|---|---|---|---|---|---|')
    for row in rows:
        print('|' + '|'.join(str(r) for r in row) + '|')
    # Detailed trace for specific cases
    trace_cases = [
        ('9.11', 'Neon'),
        ('10.04', 'Clove'),
        ('12.00', 'Breach')
    ]
    print('\n## Detailed Traces')
    for version, name in trace_cases:
        print(f'### {name} in {version}')
        # raw line
        wiki_path = os.path.join('data','patches', f'{version}.wiki')
        raw_line = ''
        with open(wiki_path, 'r', encoding='utf-8') as f:
            for line in f:
                if name.lower() in line.lower() and line.strip().startswith('*'):
                    raw_line = line.strip()
                    break
        print('**Raw wiki line:**')
        print(f'`{raw_line}`')
        # parsed JSON object
        with open(wiki_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        parsed = parser.parse_patch(version, '', raw_text)
        parsed_obj = None
        for ch in parsed.get('agent_changes', []):
            if ch.get('agent','').lower() == name.lower():
                parsed_obj = ch
                break
        if not parsed_obj:
            for ch in parsed.get('weapon_changes', []):
                if ch.get('weapon','').lower() == name.lower():
                    parsed_obj = ch
                    break
        print('**Parsed JSON object:**')
        print('```json')
        print(json.dumps(parsed_obj, indent=4))
        print('```')
        # feature object
        feat_path = os.path.join('data','processed','features', f'{version}.json')
        with open(feat_path, 'r', encoding='utf-8') as f:
            feats = json.load(f)
        feat_obj = None
        if name in feats.get('Agent Updates', {}):
            feat_obj = feats['Agent Updates'][name][0]  # first payload
        elif name in feats.get('Weapon Updates', {}):
            feat_obj = feats['Weapon Updates'][name][0]
        print('**Feature object:**')
        print('```json')
        print(json.dumps(feat_obj, indent=4))
        print('```')
        # registry value
        reg_val = registry.get(version, {}).get(name)
        print('**Final registry value:**')
        print(f'`{reg_val}`')
        print('\n')

if __name__ == '__main__':
    audit()
