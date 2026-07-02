import os
import json
import logging
from patch_ingestor import ingest_latest_patches

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("verify_parser")

def run_verification():
    target_versions = ["9.0", "9.01", "9.02", "9.03", "9.04", "12.09"]
    logger.info(f"Running validation checks for patch versions: {target_versions}")
    
    try:
        parsed_data = ingest_latest_patches(version_list=target_versions)
    except Exception as e:
        logger.error(f"Ingestion failed during verification: {e}")
        return
        
    report = []
    report.append("# Valorant Patch Parser Validation Report\n")
    report.append("This report summarizes the parsing metrics and extracted data across the target patch notes.\n")
    report.append("| Patch Version | Release Date | Sections Populated | Agents Detected | Weapons Detected | Numeric Changes |")
    report.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    failures = 0
    total_numeric = 0
    
    for version in target_versions:
        data = parsed_data.get(version)
        if not data:
            report.append(f"| {version} | N/A | FAILED TO INGEST | - | - | - |")
            failures += 1
            continue
            
        # Analyze populated sections
        sections = []
        for key in ["agent_changes", "weapon_changes", "competitive_changes", "performance_changes", "bug_fixes", "player_behavior_changes"]:
            if data.get(key):
                sections.append(key)
                
        agents = sorted(list(set(change["agent"] for change in data.get("agent_changes", []))))
        weapons = sorted(list(set(change["weapon"] for change in data.get("weapon_changes", []))))
        
        numeric_count = 0
        for wc in data.get("weapon_changes", []):
            if wc.get("old_value") is not None and wc.get("new_value") is not None:
                numeric_count += 1
        total_numeric += numeric_count
        
        report.append(
            f"| {version} | {data.get('date')} | {', '.join(sections)} | {', '.join(agents) if agents else 'None'} | {', '.join(weapons) if weapons else 'None'} | {numeric_count} |"
        )

    report.append(f"\n**Total Numeric Transitions Extracted**: {total_numeric}")
    report.append(f"**Total Ingestion Failures**: {failures}\n")
    
    # 12.09 Specific Details
    data_12_09 = parsed_data.get("12.09", {})
    report.append("## Example Outputs from Patch 12.09\n")
    
    # 1. Neon Changes
    report.append("### ⚡ Neon (Agent Changes)")
    neon_changes = [c for c in data_12_09.get("agent_changes", []) if c["agent"].lower() == "neon"]
    if neon_changes:
        for c in neon_changes:
            report.append(f"- **Ability**: `{c['ability']}` ({c['change_type']})")
            report.append(f"  - *Description*: {c['description']}")
    else:
        report.append("No changes parsed for Neon.")
        
    # 2. Bucky Changes
    report.append("\n### 🔫 Bucky (Weapon Changes)")
    bucky_changes = [c for c in data_12_09.get("weapon_changes", []) if c["weapon"].lower() == "bucky"]
    if bucky_changes:
        for c in bucky_changes:
            if c.get("old_value") is not None:
                report.append(f"- **Stat**: `{c['stat']}` ({c['change_type']})")
                report.append(f"  - *Transition*: `{c['old_value']}` >>> `{c['new_value']}`")
            else:
                report.append(f"- **General**: ({c['change_type']}) - {c.get('description')}")
    else:
        report.append("No changes parsed for Bucky.")
        
    # 3. Judge Changes
    report.append("\n### 🔫 Judge (Weapon Changes)")
    judge_changes = [c for c in data_12_09.get("weapon_changes", []) if c["weapon"].lower() == "judge"]
    if judge_changes:
        for c in judge_changes:
            if c.get("old_value") is not None:
                report.append(f"- **Stat**: `{c['stat']}` ({c['change_type']})")
                report.append(f"  - *Transition*: `{c['old_value']}` >>> `{c['new_value']}`")
            else:
                report.append(f"- **General**: ({c['change_type']}) - {c.get('description')}")
    else:
        report.append("No changes parsed for Judge.")

    # 4. Shorty Changes
    report.append("\n### 🔫 Shorty (Weapon Changes)")
    shorty_changes = [c for c in data_12_09.get("weapon_changes", []) if c["weapon"].lower() == "shorty"]
    if shorty_changes:
        for c in shorty_changes:
            if c.get("old_value") is not None:
                report.append(f"- **Stat**: `{c['stat']}` ({c['change_type']})")
                report.append(f"  - *Transition*: `{c['old_value']}` >>> `{c['new_value']}`")
            else:
                report.append(f"- **General**: ({c['change_type']}) - {c.get('description')}")
    else:
        report.append("No changes parsed for Shorty.")

    # Write report files
    report_text = "\n".join(report)
    
    with open("validation_report.md", "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info("Saved validation_report.md to workspace root.")
    
    print("\n" + "="*80)
    print("VALIDATION REPORT OUTPUT")
    print("="*80)
    # Safe printing to avoid Windows cp1252 codec errors
    print(report_text.encode('ascii', errors='replace').decode('ascii'))
    print("="*80 + "\n")

if __name__ == "__main__":
    run_verification()
