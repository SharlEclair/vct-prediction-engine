"""
v8_patch_parser.py
------------------
Schema-Driven LLM Wikitext Patch Parsing Module (v8 Architecture).

Replaces legacy regex-based parsing with a robust, schema-driven LLM semantic
extraction pipeline that outputs strictly validated JSON matching Pydantic schemas.

Key Features:
1. Pydantic v2 JSON Schema Enforcement (PatchChangeItem & PatchExtractionPayload).
2. Bug Fix Paradigm Classifier (is_mechanical_removal = True for movement/physics/collision/animation fixes).
3. Lossless Evidence Tracking (raw_evidence stored per change).
4. Containerized Backend Compatibility with httpx REST integration and offline mock/test fallback.
"""

import os
import re
import json
import logging
import hashlib
from typing import Optional, Union, List, Dict, Any
import httpx
from pydantic import BaseModel, Field, ValidationError

# Configure Module Logger
logger = logging.getLogger("v8_patch_parser")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ============================================================================
# 1. PYDANTIC SCHEMAS (SCHEMA-DRIVEN NLP PARSING)
# ============================================================================

class PatchChangeItem(BaseModel):
    """
    Schema representing a single extracted patch change or bug fix.
    """
    agent: str = Field(
        ...,
        description="Target Valorant agent or subject (e.g., 'Neon', 'Jett', 'Cypher', 'Global', 'Vandal')."
    )
    ability: str = Field(
        ...,
        description="Target ability or weapon section (e.g., 'High Gear', 'Tailwind', 'General', 'Primary Fire')."
    )
    stat_modified: str = Field(
        ...,
        description="Attribute, mechanic, or behavior modified (e.g., 'Slide Speed', 'Equip Duration', 'Collision Exploit Removal')."
    )
    old_value: Optional[Union[float, int, str]] = Field(
        default=None,
        description="Previous numeric or textual value prior to the patch, if specified (e.g., 1.0, '0.8s', None)."
    )
    new_value: Optional[Union[float, int, str]] = Field(
        default=None,
        description="New numeric or textual value after the patch, if specified (e.g., 0.8, '1.2s', None)."
    )
    is_mechanical_removal: bool = Field(
        ...,
        description=(
            "CRITICAL: Must be True if the change/bug fix alters physics, collision, movement velocity, momentum, "
            "sliding mechanics, animation cancels, or ability trajectory geometry. "
            "Must be False for standard stat tweaks or cosmetic/UI/audio bug fixes."
        )
    )
    raw_evidence: Optional[str] = Field(
        default=None,
        description="Lossless raw wikitext snippet or textual bullet point serving as ground-truth evidence."
    )


class PatchExtractionPayload(BaseModel):
    """
    Schema representing the complete structured array of extracted patch data for a version.
    """
    version: str = Field(..., description="Patch version tag (e.g., '8.11', '9.01').")
    date: Optional[str] = Field(default=None, description="Patch release date (YYYY-MM-DD).")
    changes: List[PatchChangeItem] = Field(
        default_factory=list,
        description="Array of extracted structured patch changes."
    )
    raw_wikitext_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of the input raw wikitext for audit/lineage tracking."
    )


# ============================================================================
# 2. PROMPT TEMPLATES (FEW-SHOT BUG FIX PARADIGM & SYSTEM INSTRUCTIONS)
# ============================================================================

SYSTEM_PROMPT = """You are an expert Esports Data Scientist and Valorant Patch Analyst.
Your task is to convert raw MediaWiki patch notes wikitext into a structured JSON array following a strict JSON Schema.

### CATEGORIZATION RULES & BUG FIX PARADIGM:
1. **Agent & Ability Extraction**:
   - Extract the character/subject ('agent') and specific ability/system ('ability').
   - For general weapon or map changes, use the weapon/map name as 'agent' (e.g. 'Vandal', 'Ascent') and the sub-category as 'ability'.

2. **Numeric & Stat Extraction**:
   - Identify stat modified, old value, and new value when numerical transitions exist (e.g. '1.0 >>> 0.8', 'increased from 10 to 15').
   - When no explicit numerical transition is given, capture the described mechanics in 'stat_modified' and set old/new values to null or concise descriptions.

3. **CRITICAL: The Bug Fix Paradigm (`is_mechanical_removal`)**:
   - Professional esports rules ban bug exploits. Therefore, minor cosmetic bug fixes (e.g. UI alignment, HUD icons, audio loop glitches, text typos, spectator camera bugs) have NO impact on pro meta -> Set `is_mechanical_removal: false`.
   - HOWEVER, developers frequently classify the fix of advanced movement/physics exploits as "bug fixes" (e.g., animation cancels, momentum boosts, slide cancels, wall clipping, fake collision interactions, trajectory cancels). Removing these alters combat mobility geometry and represents a massive mechanical shock -> Set `is_mechanical_removal: true`.

### EXAMPLES (FEW-SHOT):

Example Input:
"=== Neon ===
* '''High Gear'''
** Slide speed decreased from 1.0 >>> 0.8.
** Fixed a bug where Neon could execute an unintended double slide boost when cancelling animation."

Example JSON Output:
```json
{
  "version": "8.11",
  "date": "2024-06-11",
  "changes": [
    {
      "agent": "Neon",
      "ability": "High Gear",
      "stat_modified": "Slide Speed",
      "old_value": 1.0,
      "new_value": 0.8,
      "is_mechanical_removal": false,
      "raw_evidence": "Slide speed decreased from 1.0 >>> 0.8."
    },
    {
      "agent": "Neon",
      "ability": "High Gear",
      "stat_modified": "Unintended Double Slide Boost Removal",
      "old_value": "Enabled",
      "new_value": "Disabled",
      "is_mechanical_removal": true,
      "raw_evidence": "Fixed a bug where Neon could execute an unintended double slide boost when cancelling animation."
    }
  ]
}
```

Example Input:
"=== Omen ===
* '''Dark Cover'''
** Fixed an issue where Dark Cover audio loop would play continuously after round ends."

Example JSON Output:
```json
{
  "version": "8.11",
  "date": "2024-06-11",
  "changes": [
    {
      "agent": "Omen",
      "ability": "Dark Cover",
      "stat_modified": "Round-End Audio Loop Bug",
      "old_value": null,
      "new_value": null,
      "is_mechanical_removal": false,
      "raw_evidence": "Fixed an issue where Dark Cover audio loop would play continuously after round ends."
    }
  ]
}
```

Return ONLY valid JSON matching the exact schema. Do not add markdown commentary outside the JSON block.
"""


# ============================================================================
# 3. V8 PATCH PARSER ENGINE
# ============================================================================

class V8PatchParser:
    """
    LLM-powered wikitext patch parser implementing the v8 schema-driven architecture.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 45.0,
        max_retries: int = 3,
        force_offline_mock: bool = False
    ):
        # Attempt loading .env if present
        env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_file):
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
            except Exception:
                pass

        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        # If Gemini key is used, default base URL to Google's OpenAI-compatible endpoint
        default_base = "https://generativelanguage.googleapis.com/v1beta/openai" if os.getenv("GEMINI_API_KEY") else "https://api.openai.com/v1"
        self.base_url = (base_url or os.getenv("LLM_API_BASE") or default_base).rstrip("/")
        
        default_model = "gemini-flash-latest" if os.getenv("GEMINI_API_KEY") else "gpt-4o-mini"
        self.model_name = model_name or os.getenv("LLM_MODEL") or default_model
        
        self.timeout = timeout
        self.max_retries = max_retries
        self.force_offline_mock = force_offline_mock or (not self.api_key)

        if self.force_offline_mock:
            logger.info("V8PatchParser initialized in OFFLINE / MOCK mode (No API Key provided or offline requested).")
        else:
            logger.info(f"V8PatchParser initialized with model '{self.model_name}' at base URL '{self.base_url}'.")

    @staticmethod
    def preprocess_wikitext(raw_wikitext: str) -> str:
        """
        Cleans noisy MediaWiki markup while preserving evidence text.
        """
        if not raw_wikitext:
            return ""
        
        text = raw_wikitext
        # Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        # Simplify common template tags like {{ai|Neon}} -> Neon, {{abi text|High Gear}} -> High Gear
        text = re.sub(r'\{\{(?:ai|wi|ui|abi text)\|([^}]+)\}\}', r'\1', text)
        text = re.sub(r'\[\[(?:[^|\]]+\|)?([^\]]+)\]\]', r'\1', text)
        # Normalize bold quotes '''
        text = text.replace("'''", "").replace("''", "")
        return text.strip()

    @staticmethod
    def extract_infobox_date(raw_wikitext: str) -> Optional[str]:
        """
        Extracts date string from MediaWiki Infobox template if present.
        """
        date_match = re.search(r'\|\s*date\s*=\s*([^|\n}]+)', raw_wikitext, re.IGNORECASE)
        if date_match:
            raw_date = date_match.group(1).strip()
            raw_date = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', raw_date)
            # Try parsing YYYY-MM-DD or Month DD, YYYY
            for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y"):
                try:
                    from datetime import datetime
                    return datetime.strptime(raw_date, fmt).strftime("%Y-%m-%d")
                except ValueError:
                    continue
            return raw_date
        return None

    def build_prompt_messages(self, clean_wikitext: str, patch_version: str = "", date_str: str = "") -> List[Dict[str, str]]:
        """
        Builds system and user message payload for the LLM endpoint.
        """
        user_prompt = f"Patch Version: {patch_version or 'Unknown'}\n"
        if date_str:
            user_prompt += f"Patch Date: {date_str}\n"
        user_prompt += f"\n--- RAW WIKITEXT START ---\n{clean_wikitext}\n--- RAW WIKITEXT END ---"

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

    def _call_llm_api(self, messages: List[Dict[str, str]]) -> str:
        """
        Sends synchronous HTTP POST request to OpenAI-compatible LLM endpoint using httpx.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }

        url = f"{self.base_url}/chat/completions"

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Sending LLM extraction request (Attempt {attempt}/{self.max_retries})...")
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    return content
            except Exception as e:
                logger.warning(f"LLM API call attempt {attempt} failed: {e}")
                if attempt == self.max_retries:
                    raise RuntimeError(f"Failed to extract patch data from LLM after {self.max_retries} attempts: {e}")

    def parse_wikitext(self, raw_wikitext: str, version: str = "", date_str: str = "") -> PatchExtractionPayload:
        """
        Main entry point for parsing raw wikitext into a validated PatchExtractionPayload.
        """
        wikitext_hash = hashlib.sha256(raw_wikitext.encode('utf-8')).hexdigest()
        clean_text = self.preprocess_wikitext(raw_wikitext)
        extracted_date = date_str or self.extract_infobox_date(raw_wikitext)

        # Fallback to offline mock if API key is not present or offline forced
        if self.force_offline_mock:
            return self.mock_parse_wikitext(raw_wikitext, version=version, date_str=extracted_date)

        messages = self.build_prompt_messages(clean_text, patch_version=version, date_str=extracted_date)
        
        try:
            raw_response = self._call_llm_api(messages)
            parsed_json = self._clean_and_parse_json(raw_response)
            
            # Ensure version and date are populated if missing in response
            if not parsed_json.get("version"):
                parsed_json["version"] = version or "Unknown"
            if not parsed_json.get("date"):
                parsed_json["date"] = extracted_date

            parsed_json["raw_wikitext_hash"] = wikitext_hash

            # Pydantic Schema Validation
            validated_payload = PatchExtractionPayload.model_validate(parsed_json)
            logger.info(f"Successfully validated {len(validated_payload.changes)} patch change items for version {validated_payload.version}.")
            return validated_payload

        except (ValidationError, Exception) as err:
            logger.error(f"Error during LLM extraction/validation: {err}. Falling back to heuristic mock parser.")
            return self.mock_parse_wikitext(raw_wikitext, version=version, date_str=extracted_date)

    @staticmethod
    def _clean_and_parse_json(response_text: str) -> Dict[str, Any]:
        """
        Strips markdown code fences and parses raw JSON string.
        """
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            cleaned = cleaned.strip()
        
        return json.loads(cleaned)

    def mock_parse_wikitext(self, raw_wikitext: str, version: str = "8.11", date_str: Optional[str] = None) -> PatchExtractionPayload:
        """
        Heuristic offline fallback parser that demonstrates the Pydantic schema
        and Bug Fix Paradigm classification without requiring active network API calls.
        """
        wikitext_hash = hashlib.sha256(raw_wikitext.encode('utf-8')).hexdigest()
        clean_text = self.preprocess_wikitext(raw_wikitext)
        extracted_date = date_str or self.extract_infobox_date(raw_wikitext) or "2024-06-11"

        KNOWN_AGENTS = {
            "Astra", "Breach", "Brimstone", "Chamber", "Clove", "Cypher", "Deadlock", "Fade",
            "Gekko", "Harbor", "Iso", "Jett", "KAY/O", "Kayo", "Killjoy", "Neon", "Omen",
            "Phoenix", "Raze", "Reyna", "Sage", "Skye", "Sova", "Tejo", "Viper", "Vyse", "Yoru"
        }
        KNOWN_WEAPONS = {
            "Ares", "Bucky", "Bulldog", "Classic", "Frenzy", "Ghost", "Guardian", "Judge",
            "Marshal", "Operator", "Outlaw", "Phantom", "Sheriff", "Shorty", "Spectre", "Stinger", "Vandal"
        }
        KNOWN_SUBJECTS = KNOWN_AGENTS | KNOWN_WEAPONS

        changes: List[PatchChangeItem] = []
        current_agent = None
        current_ability = "General"

        mechanical_keywords = [
            "slide", "cancel", "boost", "velocity", "momentum", "collision",
            "physics", "trajectory", "out of bounds", "wall clip", "animation speed",
            "teleport collision", "satchel cancel", "hitbox", "exploit"
        ]

        lines = clean_text.split('\n')

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Heading detection (=== Agent/Weapon Name ===)
            heading_match = re.match(r'^(==+)\s*(.+?)\s*(==+)$', line_str)
            if heading_match:
                title = heading_match.group(2).strip()
                matched_subj = None
                for subj in KNOWN_SUBJECTS:
                    if subj.lower() in title.lower():
                        matched_subj = subj
                        break
                if matched_subj:
                    current_agent = matched_subj
                    current_ability = "General"
                elif title.lower() in ["agent updates", "weapon updates"]:
                    current_agent = None
                    current_ability = "General"
                continue

            # Bullet detection (depth 1 or depth 2)
            bullet_match = re.match(r'^(\*+)\s*(.*)$', line_str)
            if bullet_match:
                depth = len(bullet_match.group(1))
                bullet_text = bullet_match.group(2).strip()

                if not bullet_text:
                    continue

                # Check if bullet depth 1 is an Agent/Weapon name
                matched_subj = None
                for subj in KNOWN_SUBJECTS:
                    if subj.lower() == bullet_text.lower() or bullet_text.lower().startswith(subj.lower()):
                        matched_subj = subj
                        break

                if depth == 1 and matched_subj:
                    current_agent = matched_subj
                    current_ability = "General"
                    continue
                elif depth == 1 and current_agent and not any(kw in bullet_text.lower() for kw in ["fixed", "bug", "decreased", "increased", "changed"]):
                    # Sub-ability heading under agent (e.g. * High Gear)
                    current_ability = bullet_text
                    continue

                # If current_agent is not set, attempt to infer agent from bullet_text
                target_agent = current_agent
                if not target_agent or target_agent not in KNOWN_SUBJECTS:
                    for ag in KNOWN_AGENTS:
                        if ag.lower() in bullet_text.lower():
                            target_agent = ag
                            break

                # Skip general non-agent non-gameplay bullets
                if not target_agent:
                    continue

                # Extract numeric transition (X >>> Y or X -> Y or increased from X to Y)
                transition_match = re.search(
                    r'((?:\d+(?:\.\d+)?|\.\d+))\s*%?\s*(?:>>>|->|to)\s*((?:\d+(?:\.\d+)?|\.\d+))\s*%?',
                    bullet_text,
                    re.IGNORECASE
                )

                old_val, new_val = None, None
                if transition_match:
                    try:
                        old_val = float(transition_match.group(1))
                        new_val = float(transition_match.group(2))
                    except ValueError:
                        old_val = transition_match.group(1)
                        new_val = transition_match.group(2)

                # Bug Fix Paradigm Classification
                is_bug_fix = "fixed" in bullet_text.lower() or "bug" in bullet_text.lower() or "issue" in bullet_text.lower()
                is_mech = False

                if is_bug_fix:
                    if any(kw in bullet_text.lower() for kw in mechanical_keywords):
                        is_mech = True
                else:
                    is_mech = False

                stat_name = bullet_text.split('.')[0] if '.' in bullet_text else bullet_text
                if len(stat_name) > 60:
                    stat_name = stat_name[:57] + "..."

                item = PatchChangeItem(
                    agent=target_agent,
                    ability=current_ability,
                    stat_modified=stat_name,
                    old_value=old_val,
                    new_value=new_val,
                    is_mechanical_removal=is_mech,
                    raw_evidence=bullet_text
                )
                changes.append(item)

        return PatchExtractionPayload(
            version=version or "8.11",
            date=extracted_date,
            changes=changes,
            raw_wikitext_hash=wikitext_hash
        )


# ============================================================================
# CLI TEST ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    sample_wikitext = """
    {{Infobox patch
    | version = 8.11
    | date = June 11, 2024
    }}
    == Agent Updates ==
    === Neon ===
    * High Gear
    ** Slide speed decreased from 1.0 >>> 0.8.
    ** Fixed a bug where Neon could execute an unintended double slide boost when cancelling animation.
    
    === Omen ===
    * Dark Cover
    ** Fixed an issue where Dark Cover audio loop would play continuously after round ends.
    """

    parser = V8PatchParser(force_offline_mock=True)
    payload = parser.parse_wikitext(sample_wikitext, version="8.11")
    
    print("--- EXTRACTED PATCH PAYLOAD (JSON) ---")
    print(payload.model_dump_json(indent=2))
