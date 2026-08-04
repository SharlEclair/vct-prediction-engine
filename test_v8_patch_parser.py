"""
test_v8_patch_parser.py
------------------------
Unit and Integration tests for v8_patch_parser.py.
Verifies Pydantic schema validation, Bug Fix Paradigm classification (is_mechanical_removal),
and fallback/offline parsing.
"""

import pytest
from pydantic import ValidationError
from v8_patch_parser import PatchChangeItem, PatchExtractionPayload, V8PatchParser


def test_pydantic_schema_validation_valid():
    """Verifies that a well-formed dictionary validates cleanly into PatchChangeItem."""
    data = {
        "agent": "Neon",
        "ability": "High Gear",
        "stat_modified": "Slide Speed",
        "old_value": 1.0,
        "new_value": 0.8,
        "is_mechanical_removal": False,
        "raw_evidence": "Slide speed decreased from 1.0 >>> 0.8."
    }
    item = PatchChangeItem.model_validate(data)
    assert item.agent == "Neon"
    assert item.ability == "High Gear"
    assert item.old_value == 1.0
    assert item.new_value == 0.8
    assert item.is_mechanical_removal is False


def test_pydantic_schema_validation_missing_critical_field():
    """Verifies that missing is_mechanical_removal raises a ValidationError."""
    data = {
        "agent": "Neon",
        "ability": "High Gear",
        "stat_modified": "Slide Speed"
        # Missing is_mechanical_removal
    }
    with pytest.raises(ValidationError):
        PatchChangeItem.model_validate(data)


def test_bug_fix_paradigm_classification():
    """
    Tests Bug Fix Paradigm classification:
    - Mechanical fixes (velocity, slide, animation cancels) -> is_mechanical_removal = True
    - Audio/UI fixes -> is_mechanical_removal = False
    """
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
    
    === Yoru ===
    * Gatecrash
    ** Fixed bug allowing player to teleport through solid wall collision on Ascent B Site.
    """

    parser = V8PatchParser(force_offline_mock=True)
    payload = parser.parse_wikitext(sample_wikitext, version="8.11")

    assert payload.version == "8.11"
    assert payload.date == "2024-06-11"
    assert len(payload.changes) == 4

    neon_slide_stat = payload.changes[0]
    assert neon_slide_stat.agent == "Neon"
    assert neon_slide_stat.old_value == 1.0
    assert neon_slide_stat.new_value == 0.8
    assert neon_slide_stat.is_mechanical_removal is False

    neon_slide_bug = payload.changes[1]
    assert neon_slide_bug.agent == "Neon"
    assert neon_slide_bug.is_mechanical_removal is True  # Double slide boost removal!

    omen_audio_bug = payload.changes[2]
    assert omen_audio_bug.agent == "Omen"
    assert omen_audio_bug.is_mechanical_removal is False  # Audio loop glitch!

    yoru_wall_bug = payload.changes[3]
    assert yoru_wall_bug.agent == "Yoru"
    assert yoru_wall_bug.is_mechanical_removal is True  # Wall collision glitch!

    print("ALL BUG FIX PARADIGM TESTS PASSED!")


def test_infobox_date_extraction():
    """Verifies parsing of patch date from infobox header."""
    wikitext = """
    {{Infobox patch
    | version = 9.01
    | date = July 16, 2024
    }}
    """
    parser = V8PatchParser(force_offline_mock=True)
    date_str = parser.extract_infobox_date(wikitext)
    assert date_str == "2024-07-16"


def test_kayo_canonicalization():
    """Verifies that any variation of Kayo ('Kayo', 'kayo', 'KAYO', 'Kay/o') is forcibly normalized to 'KAY/O'."""
    item1 = PatchChangeItem.model_validate({
        "agent": "Kayo",
        "ability": "FRAG/ment",
        "stat_modified": "Damage",
        "is_mechanical_removal": False
    })
    assert item1.agent == "KAY/O"

    item2 = PatchChangeItem.model_validate({
        "agent": "kayo",
        "ability": "FLASH/drive",
        "stat_modified": "Duration",
        "is_mechanical_removal": False
    })
    assert item2.agent == "KAY/O"

    wikitext = """
    == Agent Updates ==
    === Kayo ===
    * FRAG/ment
    ** Damage decreased from 50 >>> 40.
    """
    parser = V8PatchParser(force_offline_mock=True)
    payload = parser.parse_wikitext(wikitext, version="8.11")
    assert len(payload.changes) == 1
    assert payload.changes[0].agent == "KAY/O"


if __name__ == "__main__":
    test_pydantic_schema_validation_valid()
    test_pydantic_schema_validation_missing_critical_field()
    test_bug_fix_paradigm_classification()
    test_infobox_date_extraction()
    test_kayo_canonicalization()
    print("ALL UNIT TESTS PASSED SUCCESSFULLY!")
