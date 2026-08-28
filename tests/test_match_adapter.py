import os
import sys
import unittest
import json
from datetime import datetime

# Ensure root workspace is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.match_adapter import normalize_match, validate_normalized_match
from utils.team_registry import resolve_team_info, resolve_team_id

class TestMatchAdapter(unittest.TestCase):

    def test_schema_v1_normalization(self):
        v1_sample = {
            "schema_version": "1.0",
            "scraper_version": "2026-08-06",
            "source": {
                "site": "vlr.gg",
                "match_url": "https://www.vlr.gg/712822",
                "scraped_at": "2026-08-06T12:32:54Z"
            },
            "match_id": "712822",
            "overview": {
                "metadata": {
                    "date": "2026-07-26T11:00:00Z",
                    "patch": "13.01",
                    "event": "VCT 2026: EMEA Stage 2",
                    "teams": {"team1": "BBL Esports", "team2": "Natus Vincere"},
                    "score": {"team1_score": 2, "team2_score": 1, "BBL Esports": 2, "Natus Vincere": 1},
                    "maps_played": [{"map_id": "haven", "map_name": "Haven"}]
                },
                "segments": [
                    {
                        "map_id": "haven",
                        "map_name": "Haven",
                        "winner": "Natus Vincere",
                        "score": {"team1_score": 11, "team2_score": 13, "BBL Esports": 11, "Natus Vincere": 13},
                        "players": [{"name": "Loita", "team": "BBL Esports", "kills": 15}],
                        "round_history": [{"round_num": 1, "winner": "BBL Esports", "side": "ct"}],
                        "vetoes": ["BBL ban Split", "NAVI pick Haven"]
                    }
                ]
            },
            "performance": {"maps": {"haven": {"player_stats": {}}}},
            "economy": {"maps": {"haven": {"economy_summary": {}}}}
        }

        norm = normalize_match(v1_sample)
        self.assertEqual(norm["match_id"], "712822")
        self.assertEqual(norm["_adapter"]["source_schema"], "1.0")
        self.assertEqual(norm["teams"]["team1"], "BBL Esports")
        self.assertEqual(norm["teams"]["team2"], "Natus Vincere")
        self.assertEqual(norm["score"]["BBL Esports"], 2)
        self.assertEqual(norm["maps"][0]["map_name"], "Haven")
        self.assertEqual(norm["maps"][0]["winner"], "Natus Vincere")
        self.assertIsNotNone(norm["performance"])
        self.assertIsNotNone(norm["economy"])

    def test_gen2_normalization(self):
        gen2_sample = {
            "data": {
                "status": 200,
                "segments": [
                    {
                        "match_id": "670471",
                        "team1": "Paper Rex",
                        "team2": "LEVIATÁN",
                        "map": "Haven PICK 1:11:54",
                        "winner": "Paper Rex",
                        "date": "Sunday, July 27 1:00 AM Patch 13.01",
                        "players": [{"name": "something", "team": "Paper Rex", "kills": 20}],
                        "round_history": [{"round_num": 1, "winner": "Paper Rex", "side": "t"}]
                    }
                ]
            }
        }

        norm = normalize_match(gen2_sample)
        self.assertEqual(norm["match_id"], "670471")
        self.assertEqual(norm["_adapter"]["source_schema"], "gen2")
        self.assertEqual(norm["teams"]["team1"], "Paper Rex")
        self.assertEqual(norm["teams"]["team2"], "LEVIATÁN")
        self.assertEqual(norm["maps"][0]["map_name"], "Haven")
        self.assertEqual(norm["maps"][0]["winner"], "Paper Rex")

    def test_gen1_normalization(self):
        gen1_sample = {
            "data": {
                "segments": [
                    {
                        "id": 353198,
                        "date": "2024-06-15",
                        "teams": [
                            {"name": "Sentinels", "score": 2},
                            {"name": "Fnatic", "score": 0}
                        ],
                        "map_vetos": "Sentinels ban Split; Fnatic pick Haven",
                        "maps": [
                            {
                                "map_name": "Haven",
                                "score": {"team1": 13, "team2": 9},
                                "winner": "Sentinels"
                            }
                        ]
                    }
                ]
            }
        }

        norm = normalize_match(gen1_sample)
        self.assertEqual(norm["match_id"], "353198")
        self.assertEqual(norm["_adapter"]["source_schema"], "gen1")
        self.assertEqual(norm["teams"]["team1"], "Sentinels")
        self.assertEqual(norm["teams"]["team2"], "Fnatic")
        self.assertEqual(norm["maps"][0]["map_name"], "Haven")
        self.assertEqual(norm["maps"][0]["vetoes"], ["Sentinels ban Split", "Fnatic pick Haven"])

    def test_team_registry(self):
        self.assertEqual(resolve_team_id("Paper Rex"), 624)
        self.assertEqual(resolve_team_id("PRX"), 624)
        self.assertEqual(resolve_team_id("NAVI"), 4915)
        info = resolve_team_info("LEVIATÁN")
        self.assertEqual(info["vlr_id"], 2359)

if __name__ == "__main__":
    unittest.main()
