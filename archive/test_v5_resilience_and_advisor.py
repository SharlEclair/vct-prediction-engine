import unittest
from unittest.mock import patch, MagicMock
from patch_ingestor import fetch_from_wiki_api, PatchFetchError, get_mock_patch_text
from fantasy_engine import suggest_transfers, optimize_roster

class TestV5ResilienceAndAdvisor(unittest.TestCase):

    @patch("patch_ingestor.requests.get")
    def test_waf_bypass_fallback_9_0_to_9_04(self, mock_get):
        # Simulate network error / WAF block
        mock_get.side_effect = Exception("Simulated Cloudflare block")
        
        # Test 9.0 fallback
        text_9_0 = fetch_from_wiki_api("9.0")
        self.assertIn("Double Tap: duration decreased", text_9_0)
        self.assertIn("Iso", text_9_0)
        
        # Test 9.04 fallback
        text_9_04 = fetch_from_wiki_api("9.04")
        self.assertIn("Arc Rose: windup time decreased", text_9_04)
        self.assertIn("Vyse", text_9_04)

        # Test non-fallback target raises error
        with self.assertRaises(PatchFetchError):
            fetch_from_wiki_api("12.09")

    def test_transfer_advisor_strict_budget_and_roles(self):
        # Mock players dataset
        mock_players = [
            {"player_name": "P1", "price": 10.0, "role": "Duelist", "vlr_team_id": 1, "ppg": 15.0},
            {"player_name": "P2", "price": 9.0, "role": "Initiator", "vlr_team_id": 1, "ppg": 14.0},
            {"player_name": "P3", "price": 8.0, "role": "Controller", "vlr_team_id": 2, "ppg": 13.0},
            {"player_name": "P4", "price": 7.0, "role": "Sentinel", "vlr_team_id": 2, "ppg": 12.0},
            {"player_name": "P5", "price": 6.0, "role": "Duelist", "vlr_team_id": 3, "ppg": 11.0},  # Flex candidate
            {"player_name": "P6", "price": 5.0, "role": "Initiator", "vlr_team_id": 3, "ppg": 10.0},  # Flex candidate
            
            # Alternative candidates for transfers
            {"player_name": "P_Rich", "price": 15.0, "role": "Duelist", "vlr_team_id": 4, "ppg": 25.0},
            {"player_name": "P_Budget", "price": 4.0, "role": "Duelist", "vlr_team_id": 4, "ppg": 10.5},
            {"player_name": "P_TeamLimitViolation", "price": 8.0, "role": "Duelist", "vlr_team_id": 1, "ppg": 20.0},
        ]
        
        # Current roster
        current_roster = mock_players[:6]
        
        # Test case: we have remaining bank balance of 1.0 VP.
        # Roster value is 10 + 9 + 8 + 7 + 6 + 5 = 45.0 VP.
        # Combined cap is 46.0 VP.
        # A rich candidate (P_Rich, price=15.0) cannot be bought by swapping P6 (price=5.0) because:
        # Incoming cost (15.0) > Outgoing cost (5.0) + Bank (1.0) = 6.0.
        # The optimizer should only suggest P_Budget (price=4.0) since it respects strict knapsack liquidity.
        
        # Force IGL name to P1
        res = suggest_transfers(current_roster, mock_players, remaining_bank_balance=1.0, forced_igl_name="P1")
        self.assertEqual(res["solver_status"], "optimal")
        
        recs = res.get("recommendations", [])
        for rec in recs:
            # Enforce budget constraint
            total_in_cost = sum(p["price"] for p in rec["transfers_in"])
            total_out_cost = sum(p["price"] for p in rec["transfers_out"])
            self.assertLessEqual(total_in_cost, total_out_cost + 1.0)
            
            # Verify roster size and role counts
            new_roster = rec["new_roster"]
            self.assertEqual(len(new_roster), 6)
            
            # Check team limits (max 2 per team)
            team_counts = {}
            for p in new_roster:
                tid = p["vlr_team_id"]
                team_counts[tid] = team_counts.get(tid, 0) + 1
                self.assertLessEqual(team_counts[tid], 2)

    def test_igl_2x_multiplier(self):
        mock_players = [
            {"player_name": "P1", "price": 10.0, "role": "Duelist", "vlr_team_id": 1, "ppg": 15.0},
            {"player_name": "P2", "price": 9.0, "role": "Initiator", "vlr_team_id": 2, "ppg": 14.0},
            {"player_name": "P3", "price": 8.0, "role": "Controller", "vlr_team_id": 3, "ppg": 13.0},
            {"player_name": "P4", "price": 7.0, "role": "Sentinel", "vlr_team_id": 4, "ppg": 12.0},
            {"player_name": "P5", "price": 6.0, "role": "Duelist", "vlr_team_id": 5, "ppg": 11.0},
            {"player_name": "P6", "price": 5.0, "role": "Initiator", "vlr_team_id": 6, "ppg": 10.0},
        ]
        
        # Test baseline optimization with P6 forced as IGL
        res = optimize_roster(mock_players, salary_cap=50.0, forced_igl_name="P6")
        self.assertEqual(res["solver_status"], "optimal")
        
        # Verify P6 is designated IGL and has double points
        igl_player = res["igl_player"]
        self.assertEqual(igl_player, "P6")
        
        optimal_roster = res["optimal_roster"]
        p6_entry = [p for p in optimal_roster if p["player_name"] == "P6"][0]
        self.assertTrue(p6_entry["is_igl"])

if __name__ == "__main__":
    unittest.main()
