Tasks:
- Create button to scrape VFL Game week schedule from "https://api.valorantfantasyleague.net/api/matches/schedule?gameweek=2&eventId=10"

here, 
2 for game week
10 for event id which is VCT 2026 Stage 2

- Add text boxes to enter Game week to scrape from the webui, and Event id to be used when scraping (use 2, 10 as default)

- Create button to select the gameweek to be used to make the VLF roster optimization predictions. For example, if game week 2 is selected, the teams playing in game week 2 must be used to make the predictions.

- also create similar gameweek selection button for 3 player transfer optimizer.

- game week selection button should be the default, and there should be a checkbox for manual Active Team Pool Filter, only using those manually selected teams for preds if the checkbox is selected. Also create this for 3 transfer predictions.

# JSON Structure

```text
Root
└── matches (Array<Match>)
    ├── Match
    │   ├── id (string)
    │   ├── team1 (Team)
    │   │   ├── id (string)
    │   │   ├── name (string)
    │   │   ├── shortName (string)
    │   │   ├── imagePath (string)
    │   │   ├── nextOpponentId (string | null)
    │   │   └── nextOpponent (Team | null)
    │   ├── team2 (Team)
    │   │   ├── id (string)
    │   │   ├── name (string)
    │   │   ├── shortName (string)
    │   │   ├── imagePath (string)
    │   │   ├── nextOpponentId (string | null)
    │   │   └── nextOpponent (Team | null)
    │   ├── matchLink (string)
    │   ├── scheduledMatchTime (string)
    │   └── isComplete (boolean)
    ├── Match
    └── ...
```

## Object Definitions

### Match

| Field | Type | Description |
|--------|------|-------------|
| `id` | `string` | Unique match identifier |
| `team1` | `Team` | First team |
| `team2` | `Team` | Second team |
| `matchLink` | `string` | URL to the match page |
| `scheduledMatchTime` | `string` | Unix timestamp (seconds) |
| `isComplete` | `boolean` | Whether the match has finished |

### Team

| Field | Type | Description |
|--------|------|-------------|
| `id` | `string` | Unique team identifier |
| `name` | `string` | Full team name |
| `shortName` | `string` | Team abbreviation |
| `imagePath` | `string` | Relative path to team logo |
| `nextOpponentId` | `string \| null` | ID of next opponent, if known |
| `nextOpponent` | `Team \| null` | Next opponent object, if available |
