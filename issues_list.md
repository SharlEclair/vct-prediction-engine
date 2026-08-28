# v6 Branch

## Players Ingestion

* When a team is added (e.g. **MIBR**), the player roster should be created correctly. For example, **MIBR** should have the following players:

  * `zekken`
  * `tex`
  * `v1xen`
  * `Mazino`
  * `Verno`
  * `aspas`
* Some organizations have academy teams (e.g. **MIBR Academy**). Players from academy teams **must not** be added to the main team's roster.
* Some organizations have additional rosters, such as Game Changers teams (e.g. **100 Thieves GC**). These players **must not** be added to the main team's roster.
* Some organizations have inactive players in addition to their active roster. For example, **ENVY** has the following inactive players:

  * `ion`
  * `P0PPIN`
  * `Eggsterr`

  Their active roster is:

  * `Demon1`
  * `Rossy`
  * `keznit`
  * `GLYPH`
  * `nightz`

  Add a feature to the Web UI that allows team rosters to be manually updated to handle these cases.

---

# Map Veto Information

## Best-of-1 (BO1)

The team displayed on the left side of the match page decides whether they will be **Team A** or **Team B**.

Veto order:

1. Team A bans 1 map.
2. Team B bans 1 map.
3. Team A bans 1 map.
4. Team B bans 1 map.
5. Team A bans 1 map.
6. Team B selects Map 1.
7. Team A selects the starting side for Map 1.

---

## Best-of-3 (BO3)

The higher-seeded team (based on the Swiss Stage results) decides whether they will be **Team A** or **Team B**.

Veto order:

1. Team A bans 1 map.
2. Team B bans 1 map.
3. Team A selects Map 1.
4. Team B selects the starting side for Map 1.
5. Team B selects Map 2.
6. Team A selects the starting side for Map 2.
7. Team A bans 1 map.
8. Team B bans 1 map.
9. The remaining map becomes Map 3.
10. Team A selects the starting side for Map 3.

---

## Best-of-5 (BO5)

The Upper Bracket winner decides whether they will be **Team A** or **Team B**.

Veto order:

1. Team A bans 1 map.
2. Team B bans 1 map.
3. Team A selects Map 1.
4. Team B selects the starting side for Map 1.
5. Team B selects Map 2.
6. Team A selects the starting side for Map 2.
7. Team A selects Map 3.
8. Team B selects the starting side for Map 3.
9. Team B selects Map 4.
10. Team A selects the starting side for Map 4.
11. The remaining map becomes Map 5.
12. Team B selects the starting side for Map 5.

---

## Additional Map Veto Rules

* If two teams are equally seeded (i.e. both are at the same stage of the tournament), a **Skirmish** is played.
* The winner of the **Skirmish** chooses whether they will be **Team A** or **Team B** for the map veto process.
