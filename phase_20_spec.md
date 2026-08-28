Tasks:
- in the 3 transfer advisor, the format has to be changed so that:
- there are 5 drop down selection boxes for selecting the players (Duelist, Initiator, Controller, Sentinel, and one for wildcard)
- in the dropdown for duelist, only list of duelist players should be shown, except for the ones already selected
- in the dropdown for initiator, only list of initiator players should be shown, except for the ones already selected
- in the dropdown for controller, only list of controller players should be shown, except for the ones already selected
- in the dropdown for sentinel, only list of sentinel players should be shown, except for the ones already selected
- in the dropdown for wildcard, all  players should be shown, except for the ones already selected
- when a player is selected in one of the dropdowns, it should be removed from the other dropdowns

the logic for 3 player transfer suggestion should be such that the duelist players are only replaced by duelist players, initiator by initiator, controller by controller, sentinel by sentinel, and wildcard can be replaced by any player. Given that it follows other restrictions such as total budget of the roster, as well as 2 players from same team, etc. it should also suggest the new IGL (IGL should have the highest score potential)