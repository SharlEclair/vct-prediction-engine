# Side-Conditioned Markov Simulator (Sub-Model 3)

## 1. Map Side Advantage Registry

Historical 2026 VCT data shows massive disparities in side win rates. The simulator must apply a `side_bias` baseline parameter for each map.

**Defender-Sided Maps:**

* Ascent (+4.4% Def)
* Split (+3.0% Def)
* Summit (+2.2% Def)

**Attacker-Sided Maps:**

* Lotus (+2.8% Atk)
* Abyss (+4.3% Atk)

## 2. Markov State Transition Logic

Let the current game state at Round $t$ be defined by the score, economy, and Team A's side (Defense or Attack). The probability of Team A winning the round is calculated via a logistic function:

$$
P(\text{Win}_{A, t}) = \frac{1}{1 + \exp(-Z)}
$$

Where the log-odds $Z$ is computed as:

$$
Z = \beta_0 + \beta_1(\text{ACS}_A - \text{ACS}_B) + \beta_2(\text{Eco}_A - \text{Eco}_B) + \beta_3(\text{SideAdvantage}_A)
$$

$\text{SideAdvantage}_A$: If Team A is on Defense on Ascent, this is +0.044. If Team A is on Attack on Ascent, this is -0.044.

## 3. Economy and Side-Swap Mechanics

**Economy Reset:** The winner of round $t$ receives an Eco boost for $t+1$. The loser receives a compounding loss bonus that caps out.

**Round 13 Swap:** When $t = 13$ (Score A + Score B = 12), the side alignment strictly flips. If Team A was Defense, they are now Attack. Multiply the $\text{SideAdvantage}_A$ by $-1$.

**Overtime (12-12):** If the score reaches 12-12, the required win threshold shifts from 13 to a "win-by-two" margin. Side advantage is neutralized ($\text{SideAdvantage} = 0$) or alternated every round until a terminal state is reached.
