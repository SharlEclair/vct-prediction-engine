# Architectural Blueprint for the Hybrid Micro Engine: Fusing Direct EV Regression with Monte Carlo Simulation in Valorant DFS

## 1. Executive Context and the Pathology of the Error Cascade

The transition from a macro-level predictive framework to a highly granular, event-driven simulation engine represents a critical evolutionary step in quantitative sports analytics, particularly within the nascent field of esports. In the context of the Valorant Champions Tour (VCT), the development of the V5 Bottom-Up Micro Simulation Engine was an ambitious attempt to natively capture the non-linear mechanics of tactical shooters.

By utilizing a Directed Acyclic Graph (DAG) architecture, the V5 engine deconstructed the game state into sequential nodes:

* Contextual Bandit Map Veto with Temporal Map Pools
* Hungarian Algorithm Agent Draft
* Side-Conditioned Markov Round Simulator
* Dirichlet Regression for Kill Share allocation

The primary mathematical objective of this architecture was to generate a massive, localized Monte Carlo probability distribution that accurately reflected extreme statistical ceilings. In professional Valorant, outcomes are inherently volatile and structurally bounded. Matches can result in 13-0 sweep bonuses or gruelling overtime extensions that artificially inflate raw statistics. Furthermore, player performance exhibits complex covariance structures; for instance, the success of a Duelist entry-fragger is intrinsically linked to the utility deployment of their Initiator teammate. Capturing these tail-end events and correlations is mathematically indispensable for optimizing Guaranteed Prize Pool (GPP) Daily Fantasy Sports (DFS) lineups, where capturing covariance and ceiling projections dictates tournament win equity.

However, rigorous backtesting against a holdout sample of 100 Tier 1 matches from the 2026 season revealed a catastrophic systemic vulnerability within the V5 architecture: the "Error Cascade". Because the architecture operated as a sequential DAG, the terminal outputs—specifically individual player statistics—were entirely conditionally dependent on the accuracy of the root nodes.

The mathematical fragility of this system can be understood through the lens of joint probability. The probability of an exact sequence of events occurring in the match is the product of its conditional probabilities:

$$
P(Match) = P(Map) \cdot P(Draft \mid Map) \cdot P(Rounds \mid Draft, Map) \cdot P(Stats \mid Rounds, Draft, Map)
$$

In the V5 backtest, the root node—the Contextual Bandit Map Veto—achieved a predictive accuracy of only 18.3%. This root failure propagated multiplicatively through the graph. Because the Map Veto was incorrect in over 81% of simulations, the Hungarian algorithm drafted optimal agents for the wrong topological arena. Consequently, the Markov chain simulated rounds based on inaccurate agent synergies, and the Dirichlet regression distributed kills based on incorrect entry/anchor role assumptions.

The empirical result was a Match Winner Brier Score of 0.2565—a metric demonstrating performance worse than a naive coin-flip baseline of 0.2500. Most alarmingly, the Player Kill Mean Absolute Error (MAE) swelled to 6.94 kills per map. A "Naive Baseline" model, utilizing simple historical career averages, dramatically outperformed the state-of-the-art DAG with an MAE of 4.37.

Despite this catastrophic failure in mean accuracy, the mandate is clear: the Bottom-Up Monte Carlo engine must be retained. It is the only architectural framework capable of preserving the joint probability distributions, structural covariance, and combinatorial upside required by the downstream linear programming Knapsack solver. However, to salvage the pipeline, the system must be mathematically anchored by a Top-Down Micro approach—a Direct Expected Value (EV) Regression. This hybrid approach establishes stable, highly accurate baselines while entirely bypassing the sequential fragility of the DAG.

---

## 2. Cross-Sport Algorithmic Analogues: Lessons from MLB and NFL DFS

To architect a robust Hybrid Micro Engine for Valorant, it is imperative to examine advanced ensembling techniques deployed by quantitative syndicates in traditional Daily Fantasy Sports, specifically Major League Baseball (MLB) and the National Football League (NFL). These sports share profound structural similarities with tactical esports: they are highly discrete, heavily dependent on specific matchups, and generate massive covariance between teammates.

### 2.1 The MLB Paradigm: Isolating Marginals from the Copula

In MLB DFS, predicting a player's fantasy output requires navigating immense variance. A batter might strike out four times in one game and hit two home runs in the next. Leading quantitative platforms separate the prediction of a player's baseline expectation from the simulation of the game environment.

Rather than relying on a pure bottom-up simulation to determine how good a player is, these models utilize advanced sabermetrics (e.g., projected wOBA, exit velocity, and park factors) in a Top-Down regression model to establish a highly stable mean projection and standard deviation for every player. This Top-Down output acts as the "Marginal Distribution."

Simultaneously, a Monte Carlo simulation is executed at the play-by-play level. However, the simulation is not trusted to dictate the player's true skill; rather, it is used strictly to establish the "Empirical Copula"—the underlying dependence structure of the game. For example, if the leadoff hitter reaches base, the probability of the second hitter scoring a run increases drastically.

By sampling from the empirical copula derived from historical or simulated data, the model extracts a vector of quantile values. These quantiles are then passed through the inverse Cumulative Distribution Function (CDF) of the Top-Down marginal distributions.

This exact methodology—relegating the simulation to determine correlation while trusting the regressor to determine value—is the exact mathematical blueprint required to fix the VCT Error Cascade.

### 2.2 The NFL Paradigm: Covariance Stacking and Distribution Shifting

In NFL DFS, the covariance between players is even more pronounced. The relationship between a Quarterback and their primary Wide Receiver is strongly positive; every passing yard and touchdown accrued by the receiver is simultaneously credited to the quarterback. Conversely, a running back's production may be negatively correlated with a pass-heavy game script.

Advanced NFL DFS simulators, such as SaberSim, utilize a methodology known as predictive distribution shifting or mean matching. When these platforms run a 10,000-iteration Monte Carlo simulation of an NFL game, the resulting raw mean of a player's fantasy points often deviates from the highly accurate Top-Down consensus projection.

If the simulator predicts an average of 15.0 points for a receiver, but the stable Top-Down regression model predicts 18.5 points, the simulator does not discard the Monte Carlo data. Instead, it calculates the delta (+3.5 points) and shifts the entire 10,000-iteration distribution along the x-axis to perfectly match the Top-Down mean.

This allows the DFS optimizer to access the realistic ceiling, floor, and exact correlation coefficients of the simulation, while remaining securely anchored to the superior accuracy of the continuous regressor. In the context of Valorant, where a Sentinel anchoring a bomb site relies on entirely different statistical dependencies than a Duelist aggressively taking space, this distribution-shifting technique is paramount.

The following table summarizes the cross-sport translation of DFS modeling techniques to the Valorant ecosystem:

| DFS Ecosystem  | Primary Covariance Dynamic | Top-Down Regressor Target     | Bottom-Up Simulation Role             | Valorant VCT Translation                         |
| -------------- | -------------------------- | ----------------------------- | ------------------------------------- | ------------------------------------------------ |
| MLB            | Batter / Pitcher Duel      | Player Sabermetrics (wOBA)    | Empirical Copula extraction           | Attacker vs. Defender Duel (Opening engagements) |
| NFL            | QB / WR Stacking           | Projected Expected Value (EV) | Distribution Shifting (Mean Matching) | Initiator / Duelist Synergy (Trade fragging)     |
| Valorant (VCT) | Tactical Role Economy      | Clipped EV per Round          | Covariance Matrix & 13-0 Ceilings     | The Hybrid Micro Engine                          |
## 3. Top-Down Feature Architecture: Anchoring the Direct EV Regressor

To construct a robust Top-Down EV Regressor that consistently outperforms the naive baseline MAE of 4.37, the model must forecast individual player performance directly, circumventing the Contextual Bandit Map Veto and Generative Draft stages entirely.

The objective is to produce a highly stable, continuous point estimate ($\mu_{TD}$) representing a player's expected fantasy output (e.g., Kills, First Bloods, Assists) across an aggregated expectation of all probable map and draft permutations.

For a tactical shooter like Valorant, the feature space must be engineered to isolate individual mechanical skill, recent form, and opponent suppression capabilities. The Regressor—ideally a Gradient Boosting framework such as XGBoost or LightGBM—must be fed features that are mathematically immune to the variance that plagued the V5 DAG.

### 3.1 Clipped Historical Baselines

In tactical esports, statistical distributions exhibit extreme skewness due to the "win-by-two" overtime constraints and "first-to-13" match truncation. A professional player might secure 35 kills in a grueling 18-16 overtime map, or merely 6 kills in a 13-0 sweep where the opposing team offered little resistance. Using raw arithmetic means to define a player's baseline incorporates these extreme structural outliers, resulting in high-variance projections that degrade the MAE and distort the underlying expectation of mechanical skill.

To rectify this, historical baselines must be "clipped" or Winsorized. Winsorization is a statistical transformation that limits extreme values in the dataset to reduce the effect of spurious outliers. For VCT metrics, individual match kills must first be transformed into a rate metric—specifically, Kills Per Round (KPR) or Average Combat Score (ACS) per round—before being aggregated.

By bounding historical KPR between the 5th and 95th percentiles of the global dataset, the baseline becomes immune to the volatility of extreme match lengths. For example, if a player's KPR in an anomalous 13-0 game falls below the 5th percentile threshold due to a lack of engagement opportunities, the value is clipped to the 5th percentile boundary. This provides the Gradient Boosting regressor with a highly stable target feature that accurately reflects the player's true mechanical floor and ceiling.

### 3.2 Exponential Moving Averages (EMA) for Player Form

Player performance in esports is highly non-stationary. Constant developer patch updates altering weapon recoil, agent utility costs, and map geometry induce rapid meta-shifts. Consequently, player form oscillates rapidly; a player who dominated during a "Chamber/Operator" meta may struggle significantly when the patch shifts to a "Cypher/Rifle" meta.

A simple rolling average (e.g., a trailing 30-day average) weights a match played yesterday equally with a match played four weeks ago. This fails entirely to capture immediate momentum, sudden slumps, or adaptation to recent patches. Therefore, the Top-Down regressor must rely heavily on Exponential Moving Averages (EMA), which apply weighting factors that decrease exponentially for older data points.

The EMA for a performance metric $X$ at time $t$ is defined recursively as:

$$
EMA_{t} = \alpha \cdot X_{t} + (1 - \alpha) \cdot EMA_{t-1}
$$

Where $\alpha$ is the smoothing factor ($0 < \alpha \leq 1$).

In the VCT context, multiple EMAs should be engineered using different half-lives (e.g., $\alpha$ tuned for 5-match, 15-match, and 30-match windows). By feeding the XGBoost model an array of these temporal windows, the regression algorithm gains a temporal gradient. It can mathematically detect whether a player's baseline is accelerating or decaying leading up to the current gameweek, adjusting the expected value dynamically without relying on the brittle Map Veto logic.

### 3.3 Opponent Defensive Ratings (ODR)

In zero-sum tactical shooters, an attacking player's expected output is strictly bounded by the opponent's defensive efficacy. Standard DFS models often rely on naive "Points Allowed by Position" metrics, which are deeply flawed because they fail to account for the strength of schedule. If Team A allows very few kills per match, it may be because they possess elite defensive synergy, or simply because their recent schedule consisted of mechanically inferior opponents.

To engineer an accurate Opponent Defensive Rating (ODR), the system must utilize a generalized Markov model or a Ridge-penalized mixed-effects regression. By formulating a massive system of linear equations where a team's actual kills achieved is a function of their underlying offensive strength minus the opponent's underlying defensive strength, the solver can isolate the true, schedule-adjusted ODR.

The baseline formula for this extraction is:

$$
Kills_{ij} = \mu_{league} + Offense_{i} - Defense_{j} + \epsilon_{ij}
$$

Where $Kills_{ij}$ represents the kills secured by Team $i$ against Team $j$, $\mu_{league}$ is the global average, and $\epsilon_{ij}$ is the residual error.

The extracted $Defense_{j}$ parameter serves as the pure ODR feature. When predicting Player $X$'s EV on Team $i$ against Team $j$, the XGBoost regressor leverages the ODR to mathematically suppress or elevate the player's Clipped Historical Baseline, ensuring the final $\mu_{TD}$ is contextually precise and accounts for opponent suppression.
## 4. The Mathematical Combiner: Fusing Top-Down EV with Bottom-Up Simulation

The central architectural challenge of the Hybrid Micro Engine is the synthesis of two fundamentally distinct mathematical objects.

First, the **Top-Down Output**: a highly stable, low-variance scalar Expected Value ($\mu_{TD}$) generated by the XGBoost regressor, representing the most accurate central tendency.

Second, the **Bottom-Up Output**: a high-variance, structurally sound $N$-dimensional probability distribution ($X_{MC}$) generated by the DAG Monte Carlo simulation (10,000 iterations), containing vital covariance structures, combinatorial DFS constraints, and heavy tails.

The mathematical combiner must preserve the exact mean of the Top-Down model while retaining the shape and covariance matrix of the Bottom-Up model. Several advanced ensembling techniques exist in quantitative sports analytics, but they are not equally viable for this specific, highly constrained environment.

### 4.1 Evaluating Inferior Methods: Bayesian Updating and Inverse Variance Weighting

#### The Failure of Bayesian Updating in the Error Cascade

In traditional sports forecasting, Bayesian updating is frequently deployed to fuse models. The stable, long-term regression model acts as the prior distribution, and the highly contextual, short-term simulation acts as the likelihood function. The posterior distribution is calculated via Bayes' Theorem, yielding a mathematically elegant synthesis of both models that updates iteratively as new information arrives.

However, deploying a Bayesian combiner in the V5 architecture is fundamentally flawed due to the established pathology of the Error Cascade. In this proposed framework, the Top-Down EV constitutes the prior, and the Bottom-Up DAG simulation constitutes the likelihood. Because the DAG suffers from an 18.3% Map Veto accuracy, its resulting conditional probability distributions (the likelihood) are heavily poisoned by alternate-reality game states.

Applying Bayesian updating would mathematically allow this flawed likelihood to drag the posterior mean away from the highly accurate Top-Down prior. Rather than fixing the simulation, Bayesian inference would average the truth with a hallucination, effectively re-introducing the Error Cascade into the final output and destroying the MAE benchmark.

#### The Limitation of Inverse Variance Weighting (IVW)

Inverse Variance Weighting is an ensembling technique that combines multiple independent predictions by weighting them inversely to their variance, mathematically minimizing the variance of the combined estimator. It is highly effective for combining scalar regressors or meta-analyses.

While mathematically sound for point-estimate prediction, IVW is structurally incompatible with a DFS optimization pipeline. IVW outputs a single, collapsed point estimate. A Knapsack linear programming solver tasked with maximizing GPP tournament upside requires full joint probability distributions to assess the correlation between players. For example, the solver must know that if an Initiator scores high, the corresponding Duelist is likely to score high. Collapsing the 10,000-iteration Monte Carlo distributions into IVW point estimates destroys the covariance matrix, rendering the Bottom-Up simulation entirely pointless and blinding the solver to tournament ceilings.

### 4.2 The Mathematically Optimal Solution: Copulas and Mean Shifting

To achieve the dual mandate of perfect Top-Down EV alignment and perfect Bottom-Up covariance preservation, the architecture must utilize a strict division of labor.

The Top-Down EV Regressor is solely responsible for the **Marginal Distributions** (the location and scale of the player's performance), while the Bottom-Up DAG Monte Carlo is solely responsible for the **Dependence Structure** (the Copula, capturing how players interact dynamically).

There are two primary methodologies to execute this fusion in a production environment.

#### Method A: Predictive Distribution Shifting (Mean Matching)

The most computationally efficient method for fusing these pipelines—widely utilized by commercial DFS simulators like SaberSim—is predictive mean shifting. This approach explicitly aligns the center of mass of the simulation with the stable regressor.

**Simulation Execution:** The DAG simulates 10,000 matches, resulting in an empirical distribution of fantasy points for Player $i$. Let the mean of this simulated distribution be $\mu_{MC, i}$.

**Regressor Execution:** The Top-Down regressor predicts a highly accurate Expected Value for Player $i$. Let this be $\mu_{TD, i}$.

**Delta Calculation:** The combiner calculates the variance offset:

$$
\Delta_i = \mu_{TD, i} - \mu_{MC, i}
$$

**Affine Transformation:** The entire 10,000-iteration simulated distribution for Player $i$ is mathematically shifted along the x-axis by $\Delta_i$.

This ensures that the expected value of the simulation perfectly matches the Top-Down EV:

$$
\mathbb{E}[X_{shifted}] = \mu_{TD}
$$

while the exact shape of the simulation—including the heavy tails generated by 13-0 sweeps and infinite overtimes—is perfectly preserved. Because the shift is applied as a uniform constant across all iterations simultaneously, the underlying rank-order correlation between teammates remains completely intact.

#### Method B: Gaussian Copulas via the Iman-Conover Algorithm

For a more rigorous, distribution-free mathematical fusion that perfectly preserves marginal integrity, the Iman-Conover algorithm is the quantitative industry standard for inducing a desired rank correlation onto a set of independent marginal distributions.

**Extract the Correlation Matrix:** Run the Bottom-Up DAG for 10,000 iterations. Extract the $10 \times 10$ Spearman rank correlation matrix ($\Sigma_{MC}$) representing the in-game covariance between all 10 players on the server. This matrix captures the zero-sum nature of the kill economy.

**Generate Parametric Marginals:** Using the $\mu_{TD}$ from the Top-Down EV Regressor and a variance parameter derived from historical deviations, define a continuous marginal probability distribution (e.g., a Gamma or Weibull distribution) for each player.

**Sample Independently:** Draw 10,000 independent, uncorrelated samples from each of the 10 Top-Down marginal distributions. Let this matrix be $M$.

**Iman-Conover Transformation:**

1. Compute the Van der Waerden normal scores for the columns of $M$, creating a score matrix $S$.
2. Uncorrelate $S$ by multiplying it by the inverse of its Cholesky decomposition, yielding an orthogonal matrix $Z$.
3. Compute the Cholesky decomposition of the target DAG correlation matrix $\Sigma_{MC}$, denoted as $P$.
4. Induce the target correlation by calculating:

$$
Y = Z \cdot P
$$

5. Finally, reorder the raw, independent samples in $M$ so that their rank order exactly matches the rank order of the columns in $Y$.

The resulting $10,000 \times 10$ matrix contains simulations that perfectly mirror the Top-Down EV predictions, while exhibiting the exact tactical correlations and game-theory limits dictated by the Bottom-Up DAG. This effectively bypasses the Error Cascade entirely; even if the Contextual Bandit hallucinates the wrong map, the Iman-Conover algorithm only extracts the underlying correlations of the simulation, binding them to the mathematically stable Top-Down baseline.
The following table summarizes the evaluation of the Mathematical Combiners:

| Ensembling Technique       | Mathematical Mechanism                                             | Suitability for VCT DFS Optimization                                                                                          | Verdict  |
| -------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- | -------- |
| Bayesian Updating          | Likelihood updates the Prior to form a Posterior                   | Fatal Flaw: The DAG likelihood is corrupted by the 18.3% Map Veto accuracy. It will pull the stable prior away from reality.  | Rejected |
| Inverse Variance Weighting | Weights independent predictions inversely to their variance        | Fatal Flaw: Collapses distributions into scalar point estimates, destroying the covariance matrix required by GPP solvers.    | Rejected |
| Mean Shifting              | Affine transformation of the Monte Carlo array by $\Delta_i$       | Highly Viable: Computationally fast. Matches the TD mean while perfectly preserving the simulation shape and raw covariance.  | Approved |
| Iman-Conover Copula        | Rank reordering independent TD marginals via the DAG Cholesky root | Optimal: Mathematically rigorous. Guarantees perfect marginal distributions while imposing exact structural DAG correlations. | Approved |

# 5. The Optimal Implementation Task List and Solver Integration

To transition the VCT architecture from the flawed V5 pipeline to the production-ready Hybrid Micro Engine, the engineering team must follow a strict, chronological implementation roadmap. This task list bridges the feature engineering, the mathematical combiner, and the final linear programming Knapsack solver.

## Phase 1: Top-Down Infrastructure and Feature Engineering

The foundational phase focuses entirely on bypassing the DAG to establish the stable EV baseline that will defeat the 4.37 Naive MAE benchmark.

| Task ID | Description                | Technical Specifications                                                                                                                                                                                               |
| ------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1.1** | Target Variable Generation | Ingest `/v2/match/details` telemetry. Extract raw kills, deaths, assists, and first bloods. Transform all absolute integers into rate metrics (e.g., Kills Per Round) to decouple from match length variance.          |
| **1.2** | Baseline Clipping          | Implement statistical clipping (Winsorization). Bound historical rate metrics between the 5th and 95th percentiles to mitigate outlier distortion from 13-0 sweeps or infinite overtimes.                              |
| **1.3** | EMA Construction           | Compute Exponential Moving Averages for core micro-stats. Generate multiple temporal windows ($\alpha = 0.1$ for slow decay, $\alpha = 0.4$ for rapid form detection) based on chronological match dates.              |
| **1.4** | ODR Matrix Generation      | Formulate the Opponent Defensive Rating using a Ridge-penalized regression solver across the trailing 6-month dataset. Output a continuous scalar representing expected kills suppressed per round for every VCT team. |
| **1.5** | Regressor Training         | Train a Gradient Boosting framework (XGBoost) using the engineered features to predict the continuous Expected Value ($\mu_{TD}$) for every player on the slate. Validate against the 4.37 Naive MAE benchmark.        |

## Phase 2: Bottom-Up Engine Pruning and Execution

The existing V5 DAG is not discarded; it is repurposed. Instead of treating its outputs as absolute truth, it is run strictly to generate structural boundaries, ceilings, and correlation matrices for the Copula.

| Task ID | Description           | Technical Specifications                                                                                                                                                                                                                        |
| ------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **2.1** | DAG Execution         | Execute the V5 pipeline (Contextual Bandit Veto $\rightarrow$ Hungarian Draft $\rightarrow$ Markov Round Sim $\rightarrow$ Dirichlet Kill Share) for 10,000 Monte Carlo iterations for the target match.                                        |
| **2.2** | Simulation Extraction | Aggregate the 10,000 discrete game states into a $10,000 \times 10$ matrix representing the raw, unadjusted fantasy points for all players on the server.                                                                                       |
| **2.3** | Correlation Profiling | Calculate the $10 \times 10$ Spearman Rank Correlation matrix ($\Sigma_{MC}$) from the extracted simulation matrix. This quantifies the exact negative correlation between opponents and the positive covariance between synergistic teammates. |

## Phase 3: The Copula Fusion Engine (The Mathematical Combiner)

This phase executes the mathematical fusion, anchoring the DAG's variance to the Regressor's mean. The engineering team will deploy the Iman-Conover algorithm to ensure maximum statistical rigor.

| Task ID | Description            | Technical Specifications                                                                                                                                                                            |
| ------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **3.1** | Marginal Generation    | For each player, define a continuous probability distribution (e.g., Gamma) parameterized such that its mean exactly equals $\mu_{TD}$ from Phase 1.                                                |
| **3.2** | Independent Sampling   | Draw $N = 10{,}000$ independent, uncorrelated samples from the 10 player marginal distributions, generating matrix $M$.                                                                             |
| **3.3** | Cholesky Decomposition | Compute the Van der Waerden normal scores for $M$, execute Cholesky un-correlation to yield orthogonal matrix $Z$, and multiply by the Cholesky root of the DAG correlation matrix ($\Sigma_{MC}$). |
| **3.4** | Rank Reordering        | Reorder the original independent samples in $M$ so their rank order matches the correlated matrix $Y$. This finalizes the Iman-Conover transformation.                                              |
| **3.5** | Metric Extraction      | From the combined $10,000 \times 10$ matrix, calculate the final Expected Value (mean), Floor (15th percentile), and Ceiling (85th percentile) for all players.                                     |

## Phase 4: Knapsack Solver Integration and Optimization

The final phase feeds the fused probability distributions into the Mixed-Integer Linear Programming (MILP) solver to generate the optimal VFL lineups subject to the strict salary and role constraints. Because the solver now has access to the fused 10,000-iteration distributions, it can optimize for GPP upside rather than just median outcomes.

| Task ID | Description                     | Technical Specifications                                                                                                                                                                                                                       |
| ------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **4.1** | Array Ingestion                 | Feed the $10,000 \times N$ matrix (representing all players across the entire DFS slate) into the optimization environment (e.g., Python's PuLP or cvxpy).                                                                                     |
| **4.2** | Salary Constraint Logic         | Let binary decision variable $x_i \in {0,1}$ represent drafting player $i$. Enforce the 50 VP salary cap: $\sum (c_i \cdot x_i) \leq 50.0$.                                                                                                    |
| **4.3** | Role Constraint Logic           | Enforce the strict VFL role matrix. Ensure exactly $\sum x_{Duelist} = 1$, $\sum x_{Initiator} = 1$, $\sum x_{Controller} = 1$, $\sum x_{Sentinel} = 1$, and $\sum x_{Flex} = 2$.                                                              |
| **4.4** | Roster Cap Logic                | Enforce the maximum team limit: For any real-world VCT team $T$, ensure $\sum_{i \in T} x_i \leq 2$.                                                                                                                                           |
| **4.5** | IGL Multiplier Designation      | Create a secondary binary decision variable $y_i \in {0,1}$ for the In-Game Leader constraint, where $\sum y_i = 1$ and $y_i \leq x_i$. Apply the $2\times$ EV multiplier directly to the objective function for the selected IGL.             |
| **4.6** | Objective Function Maximization | Define the objective function: $\max \sum_{i=1}^{n} (EV_i \cdot x_i + EV_i \cdot y_i)$. For GPP optimization, replace the median $EV_i$ with the 85th percentile (Ceiling) projection extracted in Task 3.5 to maximize tournament win equity. |
| **4.7** | Portfolio Simulation            | Run the finalized optimum lineup back through the 10,000 simulated slate outcomes to calculate the lineup's exact probability of finishing in the top 1% of the field, verifying true tournament viability.                                    |
# 6. Strategic Synthesis and System Outlook

The empirical failure of the V5 Bottom-Up Micro engine was not a failure of discrete event simulation methodology, but a failure of conditional dependency. By structuring the predictive architecture as a sequential Directed Acyclic Graph, the extreme uncertainty inherent in the early nodes—specifically the Contextual Bandit Map Veto—initiated a multiplicative Error Cascade. This structural flaw inherently decoupled the downstream statistical projections from reality, resulting in an unacceptable Player Kill MAE of 6.94 that underperformed even the most basic heuristic baselines.

The architecture outlined in this report—the Hybrid Micro Engine—resolves this systemic vulnerability by fundamentally separating the mathematical prediction of magnitude from the prediction of variance. The Top-Down Direct EV Regressor, fortified by Clipped Historical Baselines, Temporal Exponential Moving Averages, and Ridge-penalized Opponent Defensive Ratings, bypasses the brittle veto sequence entirely. This provides an unshakeable, highly accurate anchor for expected player performance.

Simultaneously, the DAG simulation is relegated to its most effective and mathematically defensible purpose: serving as a generative engine for covariance matrices and game-theory limits. By fusing these two distinct models via the Iman-Conover algorithm and Gaussian Copulas, the engine achieves a mathematical synthesis that is significantly greater than the sum of its parts.

The final probability matrices fed into the MILP Knapsack solver possess the absolute point-prediction accuracy of a modern machine learning regressor, alongside the combinatorial nuance and heavy-tailed variance of a discrete Monte Carlo simulation. This hybrid blueprint ensures that the VFL optimization solver generates esports rosters that not only clear strict salary and role constraints, but possess the mathematically quantified covariance necessary to secure dominant finishes in highly competitive, large-field DFS tournaments.
