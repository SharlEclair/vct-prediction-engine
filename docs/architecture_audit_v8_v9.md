# VCT / VLF DFS Prediction Engine — Architecture & Codebase Truth Audit (v8 & v9)

**Audit Date:** August 2026  
**Target Repositories/Engines:** Valorant Fantasy League (VLF) DFS Engine, v8 Differentiable Patch Engine, v9 MILP Knapsack Solver  
**Audit Scope:** Deep code-level extraction, mathematical verification, data contracts, and assumption reconciliation against live codebase files.

---

## Executive Summary of Assumption Corrections (19 Total)

| # | Prompt Assumption | Reality in Live Codebase | Source File |
|---|---|---|---|
| 1 | Field named `ability_name` in Patch schema | Actual field name is `ability` | [`v8_patch_parser.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/v8_patch_parser.py) |
| 2 | Formal Pydantic schema for historical telemetry row | Loose dictionary loaded from processed JSON files; no class | [`v9_historical_stats.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/v9_historical_stats.py) |
| 3 | Scoring includes Deaths, Assists, First Bloods, Clutches | Unscored in VLF ruleset; scoring is bracketed kill & map bonuses | [`VLF Rules International.txt`](file:///c:/Users/91704/Desktop/vct-prediction-model/VLF%20Rules%20International.txt) |
| 4 | Platform name is "VFL" | Platform name is "VLF" (Valorant Fantasy League) | [`VLF Rules International.txt`](file:///c:/Users/91704/Desktop/vct-prediction-model/VLF%20Rules%20International.txt) |
| 5 | PyTorch Gumbel-Softmax used for STE | Hand-written `torch.autograd.Function` with identity backward pass | [`v8_breakpoint_thresholds.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/v8_breakpoint_thresholds.py) |
| 6 | External temperature schedule for STE | Temperature (`tau_temp`) is a trainable `nn.Parameter`, not scheduled | [`v8_breakpoint_thresholds.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/v8_breakpoint_thresholds.py) |
| 7 | Graph neural network maps cross-agent synergy | String-based agent dictionary grouping; no graph structures | [`v8_copula_aggregation.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/v8_copula_aggregation.py) |
| 8 | LSTM Seq2Seq with encoder-decoder | Unidirectional single-LSTM with linear classification head | [`v8_dros_optimizer.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/v8_dros_optimizer.py) |
| 9 | DRos reward is fantasy points | Reward is round win probability $q_{\text{hat}}(x,a) \in (0, 1)$ | [`v8_dros_optimizer.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/v8_dros_optimizer.py) |
| 10 | Optimizer uses Gurobi or PuLP | Exclusively uses `scipy.optimize.milp` with `LinearConstraint` | [`v9_milp_optimizer.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/v9_milp_optimizer.py) |
| 11 | Sortino ratio is the global objective | Global objective is raw EV; Sortino is optional for IGL y-variables | [`v9_milp_optimizer.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/v9_milp_optimizer.py) |
| 12 | "Map Cap Rule" is a MILP matrix constraint | Map EV is calculated upstream; no Map Cap constraint in matrix | [`v9_milp_optimizer.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/v9_milp_optimizer.py) |
| 13 | Captain has 1.5× / 2.0× selectable tiers | Sole role is IGL with flat 2.0× multiplier | [`v9_milp_optimizer.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/v9_milp_optimizer.py) |
| 14 | Team Elo ratings are calculated dynamically | Hardcoded dictionary or passed as static parameters | [`v9_fantasy_engine.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/v9_fantasy_engine.py) |
| 15 | Post-gameweek calibration uses a Kalman filter | First-order momentum update ($\mu_{t+1} = \mu_t + \alpha \epsilon_t$) | [`v9_h2h_and_calibration.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/v9_h2h_and_calibration.py) |
| 16 | Pipeline orchestrated via DAG (Airflow/Prefect) | Linear Python subprocess chain via `run_pipeline.py` | [`run_pipeline.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/run_pipeline.py) |
| 17 | Compute time enforced with strict DFS roster locks | Unenforced; total runtime is passively logged after completion | [`run_pipeline.py`](file:///c:/Users/91704/Desktop/vct-prediction-model/run_pipeline.py) |
| 18 | Codebase models "Agent Mastery Inertia" | Absent from codebase (no parameters, functions, or weights) | Repository-wide |
| 19 | Codebase models "Indirect Network Effects" | Absent from codebase (no cross-agent counter-buff logic) | Repository-wide |

---

## Detailed Technical Verification

### 1. Data Schemas & Ingestion Layer

#### Pydantic v2 Schema (`v8_patch_parser.py`)
```python
class PatchChangeItem(BaseModel):
    agent: str
    ability: str
    stat_modified: str
    old_value: Optional[Union[float, int, str]] = None
    new_value: Optional[Union[float, int, str]] = None
    is_mechanical_removal: bool
    raw_evidence: Optional[str] = None

class PatchExtractionPayload(BaseModel):
    version: str
    date: Optional[str] = None
    changes: List[PatchChangeItem] = Field(default_factory=list)
    raw_wikitext_hash: Optional[str] = None
```

#### Official VLF Ruleset Breakdown (`VLF Rules International.txt`)
- **Roster Structure:** 1 Initiator, 1 Duelist, 1 Controller, 1 Sentinel, 2 Wildcards (Total = 6 starters per weekly matchup; in v9 full squad = 11 players).
- **Gameweek Aggregation:** Players score only their **highest 2 map scores** per Gameweek.
- **Kill Points:** 
  - 0 kills = $-3$ pts
  - 1–4 kills = $-1$ pt
  - 10 kills = $+1$ pt ($+1$ for every additional 5 kills: 15k = $+2$, 20k = $+3$, etc.)
- **Multi-kill Bonuses:** 4K = $+1$, 5K+ = $+3$, 6K = $+5$, 7K = $+10$.
- **Map & Series Outcomes:**
  - Map win = $+1$, Win by 5–9 rounds = $+1$, Win by 10+ rounds = $+2$.
  - Loss by 10+ rounds = $-1$.
  - 13–0 sweep win = $+5$, 0–13 loss = $-5$.
  - Best of 3 (2–0 sweep) bonus = $+2$.
  - Series wins: 2–0 = $+2$, 3–0 = $+4$, 3–1 = $+1$.
- **VLR Rating Bonuses:** Match high = $+3$, 2nd high = $+2$, 3rd high = $+1$; Rating $\ge 1.5$ = $+1$, $\ge 1.75$ = $+2$, $\ge 2.0$ = $+3$.

---

### 2. v8 Differentiable Patch Engine

#### Attention Gating & Embeddings (`v8_differentiable_base.py`)
- Category Embeddings: $d_{\text{embed}} = 16$, Category count = 5 (`combat`, `ability`, `movement`, `economy`, `general`).
- Ability Type Embeddings: $d_{\text{embed}} = 16$, Ability tier count = 5 (`signature`, `ultimate`, `basic`, `passive`, `general`).
- Context Input Vector:
  $$X_{\text{raw}} = \left[ e_{\text{cat}} \,\|\, e_{\text{ab}} \,\|\, \Delta_{\text{norm}} \,\|\, \mathbb{I}_{\text{mech}} \,\|\, \mathbb{I}_{\text{numeric}} \right] \in \mathbb{R}^{35}$$
- Context Projection: $\text{Linear}(35 \to 32) \to \text{SiLU} \to \text{Linear}(32 \to 32)$.
- Dynamic Gate: $\beta_{\text{dynamic}} = \sigma(W_{\text{attn}} X_{\text{context}} + b) \in (0, 1)$.

#### Breakpoint Straight-Through Estimator (`v8_breakpoint_thresholds.py`)
```python
class StraightThroughStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: float = 150.0) -> torch.Tensor:
        ctx.save_for_backward(x, torch.tensor(threshold, dtype=x.dtype, device=x.device))
        return (x >= threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.clone(), None
```
- Soft surrogate approximation: $y = \sigma((x - \theta) / \tau_{\text{temp}})$, where $\tau_{\text{temp}}$ is an unconstrained trainable parameter clamped at $\min=10^{-3}$.

#### Archimedean Gumbel Copula Synergy (`v8_copula_aggregation.py`)
- Parameter constraints: $\theta = 1.0 + \text{Softplus}(\text{raw\_theta}) \ge 1.0$, $\alpha = 1 / \theta \in (0, 1]$.
- Upper-Tail Dependence: $\lambda_U = 2 - 2^{1/\theta}$.
- Generator function: $\psi(u) = (-\ln u)^\alpha$, where $u_i = 1 - S_i$.
- Aggregated Concept Drift: $\text{Drift} = 1 - \exp\left( -\left( \sum_{i=1}^d (-\ln u_i)^\alpha \right)^{1/\alpha} \right)$.

#### DRos Off-Policy Evaluation (`v8_dros_optimizer.py`)
- Direct Method Reward $q_{\text{hat}}(x, a)$: Unidirectional 1-layer LSTM ($d_{\text{in}}=16, d_{\text{hidden}}=32$) predicting 4-action round win probabilities.
- Optimistic Shrinkage Weight ($\lambda = 5.0$):
  $$w_\lambda(x, a) = \frac{\lambda \cdot w(x, a)}{w(x, a)^2 + \lambda}, \quad \text{where } w(x, a) = \frac{\pi_e(a \mid x)}{\pi_0(a \mid x)}$$
- DRos Objective:
  $$V_{\text{DRos}}(\pi_e; \lambda) = \frac{1}{n} \sum_{i=1}^n \left[ \sum_a \pi_e(a \mid x_i) q_{\text{hat}}(x_i, a) + w_\lambda(x_i, a_i) \left( r_i - q_{\text{hat}}(x_i, a_i) \right) \right]$$

---

### 3. v9 DFS MILP Optimizer (`v9_milp_optimizer.py`)

- **Solver Engine:** `scipy.optimize.milp`.
- **Decision Vector:** Expanded $2N$ binary vector $x = [x_1, \dots, x_N, y_1, \dots, y_N]^T$.
- **Cost Vector:** $c = -[\text{EV}_1, \dots, \text{EV}_N, \text{EV}_{\text{IGL}, 1}, \dots, \text{EV}_{\text{IGL}, N}]^T$.
- **Constraints Matrix ($A \cdot x \le b$):**
  1. Exact Roster Size: $\sum_{i=1}^N x_i = 11$.
  2. IGL Singularity: $\sum_{i=1}^N y_i = 1$.
  3. IGL Selection Inclusion: $y_i - x_i \le 0, \quad \forall i \in \{1, \dots, N\}$.
  4. Salary Budget Cap: $\sum_{i=1}^N P_i x_i \le 100.0\text{ VP}$.
  5. Canonical Role Bounds: $2 \le \sum_{i \in \text{Role}_r} x_i \le 5$ for each of the 4 canonical roles.
  6. Maximum VCT Team Limit: $\sum_{i \in \text{Team}_t} x_i \le 2$ for all teams.

---

### 4. Bayesian Decay, CVaR & Calibration (`v9_historical_stats.py`, `v9_h2h_and_calibration.py`)

- **Kish's Effective Sample Size:**
  $$n_{\text{eff}} = \frac{\left(\sum w_i\right)^2}{\sum w_i^2}$$
- **Gaussian CVaR Quantiles:**
  $$\text{CVaR}_{10} = \mu - 1.75498 \cdot \sigma, \quad \text{CVaR}_{90} = \mu + 1.75498 \cdot \sigma$$
- **Telemetry Modifiers:**
  $$\text{EV}_{\text{mod}} = \mu_{\text{post}} \cdot (1 - 0.5 \cdot Z_{\text{FD}})$$
  $$\text{CVaR}_{10, \text{mod}} = \text{CVaR}_{10} + 1.0 \cdot Z_{\text{KAST}}, \quad \text{CVaR}_{90, \text{mod}} = \text{CVaR}_{90} + 1.0 \cdot Z_{\text{ADR}}$$
- **H2H Team Elo Proxy Multiplier ($\gamma = 0.15$):**
  $$M_{\text{proxy}} = 1.0 + 0.15 \cdot \tanh\left( \frac{R_{\text{teamA}} - R_{\text{teamB}}}{400} \right) \in [0.85, 1.15]$$
- **Momentum Calibration Loop ($\alpha = 0.20$):**
  $$\mu_{\text{prior}, t+1} = \mu_{\text{prior}, t} + 0.20 \cdot \left( \text{ActualPoints}_t - \text{PredictedEV}_t \right)$$
