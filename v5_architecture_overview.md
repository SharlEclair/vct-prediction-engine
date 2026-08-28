# 1. The Paradigm Shift: From Top-Down Macro to Bottom-Up Micro Simulation

The transition from a **"Top-Down Macro" classifier** to a **"Bottom-Up Micro" Simulation Engine** represents a critical evolution in the architectural maturity of tactical esports predictive modeling.

Historically, predictive pipelines in the Valorant Champions Tour (VCT) ecosystem relied on gradient boosting ensembles to forecast match outcomes directly from aggregated features.

The V4.3 architecture successfully mitigated concept drift through a two-dimensional **Composite Decay Matrix**, which dynamically weighted historical matches by penalizing them across:

- Chronological days elapsed ($\Delta t$)
- Patch distance ($\Delta P$)

While mathematically robust for traditional match-winner classification, this architecture is fundamentally incompatible with the constraints of a Fantasy Esports optimization solver.

Fantasy scoring systems, such as the VFL ruleset, are strictly discretized and highly non-linear.

A macro model that outputs:

$$
P(Win)=68\%
$$

provides zero mathematical utility to a linear programming solver tasked with maximizing expected fantasy points across a constrained salary cap.

To satisfy these rigorous demands, the V5 predictive architecture deconstructs the match into its sequential, causal components.

The outcome is formulated as the terminal node of a Directed Acyclic Graph (DAG) of conditional probabilities:

$$
\text{Map Veto}
\rightarrow
\text{Agent Composition}
\rightarrow
\text{Round Score}
\rightarrow
\text{Player Micro-Stats}
$$