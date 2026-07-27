# V8 Context: Differentiable Base & Attention Gating

## 1. Legacy System Flaws
The legacy `patch_analyzer.py` utilizes a static expert system[cite: 2]. It applies static elasticities, utilizing hardcoded values such as `combat=1.2`, `ability=1.0`, `movement=1.0`, `economy=0.8`, and `general=0.5`[cite: 1, 2]. It also classifies abilities with fixed scalars like Signature/Dash = 0.40, Ultimate = 0.30, and Basic = 0.15[cite: 1, 2]. 

This static formulation is context-blind and structurally over-penalizes specific agent archetypes[cite: 1]. The elasticity of a category in a tactical shooter is inherently non-stationary; the strategic value of a parameter fluctuates depending on the overarching metagame, map geometry, and team composition[cite: 1]. 

## 2. Differentiable Architectural Blueprint
To resolve these mathematical bottlenecks, the system must transition into a Differentiable Computational Graph[cite: 1]. By reformulating the patch analyzer as a neural architecture built in PyTorch, the parameters $\beta_{cat}$ and $w_{ab}$ can be jointly learned end-to-end[cite: 1].

The static script must be translated into a sequence of differentiable tensor operations[cite: 1]. The structured JSON extracted via the NLP layer (Phase 1) will be projected into feature vectors $X \in \mathbb{R}^{N \times D}$, where $N$ is the number of agents and $D$ is the dimensionality of the extracted numerical deltas[cite: 1].

## 3. Learned Category Embeddings and Attention Gating
Instead of static scalars, $\beta_{cat}$ and $w_{ab}$ must become trainable parameter matrices in a neural network layer[cite: 1]. 

The elasticity must be modeled as a context-aware gating mechanism[cite: 1]:
$$\beta_{dynamic} = \sigma(W_{attn} \cdot X_{context} + b)$$

Where $W_{attn}$ is a learned weight matrix that projects the specific textual or categorical context of the patch into an attention score[cite: 1]. This allows the neural network to dynamically decide if a combat nerf matters heavily given the current global state of the game, rather than blindly applying a permanent static scalar[cite: 1].