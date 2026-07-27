# V8 Context: Differentiable Breakpoint Thresholding

## 1. Legacy System Vulnerability: The Failure of Continuous Saturation ($k=0.5$)
The legacy pipeline calculated mechanical shock using a relative delta parameter $r_c$ passed through a continuous half-saturation diminishing returns function ($k=0.5$)[cite: 2]:
$$r_c = \frac{|\text{New} - \text{Old}|}{\max(|\text{Old}|, 0.0001)}, \quad \text{Shock}_c = \frac{r_c}{r_c + 0.5}$$

While continuous saturation models diminishing returns well in standard physical or economic systems, it is statistically unsound for tactical shooter engines[cite: 1]. Tactical shooters operate on strict, discrete mathematical breakpoints, not smooth asymptotic curves[cite: 1].

### Real-World Breakpoint Example:
Consider maximum player health in Valorant (100 HP base + 50 HP heavy shields = 150 HP total)[cite: 1].
* **Case A (No Breakpoint Crossed)**: Weapon headshot damage drops from $160 \to 155$. Both values strictly exceed 150 HP, resulting in a single-shot kill[cite: 1]. Time-To-Kill (TTK) and empirical game outcomes are completely unchanged[cite: 1].
* **Case B (Breakpoint Crossed)**: Weapon headshot damage drops from $155 \to 145$. The value falls below the 150 HP threshold, shifting the requirement from 1 shot to 2 shots[cite: 1]. This binary phase-shift doubles the TTK and fundamentally destroys the weapon's viability in professional play[cite: 1].

The legacy half-saturation function smooths over this phase-shift, registering Case A and Case B as nearly identical relative shocks[cite: 1]. This blinds the downstream model to binary threshold drops[cite: 1].

## 2. Mathematical Challenge: Non-Differentiable Step Functions
To capture binary phase-shifts accurately, the system requires step-functions (such as the Heaviside step function)[cite: 1]. However, the derivative of a step function is the Dirac delta function, which is zero everywhere except at the exact threshold where it is undefined[cite: 1]. 

A gradient of zero halts backpropagation in PyTorch (`grad = 0`), making gradient-descent optimization impossible[cite: 1].

## 3. Differentiable Relaxations
To resolve non-differentiability in PyTorch, `v8_breakpoint_thresholds.py` must implement two primary state-of-the-art relaxations[cite: 1]:

### A. Straight-Through Estimator (STE)
An STE provides a custom `torch.autograd.Function` where:
* **Forward Pass**: Evaluates the strict, discrete hard threshold (e.g., $y = 1.0$ if $\text{Damage} \ge \tau$ else $0.0$)[cite: 1].
* **Backward Pass**: Ignores the non-differentiable step and passes the upstream gradient through unmodified as an identity operation[cite: 1]:
$$\text{Forward:} \quad y = \text{sign}(x - \tau), \quad \text{Backward:} \quad \frac{\partial L}{\partial x} \approx \frac{\partial L}{\partial y}$$

### B. Temperature-Controlled Soft Step Relaxations (Gumbel-Softmax / Sigmoid Surrogate)
Alternatively, a parameterized sigmoid function acts as a continuous, differentiable approximation to discrete sampling[cite: 1]:
$$y = \sigma\left(\frac{x - \tau}{\tau_{temp}}\right)$$

Where $\tau$ is the critical game breakpoint (e.g., $150.0$ HP) and $\tau_{temp}$ is the temperature parameter[cite: 1]. 
* As $\tau_{temp} \to 0$, the function sharpens into a hard discrete step function[cite: 1].
* During training, gradient descent can optimize the threshold location and sensitivity without breaking autograd[cite: 1].