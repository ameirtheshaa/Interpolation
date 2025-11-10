# Mathematical Theory

## Wind Velocity Field Interpolation via Tucker Decomposition and PCHIP

---

## 1. Problem Formulation

### 1.1 Physical System

Consider the steady-state, turbulent flow of air around a rigid structure (e.g., building, cylinder) subjected to varying wind directions. The flow is governed by the Reynolds-Averaged Navier-Stokes (RANS) equations:

**Continuity Equation:**
$$\nabla \cdot \mathbf{u} = 0$$

**Momentum Equation:**
$$\rho (\mathbf{u} \cdot \nabla) \mathbf{u} = -\nabla p + \nabla \cdot (\mu + \mu_t)(\nabla \mathbf{u} + \nabla \mathbf{u}^T)$$

where:
- $\mathbf{u} = (u_x, u_y, u_z)$: velocity vector field
- $p$: pressure field
- $\rho$: air density
- $\mu$: dynamic viscosity
- $\mu_t$: turbulent (eddy) viscosity
- $\nu_t = \mu_t/\rho$: kinematic turbulent viscosity

### 1.2 Computational Challenge

**Given:** CFD simulation results at discrete wind angles $\{\theta_1, \theta_2, \ldots, \theta_M\}$

**Goal:** Efficiently predict flow fields at arbitrary wind angles $\theta \in [0°, 180°]$ without re-running expensive CFD simulations

**Data Structure:** For each wind angle $\theta_m$, we have:
- Spatial points: $\mathbf{x}_i \in \mathbb{R}^3$, $i = 1, \ldots, N$
- Flow variables: $\mathbf{q}_i = [p, u_x, u_y, u_z, \nu_t]^T \in \mathbb{R}^K$, $K=5$

---

## 2. Theoretical Framework

### 2.1 Tensor Representation

We organize the CFD data into a 3rd-order tensor:

$$\mathcal{T} \in \mathbb{R}^{N \times M \times K}$$

where:
- **Mode 1 (Spatial):** $N$ spatial points
- **Mode 2 (Angular):** $M$ wind angles
- **Mode 3 (Variables):** $K$ flow variables

Each element is:
$$\mathcal{T}_{imk} = q_k(\mathbf{x}_i, \theta_m)$$

### 2.2 Tucker Decomposition

The Tucker decomposition factorizes the tensor into:

$$\mathcal{T} \approx \mathcal{G} \times_1 \mathbf{U} \times_2 \mathbf{V} \times_3 \mathbf{W}$$

or equivalently:

$$\mathcal{T}_{imk} \approx \sum_{r_1=1}^{R_1} \sum_{r_2=1}^{R_2} \sum_{r_3=1}^{R_3} \mathcal{G}_{r_1 r_2 r_3} \, U_{i, r_1} \, V_{m, r_2} \, W_{k, r_3}$$

where:
- $\mathcal{G} \in \mathbb{R}^{R_1 \times R_2 \times R_3}$: **core tensor** (compressed representation)
- $\mathbf{U} \in \mathbb{R}^{N \times R_1}$: **spatial modes** (left singular vectors)
- $\mathbf{V} \in \mathbb{R}^{M \times R_2}$: **angular modes**
- $\mathbf{W} \in \mathbb{R}^{K \times R_3}$: **variable modes**
- $(R_1, R_2, R_3)$: **Tucker ranks** (typically $R_1 \ll N$, $R_2 \ll M$, $R_3 \ll K$)

### 2.3 Approximation Theory

**Theorem (Best Low-Rank Approximation):**
Given ranks $(R_1, R_2, R_3)$, the Tucker decomposition minimizes the Frobenius norm:

$$\min_{\mathcal{G}, \mathbf{U}, \mathbf{V}, \mathbf{W}} \|\mathcal{T} - \mathcal{G} \times_1 \mathbf{U} \times_2 \mathbf{V} \times_3 \mathbf{W}\|_F$$

subject to orthonormality: $\mathbf{U}^T\mathbf{U} = \mathbf{I}_{R_1}$, $\mathbf{V}^T\mathbf{V} = \mathbf{I}_{R_2}$, $\mathbf{W}^T\mathbf{W} = \mathbf{I}_{R_3}$

**Solution:** Higher-Order Singular Value Decomposition (HOSVD)
1. Unfold tensor along each mode
2. Compute truncated SVD for each unfolding
3. Construct core tensor via multi-mode multiplication

---

## 3. Methodology

### 3.1 Data Standardization

To improve numerical stability and decomposition quality:

$$\tilde{\mathcal{T}}_{imk} = \frac{\mathcal{T}_{imk} - \mu_k}{\sigma_k}$$

where $\mu_k$ and $\sigma_k$ are the mean and standard deviation computed over spatial and angular dimensions:

$$\mu_k = \frac{1}{NM} \sum_{i=1}^N \sum_{m=1}^M \mathcal{T}_{imk}$$

$$\sigma_k = \sqrt{\frac{1}{NM} \sum_{i=1}^N \sum_{m=1}^M (\mathcal{T}_{imk} - \mu_k)^2}$$

### 3.2 Tucker Decomposition Algorithm

**Input:** Standardized tensor $\tilde{\mathcal{T}} \in \mathbb{R}^{N \times M \times K}$, ranks $(R_1, R_2, R_3)$

**Output:** Core tensor $\mathcal{G}$, factor matrices $\mathbf{U}, \mathbf{V}, \mathbf{W}$

**Algorithm (HOSVD):**

1. **Mode-1 unfolding and SVD:**
   - Unfold: $\mathbf{T}_{(1)} \in \mathbb{R}^{N \times MK}$
   - SVD: $\mathbf{T}_{(1)} = \tilde{\mathbf{U}}\mathbf{\Sigma}\mathbf{Z}^T$
   - Truncate: $\mathbf{U} = \tilde{\mathbf{U}}[:, 1:R_1]$

2. **Mode-2 unfolding and SVD:**
   - Unfold: $\mathbf{T}_{(2)} \in \mathbb{R}^{M \times NK}$
   - SVD: $\mathbf{T}_{(2)} = \tilde{\mathbf{V}}\mathbf{\Sigma}\mathbf{Z}^T$
   - Truncate: $\mathbf{V} = \tilde{\mathbf{V}}[:, 1:R_2]$

3. **Mode-3 unfolding and SVD:**
   - Unfold: $\mathbf{T}_{(3)} \in \mathbb{R}^{K \times NM}$
   - SVD: $\mathbf{T}_{(3)} = \tilde{\mathbf{W}}\mathbf{\Sigma}\mathbf{Z}^T$
   - Truncate: $\mathbf{W} = \tilde{\mathbf{W}}[:, 1:R_3]$

4. **Compute core tensor:**
   $$\mathcal{G} = \tilde{\mathcal{T}} \times_1 \mathbf{U}^T \times_2 \mathbf{V}^T \times_3 \mathbf{W}^T$$

**Complexity:** $O(N M K \min(N, M, K))$ for SVDs

### 3.3 Angular Interpolation via PCHIP

After decomposition, we have a reduced representation:

$$\mathbf{A}(\theta_m) = \mathcal{G} \times_2 \mathbf{V}[m, :] \times_3 \mathbf{W}$$

where $\mathbf{A}(\theta_m) \in \mathbb{R}^{R_1 \times K}$ is the "alpha matrix" for angle $\theta_m$.

**Key Insight:** $\mathbf{A}(\theta)$ varies smoothly with wind angle $\theta$.

**PCHIP Interpolation:** For each entry $A_{r_1, k}(\theta)$, fit a piecewise cubic Hermite interpolator:

$$A_{r_1, k}(\theta) = \text{PCHIP}\{(\theta_1, A_{r_1,k}(\theta_1)), \ldots, (\theta_M, A_{r_1,k}(\theta_M))\}$$

**Properties of PCHIP:**
- $C^1$ continuous (smooth first derivative)
- Shape-preserving (no spurious oscillations)
- Local support (each piece depends on 2-4 neighboring points)

### 3.4 Prediction at New Angles

Given a new wind angle $\theta^*$:

1. **Interpolate alpha matrix:**
   $$\mathbf{A}(\theta^*) = [\text{PCHIP}_{r_1,k}(\theta^*)]_{R_1 \times K}$$

2. **Reconstruct spatial field:**
   $$\tilde{\mathbf{Q}}(\theta^*) = \mathbf{U} \cdot \mathbf{A}(\theta^*)$$
   where $\tilde{\mathbf{Q}} \in \mathbb{R}^{N \times K}$

3. **Inverse standardization:**
   $$\mathbf{Q}(\theta^*) = \tilde{\mathbf{Q}}(\theta^*) \cdot \text{diag}(\sigma) + \mu$$

---

## 4. Algorithmic Details

### 4.1 Complete Workflow

**Algorithm: Wind Field Interpolation**

```
Input:
  - CFD data files for angles {θ₁, ..., θₘ}
  - Tucker ranks (R₁, R₂, R₃)
  - Query angles {θ*₁, ..., θ*ₙ}

Output:
  - Predicted flow fields Q(θ*ᵢ) for i=1,...,n

1. DATA LOADING:
   For each angle θₘ:
     Load CSV file → DataFrame
     Extract spatial coordinates X ∈ ℝᴺˣ³
     Extract flow variables Q ∈ ℝᴺˣᴷ
     Store in tensor T[: , m, :]

2. STANDARDIZATION:
   Compute μ, σ across spatial and angular dimensions
   T_std = (T - μ) / σ

3. TUCKER DECOMPOSITION:
   [G, U, V, W] = HOSVD(T_std, (R₁, R₂, R₃))

4. COMPUTE ALPHA MATRICES:
   For m = 1 to M:
     A(θₘ) = G ×₂ V[m,:] ×₃ W  # Shape: R₁ × K

5. FIT PCHIP INTERPOLATORS:
   For r₁ = 1 to R₁:
     For k = 1 to K:
       coeffs[r₁,k] = FitPCHIP({θ₁,...,θₘ}, {A(θ₁)[r₁,k],...,A(θₘ)[r₁,k]})

6. PREDICT AT NEW ANGLES:
   For each query angle θ*:
     # Interpolate alpha matrix
     For r₁ = 1 to R₁:
       For k = 1 to K:
         A(θ*)[r₁,k] = coeffs[r₁,k](θ*)

     # Reconstruct spatial field
     Q_std(θ*) = U @ A(θ*)  # Matrix multiplication

     # Inverse standardization
     Q(θ*) = Q_std(θ*) * σ + μ

   Return Q(θ*)
```

### 4.2 Computational Complexity

**Preprocessing (one-time cost):**
- Data loading: $O(NMK)$
- Tucker decomposition: $O(NMK \cdot \min(N,M,K))$
- PCHIP fitting: $O(M \cdot R_1 \cdot K)$

**Total preprocessing:** $O(NMK \cdot \min(N,M,K))$

**Inference (per query angle):**
- Alpha matrix interpolation: $O(R_1 \cdot K \cdot \log M)$ (PCHIP evaluation)
- Spatial reconstruction: $O(N \cdot R_1 \cdot K)$

**Total inference:** $O(N \cdot R_1 \cdot K)$

**Speedup vs. CFD:**
- CFD simulation: $O(10^3 - 10^6)$ seconds
- Our method: $O(10^{-2} - 10^{0})$ seconds
- Speedup: $10^3 - 10^6 \times$

### 4.3 Memory Requirements

**Storage:**
- Original data: $NMK$ floats
- Compressed representation: $(R_1 R_2 R_3) + (NR_1) + (MR_2) + (KR_3)$ floats
- Compression ratio: $\frac{NMK}{R_1R_2R_3 + NR_1 + MR_2 + KR_3}$

**Example:**
- $N = 10^6$, $M = 13$, $K = 5$
- Original: $6.5 \times 10^7$ floats ≈ 260 MB
- Compressed with $(R_1, R_2, R_3) = (65, 13, 5)$:
  - Core: $65 \times 13 \times 5 = 4,225$
  - Factors: $10^6 \times 65 + 13 \times 13 + 5 \times 5 = 65,000,194$
  - Total: ≈ 260 MB
  - *Note: Compression is modest due to large $R_1$ relative to $M$*

**Better compression with larger $M$:**
- If $M = 100$ angles: Compression ratio ≈ 4-5×

---

## 5. Convergence Properties

### 5.1 Reconstruction Error

The Tucker approximation error is bounded by:

$$\|\mathcal{T} - \hat{\mathcal{T}}\|_F^2 = \sum_{r_1 > R_1} \sigma_{r_1}^2 + \sum_{r_2 > R_2} \sigma_{r_2}^2 + \sum_{r_3 > R_3} \sigma_{r_3}^2$$

where $\sigma_{r_i}$ are the singular values of mode-$i$ unfolding.

**Rank Selection Strategy:**
- Choose $(R_1, R_2, R_3)$ to capture $\geq 99\%$ of variance in each mode
- Trade-off: larger ranks → better accuracy, more storage

### 5.2 Interpolation Error

PCHIP interpolation error for a smooth function $f(\theta)$:

$$|f(\theta) - \text{PCHIP}_f(\theta)| \leq C \cdot h^4 \cdot \max_{\xi \in [\theta_{m}, \theta_{m+1}]} |f^{(4)}(\xi)|$$

where $h = \max_m |\theta_{m+1} - \theta_m|$ is the maximum angle spacing.

**Implications:**
- Uniform angle spacing improves interpolation accuracy
- Smooth variation of $\mathbf{A}(\theta)$ is critical (validated empirically)
- For $h = 15°$: error ~ $O(15^4) = O(50,000)$ in normalized units (typically small)

---

## 6. Theoretical Guarantees

### 6.1 Approximation Theorem

**Theorem:** Assume the flow field $\mathbf{q}(\mathbf{x}, \theta)$ has smooth dependence on $\theta$ with bounded derivatives. Then for sufficiently large Tucker ranks and sufficiently dense angle sampling:

$$\mathbb{E}_{\theta \sim \mathcal{U}[0°, 180°]} \|\mathbf{q}(\cdot, \theta) - \hat{\mathbf{q}}(\cdot, \theta)\|_2 \leq \epsilon$$

where $\epsilon$ depends on:
- Tucker ranks $(R_1, R_2, R_3)$
- Angle sampling density $1/h$
- Smoothness of $\mathbf{q}$ w.r.t. $\theta$

**Proof Sketch:**
1. Tucker approximation error bounded by tail singular values
2. PCHIP interpolation error bounded by $O(h^4)$ for $C^4$ functions
3. Combine errors via triangle inequality

### 6.2 Stability Analysis

The condition number of the spatial reconstruction is:

$$\kappa(\mathbf{U}) = \frac{\sigma_1(\mathbf{U})}{\sigma_{R_1}(\mathbf{U})}$$

For orthonormal $\mathbf{U}$: $\kappa(\mathbf{U}) = 1$ (perfect stability)

---

## 7. Extensions and Future Directions

### 7.1 Time-Varying Flows

Extend to 4th-order tensor:
$$\mathcal{T} \in \mathbb{R}^{N \times M \times K \times T}$$
where $T$ is the number of time steps.

Tucker decomposition:
$$\mathcal{T} \approx \mathcal{G} \times_1 \mathbf{U} \times_2 \mathbf{V} \times_3 \mathbf{W} \times_4 \mathbf{Z}$$

### 7.2 Uncertainty Quantification

Incorporate Bayesian inference:
- Prior on Tucker ranks
- Posterior distribution over $\mathcal{G}, \mathbf{U}, \mathbf{V}, \mathbf{W}$
- Predictive uncertainty at new angles

### 7.3 Neural Network Interpolation

Replace PCHIP with neural network:
$$\mathbf{A}(\theta) = \text{NN}_{\phi}(\theta)$$

**Advantages:**
- Learn nonlinear angle dependencies
- Incorporate physics constraints (e.g., periodicity)
- Uncertainty quantification via Bayesian NNs

### 7.4 Adaptive Rank Selection

Automatic rank selection via:
- Cross-validation on held-out angles
- Information criteria (AIC, BIC)
- Bayesian model selection

---

## References

1. Kolda, T. G., & Bader, B. W. (2009). *Tensor decompositions and applications*. SIAM Review, 51(3), 455-500.

2. Tucker, L. R. (1966). *Some mathematical notes on three-mode factor analysis*. Psychometrika, 31(3), 279-311.

3. De Lathauwer, L., De Moor, B., & Vandewalle, J. (2000). *A multilinear singular value decomposition*. SIAM Journal on Matrix Analysis and Applications, 21(4), 1253-1278.

4. Fritsch, F. N., & Carlson, R. E. (1980). *Monotone piecewise cubic interpolation*. SIAM Journal on Numerical Analysis, 17(2), 238-246.

5. Benner, P., Gugercin, S., & Willcox, K. (2015). *A survey of projection-based model reduction methods for parametric dynamical systems*. SIAM Review, 57(4), 483-531.
