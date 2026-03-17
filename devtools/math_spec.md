# xftsim Mathematical Specification

Source of truth for mathematical invariants. Derived from the manuscript
(current_manu_draft.md), the generative model in CLAUDE.md, and the legacy
reference implementation (xftsim/legacy/stats.py, xftsim/utils.py).

Every code change MUST preserve these identities. Tests that contradict
them are wrong — fix the test, not the spec.

---

## 1. Genotype Standardization

### Per-SNP standardization (Hardy-Weinberg)

Given diploid genotype matrix G (n x m) and allele frequency vector p (m,):

```
G_std = (G - 2p) / sqrt(2p(1-p))
```

When p=0 or p=1, the denominator is set to 1.0 (no scaling, only centering).

**Invariants:**
- `E[G_std[:,j]] = 0` for each variant j
- `Var(G_std[:,j]) ≈ 1` for each variant j (exact under HWE)
- `standardized_matvec(v, af) = G_std @ v`, NOT `(G - 2p) @ v`

### Allele frequency

```
af_empirical = mean(hap[:,:,0] + hap[:,:,1], axis=0) / 2
```

- Shape: (m,)
- Range: [0, 1]
- Equivalent to `diploid_genotypes.mean(axis=0) / 2`

---

## 2. Haplotype-Vector Products (Matvec Operations)

Given haplotype array H of shape (n, m, 2) and effect vector v of shape (m,):

| Operation | Definition | Shape |
|-----------|-----------|-------|
| `matvec(v)` | `(H[:,:,0] + H[:,:,1]) @ v` | (n,) |
| `matvec_maternal(v)` | `H[:,:,0] @ v` | (n,) |
| `matvec_paternal(v)` | `H[:,:,1] @ v` | (n,) |
| `rmatvec(v)` | `G.T @ v` where G = diploid | (m,) |
| `standardized_matvec(v, af)` | `((G - 2p) / sqrt(2pq)) @ v` | (n,) |

**Identities:**
- `matvec_maternal(v) + matvec_paternal(v) = matvec(v)` (always exact)
- `rmatvec` is the transpose: `v.T @ matvec(u) = rmatvec(v).T @ u`
- For 2D v of shape (m, k): outputs have shape (n, k) or (m, k) respectively

### GraphHaplotypeOperator optimization

The graph operator avoids materializing the standardized matrix. Instead:

```
v_scaled = v / sqrt(2p(1-p))
standardized_matvec(v) = matvec(v_scaled) - 2p @ v_scaled
```

This MUST produce identical results to the dense path.

---

## 3. Effect Sizes and Heritability

### Drawing effects for target h2

```
β ~ N(0, h²/m)   =>   E[Σβ²] = h²
```

Under standardized genotypes where Var(G_std[:,j]) = 1:

```
Var(Gβ) = Σ β_j² Var(G_std[:,j]) = Σ β_j² ≈ h²
```

### Round-trip invariant (CRITICAL)

If effects are drawn via `AdditiveEffects.from_h2(h2)` and applied through
`standardized_matvec`, then at generation 0 with no VT or G×E:

```
Var(Y) = Var(G) + Var(ε) = h² + (1 - h²) = 1
h²_realized = Var(G) / Var(Y) ≈ h²_design
```

Tolerance: |h²_realized - h²_design| < 0.10 for n ≥ 1000, m ≥ 500.

**If this fails, there is a standardization or effect-size bug.**

---

## 4. Phenotypic Generative Model

From the manuscript, for K traits:

```
X := (X[1],...,X[K]) ← meiosis(X*, X**)          # Offspring haplotypes
G_k = X[k] β_k                                     # Additive genetic value
ε_k ~ N(0, σ²_ε)                                   # Noise
T_k = sqrt(θ / (2K)) × Σ_j (Y*_j + Y**_j)         # Vertical transmission
E_k = T_k + ε_k                                     # Environmental component
Y_k = G_k + E_k + sqrt(φ / (σ²_g(σ²_e - θ))) × (G_k ∘ E_k)  # With G×E
```

**Variance decomposition at generation 0 (no VT, no G×E):**
```
Var(Y_k) = Var(G_k) + Var(ε_k) = h² + (1 - h²)
```

**With VT (no G×E):**
```
Var(E_k) = θ + σ²_ε
```
where θ is the VT variance contribution per trait.

---

## 5. Haseman-Elston Regression (GRM-based)

### GRM construction

```
K = G_std @ G_std.T / m
```

where G_std is the per-SNP standardized genotype matrix.

### Estimator

```
cov_g = Y.T @ (K@Y - Y) / (tr(K²) - n)
```

For a single trait Y:
```
h²_HE = cov_g / Var(Y)
```

**Properties:**
- Works with ANY relatedness structure (unrelated, sibling, mixed)
- Works at generation 0 (does not require LD buildup)
- `tr(K²)` can be computed exactly as `||G_std.T @ G_std||_F² / m²` or stochastically
- This is NOT sibling-ICC (h² = 2r_sib). That is a different estimator.

**Reference implementation:** `xftsim/legacy/stats.py:haseman_elston()` (line ~373)

---

## 6. Assortative Mating

### Linear (unidimensional) xAM

For exchangeable cross-mate correlations r across K traits:
```
latent_correlation = r × K
```

**Constraint:** r × K ≤ 1 (otherwise mathematically impossible).

At r=0.2 with K=5: latent correlation = 1.0 (boundary/extreme case).

### Implementation

Males and females are independently sorted on a linear combination of
phenotypes plus Gaussian noise, then paired by rank. The noise variance
controls the target cross-mate correlation.

### High-dimensional xAM

For arbitrary (non-exchangeable) cross-mate correlation structures, the mating
assignment is formulated as a Quadratic Assignment Problem:

```
P* = argmin_P ||Ỹ'PY - Ω̂||²_F
```

---

## 7. GWAS Test Statistics

Under multivariate xAM, the GWAS test statistic at variant j for trait k
follows a non-central chi-squared distribution:

```
T_jk ~ χ²(1, λ_jk)
```

where the non-centrality parameter λ depends on:
- The additive genetic covariance matrix Σ_g
- The excess genetic covariance from xAM (matrix A)
- Sample size N

The manuscript derives closed-form expressions for power and type-I error
conditional on stabilized covariance matrices from simulation.

---

## 8. Vertical Transmission

```
T_k = sqrt(θ / (2K)) × Σ_j (Y*_j + Y**_j)
```

- θ controls the fraction of phenotypic variance explained by VT
- The sum is over all K parental phenotypes (both parents)
- VT creates gene-environment correlations that amplify xAM-induced biases

---

## Invariant Checklist (for adversarial review)

1. [ ] `standardized_matvec` divides by sqrt(2pq), not just centers
2. [ ] `from_h2(h2)` draws β ~ N(0, h²/m) — matched to standardized genotypes
3. [ ] h² round-trip: design h² ≈ realized h² at gen 0 (within ±0.10)
4. [ ] HE estimator uses GRM formula, not sibling-ICC
5. [ ] HE works at generation 0 (doesn't require LD from AM)
6. [ ] maternal + paternal = diploid matvec (exact identity)
7. [ ] Graph operator standardized_matvec matches dense path
8. [ ] Division-by-zero protection when p=0 or p=1
9. [ ] AF computation: mean(diploid) / 2, shape (m,), range [0,1]
10. [ ] VT formula uses sqrt(θ/(2K)) scaling
