# Per-Commit Adversarial Review Workflow

## Purpose

Catch bugs where the code does something mathematically wrong but tests
pass because they validate the implementation rather than the specification.

The canonical example: `standardized_matvec` only centered (G - 2p) instead
of standardizing (G - 2p)/sqrt(2pq). Tests passed because they computed the
"expected" value using the same buggy formula.

## How It Works

On each commit (or PR), a fresh Claude agent reviews the diff against the
mathematical specification in `devtools/math_spec.md`. The agent has NOT
seen the code before and approaches it adversarially — looking for ways
the implementation might violate the spec, not confirming it works.

## Running an Adversarial Review

### Manual (recommended for significant changes)

Ask Claude Code:

```
Review the last commit against devtools/math_spec.md. Act as an adversarial
reviewer: assume the code is wrong and try to find where it violates the
mathematical specification. Check both the implementation AND the tests.
For each test, ask: "Would this test still pass if the math were wrong?"
```

### On a specific file

```
Adversarially review xftsim/struct.py against devtools/math_spec.md,
focusing on standardized_matvec and to_diploid_standardized.
```

### On a PR

```
Review PR #N against devtools/math_spec.md. For each changed file,
check whether the change preserves all mathematical invariants.
```

## Review Checklist

The reviewer agent MUST check each of these. A review is incomplete
if any item is skipped.

### A. Standardization correctness

1. Does `standardized_matvec` produce `((G - 2p) / sqrt(2pq)) @ v`?
2. Does `to_diploid_standardized(scale=True)` divide by sqrt(2pq)?
3. Is `scale=True` passed wherever standardized genotypes are needed?
4. Are the GraphHaplotypeOperator and DenseHaplotypeArray paths consistent?

### B. Effect size / heritability chain

5. Does `from_h2(h2, m)` draw β ~ N(0, h²/m)?
6. Are effects applied through `standardized_matvec` (not raw `matvec`)?
7. Is the round-trip h² preserved? (design ≈ realized at gen 0)

### C. Estimator correctness

8. Does HE use `cov_g = Y'(KY - Y) / (tr(K²) - n)` with K = GG'/m?
9. Does HE work at generation 0 (no reliance on LD from AM)?
10. Are tr(K²) computations numerically stable?

### D. Test quality (CRITICAL)

For EVERY numerical test in the diff, ask:

11. Does the test compute its expected value from the SPEC, or from the
    CODE being tested? (If from the code, it's a tautology.)
12. Would this test catch a centering-only bug (standardize without scaling)?
13. Would this test catch an off-by-sqrt(m) error?
14. Are tolerances tight enough to distinguish correct from buggy?
    (h²=0.5 ± 0.3 is useless; h²=0.5 ± 0.10 is meaningful)
15. Does the test use enough samples for statistical power? (n < 100 is
    usually too small for h² estimation tests)

### E. Identity preservation

16. `matvec_maternal(v) + matvec_paternal(v) == matvec(v)` (exact)
17. Division-by-zero protection when p=0 or p=1

## Red Flags

These patterns indicate likely bugs:

- **Test mirrors implementation**: `expected = code_under_test.some_method()`
  then `assert result == expected`. This tests nothing.
- **Wide tolerance on tight quantity**: h² estimation with atol=0.3 when
  the correct value is 0.5. This passes for any value in [0.2, 0.8].
- **Missing scaling factor**: Any formula involving genotypes that centers
  (subtracts 2p) but doesn't scale (divide by sqrt(2pq)).
- **scale=False in standardized context**: If the code path is supposed to
  produce standardized genotypes but passes `scale=False`.
- **Sibling-ICC instead of GRM**: `h2 = 2 * r_sib` is a DIFFERENT estimator
  from GRM-based HE. They are not interchangeable.
- **Small n in numerical tests**: n=10 for a heritability test will have
  huge sampling variance and prove nothing.

## Output Format

The review should produce a structured report:

```
## Adversarial Review: [commit hash or PR]

### Spec Violations Found
- [CRITICAL/MODERATE/MINOR] Description of violation
  - File: path:line
  - Spec reference: math_spec.md section N
  - Evidence: what the code does vs what the spec requires

### Tautological Tests Found
- File: path:line
  - Why it's tautological: ...
  - Suggested fix: ...

### Invariants Verified
- [x] Standardization uses sqrt(2pq)
- [x] h² round-trip preserved
- ...

### Risk Assessment
[Summary of overall risk level for this change]
```

## When to Run

- **Always**: Changes to `struct.py`, `stats.py`, `effect.py`, `arch.py`
- **Always**: Changes to numerical tests (`tests/numerical/`)
- **Recommended**: Changes to `sim.py`, `mate.py`, `reproduce.py`
- **Optional**: Changes to I/O, documentation, tooling
