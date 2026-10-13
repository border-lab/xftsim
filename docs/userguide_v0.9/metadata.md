% module xftsim

# Metadata and ancestry

Every individual carries a `SampleMeta`: `iid`, `fid`, `sex`, `generation`,
and an `extra` dict of arbitrary per-individual arrays. Founders built with
`founder_haplotypes_from_stdpopsim_grg()` store each individual's
population label in `samples.extra["population"]`; you can add your own
fields with `SampleMeta.with_extra()`.

Offspring don't inherit these fields automatically. Each field needs an
**inheritance rule** saying how a child's value follows from its parents'.
Without rules, `extra` is empty from generation 1 on.

## Inheritance rules

Pass rules from `xftsim.inherit` to the simulation:

```python
import xftsim as xft
from xftsim.inherit import AncestryProportions, Maternal

sim = xft.sim.Simulation(
    founder_haplotypes=founders, architecture=arch,
    mating_regime=xft.mate.RandomMating(),
    recombination_map=rmap, seed=1,
    metadata_rules=[AncestryProportions('population'),
                    Maternal('population')],
)
sim.run(10)
sim.phenotypes.samples.to_dataframe().head()
```

| Rule | Offspring value |
|------|-----------------|
| `Midparent(*keys)` | mean of the parents (numeric fields) |
| `Maternal(*keys)` / `Paternal(*keys)` | copy of the mother's / father's value |
| `RandomParent(*keys)` | copy from a randomly chosen parent (draws from the simulation RNG) |
| `Custom(keys, fn)` | `fn(mother, father, rng) -> dict`, where `mother` and `father` hold all parent fields aligned to the offspring |
| `AncestryProportions(source, prefix)` | see below |

**Ancestry proportions.** `AncestryProportions('population')` turns the
founders' labels into one float column per population (`ancestry_YRI`,
`ancestry_CEU`, …), one-hot at generation 0. From then on each column is
the mean of the parents' values, so every row sums to 1. These are
*expected* global ancestry proportions given the pedigree; realized
genome-wide ancestry also varies with Mendelian segregation, which is not
tracked. Founders that already have fractional `ancestry_*` columns (e.g.
from admixed real data) are used as given.

Under random mating, the population mean of each column stays put and its
variance roughly halves every generation as the population becomes admixed.

**Checks.** Every field that a later generation reads must come from some
rule: `| field` grouping, `meta(field)`, `GroupedMating(by=field)`,
`MetadataFilter(field, ...)` or `MetadataSummary(keys=[field])`. If one
doesn't, `run()` raises before generation 0. Parent fields that no rule
handles are dropped with one warning. Deterministic rules never draw random
numbers, so adding them leaves a seeded simulation's genotypes and
phenotypes unchanged.

## Using metadata

**In the architecture.** `meta(field)` copies a numeric field into the
phenotypes, where aggregations and other components can use it:

```text
Y.G ~ genetic(eff)
Y.E ~ noise(0.5)
Y.A ~ meta(ancestry_YRI)
Y ~ Y.G + Y.E + 0.3*Y.A
```

Any categorical field also works as a grouping variable, for example a
shared environment per deme: `Y.D ~ noise(0.2) | deme`.

**In mating.** Mate on ancestry by adding the `meta()` node and naming it in
an assortative regime, e.g. `LinearAssortativeMating(['Y.A'], r=0.6)`. To
allow mating only within groups, wrap any regime:

```python
mating = xft.mate.GroupedMating(xft.mate.RandomMating(), by='population')
```

Combined with `Maternal('population')`, this keeps the populations
reproductively isolated, so ancestry stays one-hot.

**In filters and statistics.**

```python
filters = {'yri': xft.filters.MetadataFilter('ancestry_YRI',
                                             predicate=lambda a: a >= 0.75)}
statistics = [
    xft.stats.MetadataSummary(by='population'),         # mean/var/counts per generation
    xft.stats.SampleStatistics(filter_name='yri'),      # covariance in the subset
]
```

`MetadataSummary` results go into `sim.results` every generation, so the
full trajectory survives the `retain_phenotypes` pruning.

## Saving and resuming

Checkpoints (`xft.io.save_simulation_checkpoint`) and the npz helpers
save `SampleMeta.extra` with each generation, and checkpoints also save the
built-in rules. A `Custom` rule's function can't be saved, so when resuming
pass the rules again:

```python
sim = xft.sim.Simulation.from_checkpoint(path, metadata_rules=rules)
```
