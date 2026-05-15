# Refactoring Plan

**→ Add in SampleMeta and VariantMeta to index the new genotype array**
- New Constructor that is DenseHaplotypeArray(genotypes, generation, samples, variants)

## Founders.py
- All functions include new version of DenseHaplotypeArray

## io.py
- read_plink() as pseudohaplotypes()
- haplotypes_from_sgkit_dataset()

## reproduce.py
- RecombinationMap.from_haplotypes() [adjust return values]
- Meiosis.reproduce()
- → Create offspring_samples of type SampleMeta
- Return new DenseHaplotypeArray()

## Sim.py
```python
self.pedigree = xft.ped.Pedigree(founder_haplotypes, samples)
```

## arch.py
```python
sample_indexer = SampleMeta
```
→ Used in instantiating PhenotypeArray
