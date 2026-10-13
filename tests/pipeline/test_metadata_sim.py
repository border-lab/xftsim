"""
Pipeline tests: per-individual metadata (population / ancestry) carried
across generations by Simulation(metadata_rules=...).

Founders are split into three synthetic populations. SNP 0 is fixed for the
'A' allele in population A and absent elsewhere, so at generation 1 an
offspring's dosage there is exactly 2 * ancestry_A, which checks end to end
that offspring metadata lines up with offspring genotypes.
"""
import glob
import json
import os
import warnings

import numpy as np
import pytest

import xftsim as xft
from xftsim.arch import Architecture
from xftsim.effect import AdditiveEffects
from xftsim.inherit import AncestryProportions, Maternal, Custom
from xftsim.mate import RandomMating, LinearAssortativeMating, GroupedMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import Simulation
from xftsim.stats import MetadataSummary
from tests.testdata import TestSimulation

N, M, SEED = 600, 40, 7
POPS = np.array(['A', 'B', 'C'])
ANC = ['ancestry_A', 'ancestry_B', 'ancestry_C']


@pytest.fixture
def founders():
    hap = TestSimulation.founder_haplotypes(n=N, m=M, seed=SEED)
    pop = POPS[np.arange(N) % 3]
    hap.genotypes[:, 0, :] = (pop == 'A')[:, None]
    hap.samples = hap.samples.with_extra({'population': pop.astype(object)})
    return hap


def _arch(extra_lines=""):
    eff = AdditiveEffects.from_h2(h2=0.5, m=M, seed=SEED)
    return Architecture.from_formula(f"""
    Y.G ~ genetic(eff)
    Y.E ~ noise(0.5)
    {extra_lines}
    Y ~ Y.G + Y.E
    """, effects={'eff': eff})


def _sim(founders, rules, n_gen=6, arch=None, mating=None, **kw):
    sim = Simulation(
        founder_haplotypes=founders, architecture=arch or _arch(),
        mating_regime=mating or RandomMating(offspring_per_pair=2),
        recombination_map=RecombinationMap.constant_map(m=M, p=0.1),
        seed=SEED, retain_phenotypes=n_gen, metadata_rules=rules, **kw,
    )
    sim.run(n_gen)
    return sim


def _anc(samples):
    return np.column_stack([samples.extra[k] for k in ANC])


RULES = [AncestryProportions('population'), Maternal('population')]


class TestAncestryInvariants:
    def test_fields_present_every_generation(self, founders):
        sim = _sim(founders, RULES, retain_haplotypes=6)
        assert len(sim.haplotype_history) == 6
        for gen, pheno in sim.phenotype_history.items():
            assert set(pheno.samples.extra) == {'population', *ANC}, gen
            assert sim.haplotype_history[gen].samples is pheno.samples
            if gen > 0:
                assert sim.pedigree_history[gen].offspring_samples is pheno.samples

    def test_rows_sum_to_one_and_are_dyadic(self, founders):
        sim = _sim(founders, RULES)
        for gen, pheno in sim.phenotype_history.items():
            anc = _anc(pheno.samples)
            np.testing.assert_array_equal(anc.sum(axis=1), 1.0)
            scaled = anc * 2 ** gen
            np.testing.assert_array_equal(scaled, np.round(scaled))

    def test_population_mean_conserved_exactly(self, founders):
        """Balanced sexes, everyone mates once, 2 offspring per pair."""
        sim = _sim(founders, RULES)
        sums = [_anc(p.samples).sum(axis=0) for p in sim.phenotype_history.values()]
        for s in sums[1:]:
            np.testing.assert_array_equal(s, sums[0])

    def test_maternal_label_matches_mother(self, founders):
        sim = _sim(founders, RULES, n_gen=3)
        ped = sim.pedigree_history[2]
        mothers = sim.phenotype_history[1].samples.extra['population']
        np.testing.assert_array_equal(
            sim.phenotype_history[2].samples.extra['population'],
            mothers[ped.maternal_idx])

    def test_gen1_dosage_equals_twice_ancestry(self, founders):
        sim = _sim(founders, RULES, n_gen=2, retain_haplotypes=2)
        hap = sim.haplotype_history[1]
        dosage = hap.genotypes[:, 0, :].sum(axis=1)
        np.testing.assert_array_equal(dosage, 2 * hap.samples.extra['ancestry_A'])

    def test_variance_halves_under_random_mating(self, founders):
        sim = _sim(founders, RULES)
        v = [_anc(p.samples).var(axis=0).mean() for p in sim.phenotype_history.values()]
        ratios = np.array(v[1:]) / np.array(v[:-1])
        assert 0.4 < ratios.mean() < 0.6

    def test_ancestry_assortment_slows_decay(self, founders):
        arch = _arch("A ~ meta(ancestry_A)")
        rand = _sim(founders, RULES, n_gen=5, arch=arch)
        assort = _sim(founders, RULES, n_gen=5, arch=arch,
                      mating=LinearAssortativeMating(['A'], r=0.9))
        v_rand = rand.phenotype_history[4].samples.extra['ancestry_A'].var()
        v_assort = assort.phenotype_history[4].samples.extra['ancestry_A'].var()
        assert v_assort > 3 * v_rand
        # The meta() node copies the metadata into the phenotypes.
        np.testing.assert_array_equal(
            assort.phenotype_history[4]['A'],
            assort.phenotype_history[4].samples.extra['ancestry_A'])


class TestSimulationBehavior:
    def test_founder_object_unchanged(self, founders):
        before = founders.samples
        _sim(founders, RULES, n_gen=2)
        assert founders.samples is before
        assert set(founders.samples.extra) == {'population'}

    def test_deterministic_rules_are_bit_identical(self, founders):
        plain = _sim(founders, None, n_gen=4, retain_haplotypes=4)
        tracked = _sim(founders, RULES, n_gen=4, retain_haplotypes=4)
        for gen in plain.haplotype_history:
            np.testing.assert_array_equal(plain.haplotype_history[gen].genotypes,
                                          tracked.haplotype_history[gen].genotypes)
        for gen in plain.phenotype_history:
            np.testing.assert_array_equal(plain.phenotype_history[gen]['Y'],
                                          tracked.phenotype_history[gen]['Y'])
        for gen in plain.pedigree_history:
            np.testing.assert_array_equal(plain.pedigree_history[gen].maternal_idx,
                                          tracked.pedigree_history[gen].maternal_idx)

    def test_no_rules_keeps_legacy_behavior(self, founders):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            sim = _sim(founders, None, n_gen=2)
        assert sim.phenotype_history[1].samples.extra == {}

    def test_unruled_fields_warn_once(self, founders):
        with pytest.warns(UserWarning, match="population") as record:
            _sim(founders, [], n_gen=4)
        assert sum('no inheritance rule' in str(w.message) for w in record) == 1

    def test_missing_rule_fails_before_running(self, founders):
        arch = _arch("Y.D ~ noise(0.1) | population")
        sim = Simulation(
            founder_haplotypes=founders, architecture=arch,
            mating_regime=RandomMating(),
            recombination_map=RecombinationMap.constant_map(m=M, p=0.1),
            seed=SEED, metadata_rules=[AncestryProportions()],
        )
        with pytest.raises(ValueError, match="Maternal\\('population'\\)"):
            sim.run(3)
        assert sim.phenotype_history == {}

    def test_validation_without_rules(self, founders):
        """Without rules these consumers used to return nothing after gen 0."""
        with pytest.raises(ValueError, match="'population'"):
            _sim(founders, None, n_gen=2,
                 filters={'f': xft.filters.MetadataFilter('population', values=['A'])})

    def test_validation_covers_mating_filters_stats(self, founders):
        with pytest.raises(ValueError) as exc:
            _sim(founders, [], n_gen=2,
                 mating=GroupedMating(RandomMating(), by='deme'),
                 filters={'f': xft.filters.MetadataFilter('band', values=[1])},
                 statistics=[MetadataSummary(keys=['tag'])])
        msg = str(exc.value)
        assert "'deme'" in msg and "'band'" in msg and "'tag'" in msg

    def test_single_generation_needs_no_rules(self, founders):
        arch = _arch("Y.D ~ noise(0.1) | population")
        _sim(founders, [], n_gen=1, arch=arch)

    def test_grouping_by_inherited_label(self, founders):
        arch = _arch("Y.D ~ noise(0.3) | population")
        sim = _sim(founders, RULES, n_gen=3, arch=arch)
        for pheno in sim.phenotype_history.values():
            pop = pheno.samples.extra['population']
            for label in POPS:
                assert np.unique(pheno['Y.D'][pop == label]).size == 1

    def test_grouped_mating_prevents_admixture(self, founders):
        sim = _sim(founders, RULES, n_gen=4,
                   mating=GroupedMating(RandomMating(), by='population'))
        for pheno in sim.phenotype_history.values():
            s = pheno.samples
            for label in POPS:
                np.testing.assert_array_equal(
                    s.extra[f'ancestry_{label}'],
                    (s.extra['population'] == label).astype(float))

    def test_metadata_summary_trajectory(self, founders):
        sim = _sim(founders, RULES, statistics=[MetadataSummary(keys=ANC)])
        var = [r.statistics['MetadataSummary']['ancestry_A']['var'] for r in sim.results]
        mean = [r.statistics['MetadataSummary']['ancestry_A']['mean'] for r in sim.results]
        assert len(var) == 6
        assert var[0] == pytest.approx(2 / 9)
        assert all(b < a for a, b in zip(var, var[1:]))
        np.testing.assert_allclose(mean, 1 / 3)


class TestCheckpoint:
    def test_resume_matches_uninterrupted_run(self, founders, tmp_path):
        full = _sim(founders, RULES, n_gen=5)
        part = _sim(founders, RULES, n_gen=3)
        xft.io.save_simulation_checkpoint(part, str(tmp_path))
        resumed = Simulation.from_checkpoint(str(tmp_path))
        resumed.continue_run(2)
        for gen in (3, 4):
            a = full.phenotype_history[gen]
            b = resumed.phenotype_history[gen]
            np.testing.assert_array_equal(a['Y'], b['Y'])
            for k, v in a.samples.extra.items():
                np.testing.assert_array_equal(b.samples.extra[k], v)

    def test_roundtrip_and_resume(self, founders, tmp_path):
        sim = _sim(founders, RULES, n_gen=3)
        xft.io.save_simulation_checkpoint(sim, str(tmp_path))
        resumed = Simulation.from_checkpoint(str(tmp_path))
        assert [r.to_config() for r in resumed.metadata_rules] == \
            [r.to_config() for r in RULES]
        for gen, pheno in sim.phenotype_history.items():
            back = resumed.phenotype_history[gen].samples
            assert back.generation == gen
            assert set(back.extra) == set(pheno.samples.extra)
            for k, v in pheno.samples.extra.items():
                np.testing.assert_array_equal(back.extra[k], v)
        ped = resumed.pedigree_history[2].offspring_samples
        assert ped.generation == 2 and 'ancestry_A' in ped.extra
        resumed.continue_run(2)
        s = resumed.phenotypes.samples
        assert s.generation == 4
        np.testing.assert_array_equal(_anc(s).sum(axis=1), 1.0)

    def test_custom_rule_must_be_repassed(self, founders, tmp_path):
        double = Custom('double_A', lambda m, f, rng: {
            'double_A': m['ancestry_A'] + f['ancestry_A']})
        founders.samples = founders.samples.with_extra(
            {'double_A': np.zeros(N)}, replace=False)
        rules = RULES + [double]
        sim = _sim(founders, rules, n_gen=2)
        xft.io.save_simulation_checkpoint(sim, str(tmp_path))
        # Loading for inspection works; resuming without the function fails.
        assert 'metadata_rules' in xft.io.load_simulation_checkpoint(str(tmp_path))
        with pytest.raises(ValueError, match="Custom"):
            Simulation.from_checkpoint(str(tmp_path))
        resumed = Simulation.from_checkpoint(str(tmp_path), metadata_rules=rules)
        resumed.continue_run(1)
        assert 'double_A' in resumed.phenotypes.samples.extra

    def test_old_checkpoint_without_metadata(self, founders, tmp_path):
        sim = _sim(founders, None, n_gen=2)
        xft.io.save_simulation_checkpoint(sim, str(tmp_path))
        # Rewrite the checkpoint in the pre-metadata format.
        for path in glob.glob(str(tmp_path / '*' / '*.npz')):
            data = dict(np.load(path, allow_pickle=True))
            np.savez(path, **{k: v for k, v in data.items()
                              if 'extra' not in k
                              and k not in ('sample_generation', 'offspring_generation')})
        meta_path = os.path.join(str(tmp_path), 'meta.json')
        meta = json.load(open(meta_path))
        del meta['metadata_rules']
        json.dump(meta, open(meta_path, 'w'))

        old = Simulation.from_checkpoint(str(tmp_path))
        assert old.metadata_rules is None
        assert old.phenotype_history[1].samples.extra == {}
        old.continue_run(1)

        resumed = Simulation.from_checkpoint(
            str(tmp_path), metadata_rules=[AncestryProportions()])
        with pytest.raises(KeyError, match="checkpoint"):
            resumed.continue_run(1)
