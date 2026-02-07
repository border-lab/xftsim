"""Unit tests for GRG I/O (load_grg)."""
import numpy as np
import pytest

pygrgl = pytest.importorskip("pygrgl")

from xftsim.io import load_grg
from xftsim.struct import GraphHaplotypeOperator
from tests.testdata import TestGRG


class TestLoadGRG:
    def test_load_with_bim(self):
        op = load_grg(TestGRG.TINY_GRG_PATH, bim_path=TestGRG.TINY_BIM_PATH)
        assert isinstance(op, GraphHaplotypeOperator)
        assert op.n == 20
        assert op.m == 100
        assert op.variants.chrom is not None

    def test_load_without_bim(self):
        op = load_grg(TestGRG.TINY_GRG_PATH)
        assert isinstance(op, GraphHaplotypeOperator)
        assert op.variants.chrom is None

    def test_load_small(self):
        op = load_grg(TestGRG.SMALL_GRG_PATH, bim_path=TestGRG.SMALL_BIM_PATH)
        assert op.n == 100
        assert op.m == 1000

    def test_individual_ids(self):
        op = load_grg(TestGRG.TINY_GRG_PATH, bim_path=TestGRG.TINY_BIM_PATH)
        # Should have string IIDs from GRG
        assert len(op.samples.iid) == 20

    def test_nonexistent_path_raises(self):
        with pytest.raises(Exception):
            load_grg("/nonexistent/path.grg")

    def test_generation_passthrough(self):
        op = load_grg(TestGRG.TINY_GRG_PATH, generation=5)
        assert op.generation == 5
