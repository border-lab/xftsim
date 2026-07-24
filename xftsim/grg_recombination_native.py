"""
grg_recombination_native.py

Thin adapter around the C++ NonDuplicationRecombiner so that it exposes the
same interface as the Python ``NonDuplicationRecombination`` in
``grg_recombination.py``.  Callers (``GraphHaplotypeOperator.meiosis``, tests)
can swap implementations without changing their code.

Import will raise ``ImportError`` if the C++ extension
(``grg_recomb_native._grg_recomb_native``) is not installed.
"""

from grg_recomb_native._grg_recomb_native import NonDuplicationRecombiner as _NativeRecombiner


class NonDuplicationRecombination:
    """Drop-in replacement for
    ``grg_recombination.NonDuplicationRecombination`` backed by the C++
    ``NonDuplicationRecombiner``.

    Only the public surface consumed by ``GraphHaplotypeOperator.meiosis``
    and test code is wrapped; internal caches are not exposed.
    """

    debug_mode = False

    def __init__(self, grg, instrument=False):
        self._impl = _NativeRecombiner(grg, instrument)
        self.grg = grg
        self.genome_length = grg.bp_range[1]
        self.original_bp_range = grg.bp_range

    # -- attributes expected by struct.py meiosis ----------------------------

    @property
    def NEGATIVE_NODE_IDS(self):
        return self._impl.negative_node_ids

    @property
    def defer_sample_updates(self):
        return self._impl.defer_sample_updates

    @defer_sample_updates.setter
    def defer_sample_updates(self, value):
        self._impl.defer_sample_updates = value

    class _PendingSampleRemovalsProxy:
        __slots__ = ("_impl",)

        def __init__(self, impl):
            self._impl = impl

        def clear(self):
            self._impl.clear_pending_sample_removals()

    @property
    def _pending_sample_removals(self):
        return self._PendingSampleRemovalsProxy(self._impl)

    # -- debug / pre-prune ---------------------------------------------------

    @property
    def pre_prune_enabled(self):
        return self._impl.pre_prune_enabled

    @pre_prune_enabled.setter
    def pre_prune_enabled(self, value):
        self._impl.pre_prune_enabled = value

    # -- public entry points -------------------------------------------------

    def recombine(self, haplotype_A, haplotype_B, breakpoint):
        return self._impl.recombine(haplotype_A, haplotype_B, breakpoint)

    def recombine_multi(self, segments):
        return self._impl.recombine_multi(segments)

    def flush_sample_updates(self):
        self._impl.flush_sample_updates()

    def end_generation(self):
        self._impl.end_generation()

    # -- audit / stats (used by tests and diagnostics) -----------------------

    @property
    def audit(self):
        return self._impl.audit

    @property
    def stats(self):
        return self._impl.stats

    def reset_audit(self):
        self._impl.reset_audit()

    def reset_stats(self):
        self._impl.reset_stats()

    def audit_check(self, raise_on_fail=True):
        """Verify the three audit-counter identities.

        Returns a dict ``{identity_name: {pass, lhs, rhs}}`` mirroring
        the Python implementation.
        """
        a = self._impl.audit
        results = {}

        # (1) extractBubbleCalls == sum of six bubble-case counters
        lhs = a["extract_bubble_calls"]
        rhs = (a["bubble_strip"] + a["bubble_split"]
               + a["bubble_fill"]
               + a["bubble_strip_partial"] + a["bubble_split_partial"]
               + a["bubble_strip_partial_rt"])
        results["bubble_sum"] = {"pass": lhs == rhs, "lhs": lhs, "rhs": rhs}

        # (2) connect calls == 2*bubbles + direct attaches
        lhs = a["connect_calls_in_attach"] + a["connect_calls_in_extract"]
        rhs = (2 * a["extract_bubble_calls"]
               + a["direct_attach"] + a["direct_attach_root"]
               + a["path_compression_attach"])
        results["connect_sum"] = {"pass": lhs == rhs, "lhs": lhs, "rhs": rhs}

        # (3) makeNodeCalls == recombineCalls + extractBubbleCalls
        lhs = a["make_node_calls"]
        rhs = a["recombine_calls"] + a["extract_bubble_calls"]
        results["make_node_sum"] = {"pass": lhs == rhs, "lhs": lhs, "rhs": rhs}

        if raise_on_fail:
            for name, r in results.items():
                if not r["pass"]:
                    raise AssertionError(
                        f"Audit identity {name!r} failed: "
                        f"lhs={r['lhs']} != rhs={r['rhs']}"
                    )
        return results
