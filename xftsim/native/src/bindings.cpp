// pybind11 binding for the NonDuplicationRecombiner C++ class.
//
// Field-name conventions in the Python-facing surface:
//   - Class methods are snake_case to match the existing Python
//     NonDuplicationRecombination interface (e.g. recombine_multi, end_generation).
//   - Audit / stats dict keys are snake_case (e.g. "extract_bubble_calls",
//     "sort_mutations_time") so audit_check / audit_summary / --diagnostics
//     in the Python wrapper continue to work byte-for-byte.
//   - recombine() and recombine_multi() release the GIL via call_guard since
//     no Python callbacks fire inside the C++ inner loop.

#include "grgl/recombination.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

py::dict toAuditDict(const grgl::AuditCounters& a) {
    py::dict d;
    d["pruning"] = a.pruning;
    d["pruning_root"] = a.pruningRoot;
    d["path_compression"] = a.pathCompression;
    d["path_compression_attach"] = a.pathCompressionAttach;
    d["decomposition"] = a.decomposition;
    d["direct_attach"] = a.directAttach;
    d["direct_attach_root"] = a.directAttachRoot;
    d["bubble_strip"] = a.bubbleStrip;
    d["bubble_split"] = a.bubbleSplit;
    d["direct_attach_dup"] = a.directAttachDup;
    d["bubble_fill"] = a.bubbleFill;
    d["bubble_strip_partial"] = a.bubbleStripPartial;
    d["bubble_split_partial"] = a.bubbleSplitPartial;
    d["bubble_strip_partial_rt"] = a.bubbleStripPartialRt;
    d["skip_empty_interval"] = a.skipEmptyInterval;
    d["skip_already_visited"] = a.skipAlreadyVisited;
    d["skip_empty_trim"] = a.skipEmptyTrim;
    d["visits"] = a.visits;
    d["extract_bubble_calls"] = a.extractBubbleCalls;
    d["connect_calls_in_attach"] = a.connectCallsInAttach;
    d["connect_calls_in_extract"] = a.connectCallsInExtract;
    d["make_node_calls"] = a.makeNodeCalls;
    d["recombine_calls"] = a.recombineCalls;
    d["recurse_attach_calls"] = a.recurseAttachCalls;
    return d;
}

py::dict toStatsDict(const grgl::RecombinerStats& s) {
    py::dict d;
    d["init_caches_time"] = s.initCachesTime;
    d["recurse_attach_time"] = s.recurseAttachTime;
    d["apply_bubbles_time"] = s.applyBubblesTime;
    d["sync_to_grg_time"] = s.syncToGrgTime;
    d["clear_caches_time"] = s.clearCachesTime;
    d["flush_samples_time"] = s.flushSamplesTime;
    d["sort_mutations_time"] = s.sortMutationsTime;
    d["get_mutation_by_id_time"] = s.getMutationByIdTime;
    d["add_mutation_time"] = s.addMutationTime;
    d["remove_mutation_time"] = s.removeMutationTime;
    d["make_node_time"] = s.makeNodeTime;
    d["connect_time"] = s.connectTime;
    d["get_mutation_by_id_calls"] = s.getMutationByIdCalls;
    d["add_mutation_calls"] = s.addMutationCalls;
    d["remove_mutation_calls"] = s.removeMutationCalls;
    d["make_node_calls"] = s.makeNodeCalls;
    d["connect_calls"] = s.connectCalls;
    d["offspring_count"] = s.offspringCount;
    d["recurse_attach_calls"] = s.recurseAttachCalls;
    d["segments_processed"] = s.segmentsProcessed;
    d["visits_total"] = s.visitsTotal;
    d["bubbles_created"] = s.bubblesCreated;
    d["mutations_moved"] = s.mutationsMoved;
    d["pre_pruned_skips"] = s.prePrunedSkips;
    return d;
}

size_t numNodesOf(grgl::MutableGRGPtr grg) { return grg->numNodes(); }

} // namespace

PYBIND11_MODULE(_grg_recomb_native, m) {
    m.doc() = "Native C++ recombination algorithms for pygrgl.";

    py::class_<grgl::NonDuplicationRecombiner>(m, "NonDuplicationRecombiner")
        .def(py::init<grgl::MutableGRGPtr, bool>(),
             py::arg("grg"),
             py::arg("instrument") = false,
             "Construct a recombiner over the given MutableGRG. Runs a one-pass\n"
             "O(V + E) cache initialization eagerly; subsequent recombine() calls\n"
             "execute entirely in C++ with no Python callbacks. When instrument is\n"
             "true, per-phase wallclock timings are recorded in stats; audit\n"
             "counters accumulate either way.")
        .def("recombine",
             &grgl::NonDuplicationRecombiner::recombine,
             py::arg("hap_a"),
             py::arg("hap_b"),
             py::arg("breakpoint"),
             py::call_guard<py::gil_scoped_release>(),
             "Two-parent recombine with a single crossover at `breakpoint`. Returns\n"
             "the negative NodeID of the newly created offspring.")
        .def("recombine_multi",
             &grgl::NonDuplicationRecombiner::recombineMulti,
             py::arg("segments"),
             py::call_guard<py::gil_scoped_release>(),
             "Multi-segment recombine. `segments` is a list of (source_parent,\n"
             "end_coord) pairs covering [0, genome_length). Returns the negative\n"
             "NodeID of the newly created offspring.")
        .def("apply_pending_bubbles", &grgl::NonDuplicationRecombiner::applyPendingBubbles)
        .def("flush_sample_updates", &grgl::NonDuplicationRecombiner::flushSampleUpdates)
        .def("end_generation",
             &grgl::NonDuplicationRecombiner::endGeneration,
             "End-of-generation hook: sorts mutations on the GRG, clears the\n"
             "mutation/position caches (sortMutations renumbers MutationIds), and\n"
             "records sort_mutations_time if instrumented. Position-derived caches\n"
             "(span, anc_cov, up_edges) survive intentionally.")
        .def("clear_pending_sample_removals",
             &grgl::NonDuplicationRecombiner::clearPendingSampleRemovals,
             "Drop accumulated pending-sample-removals. Called after a wholesale\n"
             "grg.set_samples(...) makes the accumulated removals moot.")
        .def("reset_audit", &grgl::NonDuplicationRecombiner::resetAudit)
        .def("reset_stats", &grgl::NonDuplicationRecombiner::resetStats)
        .def_property("defer_sample_updates",
                      &grgl::NonDuplicationRecombiner::getDeferSampleUpdates,
                      &grgl::NonDuplicationRecombiner::setDeferSampleUpdates)
        .def_property("pre_prune_enabled",
                      &grgl::NonDuplicationRecombiner::getPrePruneEnabled,
                      &grgl::NonDuplicationRecombiner::setPrePruneEnabled,
                      "Push-site pre-prune optimization toggle (default: True). Disable\n"
                      "for byte-for-byte audit-dict parity with the Python reference;\n"
                      "the Python implementation does not pre-prune, so the C++ `visits`\n"
                      "and `pruning` counts diverge when this is on. See\n"
                      "setPrePruneEnabled() docs on the C++ class for details.")
        .def_property(
            "debug_mode", &grgl::NonDuplicationRecombiner::getDebugMode, &grgl::NonDuplicationRecombiner::setDebugMode)
        .def_property_readonly("instrument", &grgl::NonDuplicationRecombiner::isInstrumented)
        .def_property_readonly(
            "audit",
            [](const grgl::NonDuplicationRecombiner& self) { return toAuditDict(self.getAudit()); },
            "Snapshot of the audit-counter dict. Returns a fresh dict on each\n"
            "access; modifications to the returned dict do not affect the recombiner.")
        .def_property_readonly(
            "stats",
            [](const grgl::NonDuplicationRecombiner& self) { return toStatsDict(self.getStats()); },
            "Snapshot of the stats dict (phase/C++-call wallclock plus algorithmic\n"
            "counters). Returns a fresh dict on each access. Populated only when\n"
            "the recombiner was constructed with instrument=True.")
        .def_property_readonly("negative_node_ids", &grgl::NonDuplicationRecombiner::getNegativeNodeIds)
        .def_property_readonly("last_raw_offspring_id", &grgl::NonDuplicationRecombiner::getLastRawOffspringId)
        .def_property_readonly("grg", &grgl::NonDuplicationRecombiner::getGrg);

    // Surviving sentinel from the build-infrastructure smoke test; harmless
    // and convenient for diagnostic prodding from Python.
    m.def("_smoke_num_nodes",
          &numNodesOf,
          py::arg("grg"),
          "Returns grg.num_nodes; verifies cross-module MutableGRG type sharing.");
}
