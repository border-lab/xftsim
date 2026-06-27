#ifndef GRGL_RECOMBINATION_H
#define GRGL_RECOMBINATION_H

#include "grgl/grg.h"
#include "grgl/mutation.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace grgl {

/**
 * Non-owning read-only view over a contiguous range of T. C++11-compatible
 * stand-in for std::span (C++20). Used by NonDuplicationRecombiner to expose
 * cache contents without forcing callers to know whether the storage is a
 * CSR slice (m_inputPos[lo..hi)) or a heap vector (m_overridePos[node]).
 *
 * The pointer is valid only as long as the underlying storage is unmodified.
 * In particular: inserting into the override maps may rehash and invalidate
 * any Span pointing into them. Callers must re-acquire spans after any
 * potentially-mutating operation (extractBubble, lazy getXxxView on a new
 * node).
 */
template <typename T> class Span {
public:
    Span()
        : m_data(nullptr),
          m_size(0) {}
    Span(const T* data, std::size_t size)
        : m_data(data),
          m_size(size) {}

    const T* begin() const { return m_data; }
    const T* end() const { return m_data + m_size; }
    const T* data() const { return m_data; }
    const T& operator[](std::size_t i) const { return m_data[i]; }
    const T& front() const { return m_data[0]; }
    const T& back() const { return m_data[m_size - 1]; }
    std::size_t size() const { return m_size; }
    bool empty() const { return m_size == 0; }

private:
    const T* m_data;
    std::size_t m_size;
};

/**
 * Three-state interval cache slot. Mirrors the Python tri-state used for
 * span_cache[node] and anc_cov_cache[node]:
 *   - Uninit: cache miss; lazy DFS will populate
 *   - Empty:  computed; node has no own mutations and no parent spans
 *             (e.g. a root with no mutations) -- distinguished from Set so
 *             callers can avoid an extra sentinel value on (lo, hi)
 *   - Set:    [lo, hi] computed and stored
 *
 * Stored densely as std::vector<IntervalOpt>, indexed by NodeID. Default
 * construction leaves the slot in Uninit so std::vector::resize() correctly
 * initializes freshly grown entries on bubble/offspring creation.
 */
struct IntervalOpt {
    enum class State : uint8_t {
        Uninit = 0,
        Empty = 1,
        Set = 2,
    };

    BpPosition lo;
    BpPosition hi;
    State state;

    IntervalOpt()
        : lo(0),
          hi(0),
          state(State::Uninit) {}
    IntervalOpt(BpPosition lowIn, BpPosition highIn, State stateIn)
        : lo(lowIn),
          hi(highIn),
          state(stateIn) {}

    static IntervalOpt makeEmpty() { return IntervalOpt(0, 0, State::Empty); }
    static IntervalOpt makeSet(BpPosition lowIn, BpPosition highIn) { return IntervalOpt(lowIn, highIn, State::Set); }

    bool isUninit() const { return state == State::Uninit; }
    bool isEmpty() const { return state == State::Empty; }
    bool isSet() const { return state == State::Set; }
};

/**
 * Per-decision-case histogram for the 13-cell decision matrix in recurseAttach().
 * See benchmark/grg_recombination.py::_fresh_audit() for canonical cardinalities;
 * audit_check() asserts the following three invariants on these counters:
 *
 *   (1) extractBubbleCalls equals the sum of the six bubble-case counters.
 *   (2) Total connect() calls equal twice the bubble count plus firing direct
 *       attaches: (connectCallsInAttach + connectCallsInExtract) ==
 *       2 * extractBubbleCalls + directAttach + directAttachRoot + pathCompressionAttach.
 *   (3) makeNodeCalls equals recombineCalls plus extractBubbleCalls.
 *
 * Field names map to snake_case dict keys in the Python binding for parity
 * with the existing audit dict structure.
 */
struct AuditCounters {
    // Row "no relevant" (Mu cap I = empty)
    uint64_t pruning;               // has_no_relevant + ancestral_disjoint
    uint64_t pruningRoot;           // has_no_relevant + Iu is None
    uint64_t pathCompression;       // has_no_relevant + full_coverage (recursed)
    uint64_t pathCompressionAttach; // has_no_relevant + full_coverage + fanout>1
                                    // (attached)
    uint64_t decomposition;         // has_no_relevant + partial overlap
    // Row "all covered" (Mu subset I)
    uint64_t directAttach;     // has_all_relevant + full_coverage
    uint64_t directAttachRoot; // has_all_relevant + Iu is None
    uint64_t bubbleStrip;      // has_all_relevant + ancestral_disjoint
    uint64_t bubbleSplit;      // has_all_relevant + partial overlap
    uint64_t directAttachDup;  // connected_gen guard skipped a duplicate edge
    // Row "partial relevant"
    uint64_t bubbleFill;           // has_partial_relevant + full_coverage
    uint64_t bubbleStripPartial;   // has_partial_relevant + ancestral_disjoint
    uint64_t bubbleSplitPartial;   // has_partial_relevant + partial overlap
    uint64_t bubbleStripPartialRt; // has_partial_relevant + Iu is None
    // Informational skips (not in matrix)
    uint64_t skipEmptyInterval;  // L >= R at top of loop
    uint64_t skipAlreadyVisited; // gen_v guard fired
    uint64_t skipEmptyTrim;      // newL >= newR after trim
    // Primitives that reconcile against case sums
    uint64_t visits; // entries past both skip guards
    uint64_t extractBubbleCalls;
    uint64_t connectCallsInAttach;  // connects fired inside recurseAttach
    uint64_t connectCallsInExtract; // connects fired inside extractBubble (2/call)
    uint64_t makeNodeCalls;         // offspring + bubble nodes created
    uint64_t recombineCalls;        // recombine() + recombineMulti() entries
    uint64_t recurseAttachCalls;

    AuditCounters()
        : pruning(0),
          pruningRoot(0),
          pathCompression(0),
          pathCompressionAttach(0),
          decomposition(0),
          directAttach(0),
          directAttachRoot(0),
          bubbleStrip(0),
          bubbleSplit(0),
          directAttachDup(0),
          bubbleFill(0),
          bubbleStripPartial(0),
          bubbleSplitPartial(0),
          bubbleStripPartialRt(0),
          skipEmptyInterval(0),
          skipAlreadyVisited(0),
          skipEmptyTrim(0),
          visits(0),
          extractBubbleCalls(0),
          connectCallsInAttach(0),
          connectCallsInExtract(0),
          makeNodeCalls(0),
          recombineCalls(0),
          recurseAttachCalls(0) {}
};

/**
 * Phase-level wallclock + per-C++-call timing, populated when the recombiner
 * was constructed with `instrument = true`. Field names match the Python
 * `stats` dict in NonDuplicationRecombination._fresh_stats() (camelCase here,
 * snake_case in the dict serialization for parity with the Python interface).
 *
 * Overhead: chrono probes within instrumented phases add <2% to their
 * wallclock (~40-100 ns per steady_clock::now() on Linux x86_64, with ~10
 * probes per moved mutation in applyPendingBubbles). visitsTotal is counted
 * incrementally at the visit site in recurseAttach (O(1) per visit).
 */
struct RecombinerStats {
    // Phase wallclock (seconds)
    double initCachesTime;
    double recurseAttachTime;
    double applyBubblesTime;
    double syncToGrgTime;
    double clearCachesTime;
    double flushSamplesTime;
    double sortMutationsTime;
    // C++-call wallclock (seconds)
    double getMutationByIdTime;
    double addMutationTime;
    double removeMutationTime;
    double makeNodeTime;
    double connectTime;
    // C++-call counts
    uint64_t getMutationByIdCalls;
    uint64_t addMutationCalls;
    uint64_t removeMutationCalls;
    uint64_t makeNodeCalls;
    uint64_t connectCalls;
    // Algorithmic counters
    uint64_t offspringCount;
    uint64_t recurseAttachCalls;
    uint64_t segmentsProcessed;
    uint64_t visitsTotal;
    uint64_t bubblesCreated;
    uint64_t mutationsMoved;
    // Push-site pre-prune skips: parents that would have hit the `pruning` case
    // were detected from the parent and never pushed onto the DFS stack. Drops
    // the equivalent count from `visits` / `pruning` on the audit dict (the
    // skipped node was never visited), so audit-dict byte-parity with the
    // Python reference does not hold when pre-prune is enabled.
    uint64_t prePrunedSkips;

    RecombinerStats()
        : initCachesTime(0.0),
          recurseAttachTime(0.0),
          applyBubblesTime(0.0),
          syncToGrgTime(0.0),
          clearCachesTime(0.0),
          flushSamplesTime(0.0),
          sortMutationsTime(0.0),
          getMutationByIdTime(0.0),
          addMutationTime(0.0),
          removeMutationTime(0.0),
          makeNodeTime(0.0),
          connectTime(0.0),
          getMutationByIdCalls(0),
          addMutationCalls(0),
          removeMutationCalls(0),
          makeNodeCalls(0),
          connectCalls(0),
          offspringCount(0),
          recurseAttachCalls(0),
          segmentsProcessed(0),
          visitsTotal(0),
          bubblesCreated(0),
          mutationsMoved(0),
          prePrunedSkips(0) {}
};

/**
 * Non-duplication GRG recombination algorithm. Owns a MutableGRG plus all
 * per-node caches required by the iterative DFS over its parents. Constructing
 * the recombiner runs a one-pass O(V + E) topological initialization of every
 * cache, after which recombine() / recombineMulti() execute entirely in C++
 * with no Python callbacks (release the GIL via py::gil_scoped_release in the
 * binding).
 *
 * Cache layout: per-node mutation, position, and up-edge data are stored in
 * a hybrid CSR + override structure with a per-input-node dirty-flag fast
 * path. Input-graph nodes (NodeID < m_inputGraphNodeCount) live in three
 * pairs of flat CSR arrays (m_inputPos / m_inputMutIds / m_inputMutOffsets
 * and m_inputUpEdges / m_inputUpEdgesOffsets). Bubble + offspring nodes --
 * and any input node whose up-edges were modified by extractBubble or whose
 * mutations were relocated by applyPendingBubbles -- live in three override
 * hash maps that shadow the CSR. Lookups check the per-node m_dirty bitmask
 * (DIRTY_MUT=0x01, DIRTY_UP=0x02) first: when clear for the relevant slot,
 * the accessor reads the CSR directly with zero hash work. Only dirty input
 * nodes and bubble/offspring nodes consult the override map. See "Cache
 * flattening to SoA" and "Dirty-bit guard" in DESIGN.md.
 *
 * Push-site pre-prune (m_prePruneEnabled, default true): before pushing a
 * parent onto the DFS stack in recurseAttach, check whether it would land in
 * the `pruning` audit case (zero mutations + ancestral coverage disjoint
 * from the push interval). Such parents are skipped entirely -- no visit,
 * no `visits++`. The check is conservative; any uncertainty (mutation
 * override dirty, Uninit/Empty ancCov, bubble/offspring node) falls through
 * to a normal push + visit. Disable via setPrePruneEnabled(false) for
 * byte-for-byte audit-dict parity with the Python reference (which does not
 * pre-prune). See "Push-site pre-prune" in DESIGN.md.
 *
 * Cache invalidation rules (see recombination.cpp for the full discussion):
 *   - extractBubble() invalidates node_id (lost mutations) and bubble_id (new
 *     slot); it does NOT invalidate node_id's parents, since their span /
 *     anc_cov depend only on themselves and their own ancestors -- neither of
 *     which changes when a bubble is extracted.
 *   - extractBubble() eagerly refreshes m_overrideUpEdges[node_id] (sets
 *     DIRTY_UP) so the newly-added bubble parent is visible to subsequent
 *     within-call traversals.
 *   - endGeneration() calls sortMutations() and then fully rebuilds the
 *     input-graph mutation CSR (sortMutations renumbers MutationIds), clears
 *     m_overrideMutIds / m_overridePos, and clears DIRTY_MUT for every input
 *     node. The up-edges CSR and DIRTY_UP flags survive across generations.
 *
 * Offspring node-ID convention: negative IDs allocated by calling
 * grg->makeNode() with count=1 and negative=true. The reverse-lookup table
 * (m_negativeNodeIds) is exposed via getNegativeNodeIds() for the Python
 * wrapper, which mirrors it as the public NEGATIVE_NODE_IDS attribute used by
 * the simulator for stable offspring identifiers across generations.
 */
class NonDuplicationRecombiner {
public:
    /**
     * @param grg The MutableGRG to operate on. The recombiner holds a shared_ptr,
     *            so the GRG outlives the recombiner if the caller drops their
     *            reference. The one-pass cache initialization runs here.
     * @param instrument When true, populate the RecombinerStats accumulators
     *                   with std::chrono-derived timings on every phase and
     *                   every MutableGRG mutator call. When false (the default),
     *                   the audit counters still accumulate but no timing
     *                   probes fire. Audit counters are the test surface; stats
     *                   are the diagnostic surface for `--diagnostics`.
     */
    explicit NonDuplicationRecombiner(MutableGRGPtr grg, bool instrument = false);

    /**
     * Two-parent recombine with a single crossover at `breakpoint`. Equivalent
     * to recombineMulti({{hapA, breakpoint}, {hapB, genomeLength}}).
     *
     * @return Negative external offspring identifier `-(idx + 1)` where `idx`
     *         is the offspring's position in m_negativeNodeIds. The raw
     *         positive NodeID created by makeNode() is retrievable via
     *         getLastRawOffspringId().
     */
    SignedNodeID recombine(NodeID hapA, NodeID hapB, BpPosition breakpoint);

    /**
     * Multi-segment recombine.
     *
     * @param segments List of (sourceParent, endCoord) pairs covering
     *                 [0, genomeLength). The start coordinate of each segment
     *                 is implicit from the preceding segment's end (0 for the
     *                 first). The final endCoord must equal genomeLength.
     * @return Negative external offspring identifier `-(idx + 1)`; see
     *         recombine() for the relationship to the raw NodeID.
     */
    SignedNodeID recombineMulti(const std::vector<std::pair<NodeID, BpPosition>>& segments);

    /**
     * Apply pending bubble mutation moves. Called automatically at the end of
     * recombine() / recombineMulti(); exposed for the rare caller that wants
     * to control flushing explicitly.
     */
    void applyPendingBubbles();

    /** Flush any pending sample-removal updates by calling grg->setSamples(...).
     */
    void flushSampleUpdates();

    /**
     * End-of-generation hook. Encapsulates the post-loop bookkeeping that
     * simulate_grg_recombination drives once per generation:
     *   1. grg->sortMutations() (compacts soft-deleted mutation entries)
     *   2. Clear mutationCache + posCache (sortMutations renumbers MutationIds)
     *   3. Record sortMutationsTime if instrumented
     * Position-derived caches survive intentionally because positions do not
     * change across sortMutations.
     */
    void endGeneration();

    /**
     * Clear pending sample removals without flushing. Used after a wholesale
     * grg->setSamples() replaces the sample set entirely, making the
     * accumulated removal queue moot.
     */
    void clearPendingSampleRemovals();

    void setDeferSampleUpdates(bool defer) { m_deferSampleUpdates = defer; }
    bool getDeferSampleUpdates() const { return m_deferSampleUpdates; }

    void setDebugMode(bool d) { m_debug = d; }
    bool getDebugMode() const { return m_debug; }

    bool isInstrumented() const { return m_instrument; }

    /**
     * Enable/disable the push-site pre-prune optimization (default: on).
     *
     * When enabled, before pushing a parent onto the DFS stack we check
     * whether the parent would land in the `pruning` audit case — i.e. it
     * has zero mutations and its ancestral coverage is disjoint from the
     * push interval [newL, newR). Such parents are skipped entirely
     * (no visit, no `visits++`, no `pruning++`).
     *
     * Disable for byte-for-byte audit-dict parity with the Python
     * reference (the Python implementation does not pre-prune, so `visits`
     * and `pruning` counts diverge when this is on).
     */
    void setPrePruneEnabled(bool enabled) { m_prePruneEnabled = enabled; }
    bool getPrePruneEnabled() const { return m_prePruneEnabled; }

    const AuditCounters& getAudit() const { return m_audit; }
    void resetAudit() { m_audit = AuditCounters(); }

    const RecombinerStats& getStats() const { return m_stats; }

    /** Zero both stats and audit (matches the Python `reset_stats()` semantics).
     */
    void resetStats() {
        m_stats = RecombinerStats();
        m_audit = AuditCounters();
    }

    /**
     * Reverse-lookup table of offspring negative NodeIDs in creation order.
     * Mirrors the Python NEGATIVE_NODE_IDS list; the Python wrapper exposes
     * this directly as the public NEGATIVE_NODE_IDS attribute. Read-only;
     * managed internally as offspring are registered.
     */
    const std::vector<NodeID>& getNegativeNodeIds() const { return m_negativeNodeIds; }

    /**
     * Raw NodeID of the most recently registered offspring. Returns
     * INVALID_NODE_ID if no offspring has been created yet. Used by the
     * Python wrapper to grow its mirrored NEGATIVE_NODE_IDS list in O(1)
     * per recombine, avoiding the O(N) copy that getNegativeNodeIds() incurs
     * through pybind11's std::vector conversion.
     */
    NodeID getLastRawOffspringId() const {
        return m_negativeNodeIds.empty() ? INVALID_NODE_ID : m_negativeNodeIds.back();
    }

    /** Pointer to the GRG this recombiner operates on. */
    const MutableGRGPtr& getGrg() const { return m_grg; }

private:
    // --- Construction ---
    void buildAncestralCaches();
    void growNodeArrays(NodeID nodeId);
    void syncToGrg();

    // --- Hot path ---
    void recurseAttach(NodeID rootId, NodeID offspringId, BpPosition leftBound, BpPosition rightBound);
    NodeID extractBubble(NodeID nodeId, const std::vector<MutationId>& relMutIds, NodeID offspringId);

    SignedNodeID registerOffspring(NodeID offspringId);

    // --- Cache access (CSR for input nodes, override map for new / modified
    // input nodes). Each view points either into the flat CSR storage or into
    // a vector held by an override-map entry; the view is invalidated by any
    // subsequent operation that mutates the corresponding cache layer (see
    // Span<T> doc for the constraint).
    Span<BpPosition> getPositionsView(NodeID nodeId);
    Span<MutationId> getMutationIdsView(NodeID nodeId);
    Span<NodeID> getUpEdgesView(NodeID nodeId);

    // --- Lazy / eager override population helpers ---
    // Populate (or refresh) m_overrideMutIds[nodeId] + m_overridePos[nodeId]
    // from the GRG's current state. Used both for lazy-fetch of bubble /
    // offspring nodes on first access and for eager refresh of input nodes
    // whose mutations were relocated by applyPendingBubbles.
    void populateMutationOverride(NodeID nodeId);
    // Populate (or refresh) m_overrideUpEdges[nodeId] from the GRG. Used for
    // lazy-fetch on bubble nodes and for eager refresh of input nodes after
    // extractBubble.
    void populateUpEdgesOverride(NodeID nodeId);

    IntervalOpt getNodeAndAncestorSpan(NodeID nodeId);
    IntervalOpt getAncestralCoverage(NodeID nodeId);

    // --- Cleanup ---
    void clearModifiedCaches();
    // Rebuild m_inputPos / m_inputMutIds / m_inputMutOffsets from the GRG's
    // current state. Called by endGeneration after sortMutations renumbers
    // MutationIds; cheap (O(V + E)) and runs once per generation.
    void rebuildInputMutationCSR();

    // --- State ---
    struct BubbleOp {
        NodeID nodeId;
        NodeID bubbleId;
        std::vector<MutationId> relMutIds;
    };

    MutableGRGPtr m_grg;
    BpPosition m_genomeLength;

    // Per-node arrays (dense, indexed by NodeID, grown via growNodeArrays as
    // bubbles/offspring are created).
    std::vector<IntervalOpt> m_spanCache;
    std::vector<IntervalOpt> m_ancCovCache;
    std::vector<uint64_t> m_visitedGen;
    std::vector<uint64_t> m_connectedGen;
    uint64_t m_genVisited;
    uint64_t m_genConnected;

    // --- Input-graph CSR cache ---
    // m_inputGraphNodeCount: NodeID cutoff between "input-graph" and "new"
    // nodes. Nodes with NodeID < m_inputGraphNodeCount lived in the GRG at
    // construction time and have CSR entries; NodeIDs >= m_inputGraphNodeCount
    // are bubbles or offspring created mid-run and live in the override maps.
    // Stable across endGeneration (bubble/offspring NodeIDs from prior
    // generations remain >= m_inputGraphNodeCount).
    std::size_t m_inputGraphNodeCount;

    // Mutation IDs and positions, sorted by position within each node's slice.
    // m_inputMutIds[m_inputMutOffsets[node] .. m_inputMutOffsets[node + 1])
    // and m_inputPos[same slice] are parallel arrays (i-th entry of each
    // describes the same mutation). m_inputMutOffsets has length
    // m_inputGraphNodeCount + 1; the last element holds the total count.
    std::vector<MutationId> m_inputMutIds;
    std::vector<BpPosition> m_inputPos;
    std::vector<std::size_t> m_inputMutOffsets;

    // Up-edges flat array. m_inputUpEdges[m_inputUpEdgesOffsets[node] ..
    // m_inputUpEdgesOffsets[node + 1]) is node's parent list at construction
    // time. Bubbles added by extractBubble extend the parent list of input
    // nodes, but the CSR is never mutated; the new parent list is stored in
    // m_overrideUpEdges to shadow the CSR.
    std::vector<NodeID> m_inputUpEdges;
    std::vector<std::size_t> m_inputUpEdgesOffsets;

    // --- Override hash maps ---
    // Shadow the CSR when present; sole storage for bubble/offspring nodes
    // (NodeID >= m_inputGraphNodeCount). Lookups consult the override first.
    //
    // - m_overrideMutIds + m_overridePos: parallel; populated by
    //   populateMutationOverride. Both refreshed at clearModifiedCaches for
    //   input nodes whose mutations changed. Cleared at endGeneration (CSR
    //   gets rebuilt with current state).
    // - m_overrideUpEdges: populated by extractBubble (eager for the modified
    //   input node) and by populateUpEdgesOverride (lazy for new nodes).
    //   Survives endGeneration since up-edges are unaffected by sortMutations.
    std::unordered_map<NodeID, std::vector<MutationId>> m_overrideMutIds;
    std::unordered_map<NodeID, std::vector<BpPosition>> m_overridePos;
    std::unordered_map<NodeID, std::vector<NodeID>> m_overrideUpEdges;

    // --- Dirty flags for input-graph nodes ---
    // Per-node bitmask indicating which override maps shadow the CSR for this
    // input node. When a bit is clear the accessor skips the hash probe and
    // reads straight from the CSR. Indexed by NodeID; only meaningful for
    // NodeID < m_inputGraphNodeCount (bubble/offspring nodes always probe the
    // override map). Bit layout: 0x01 = mut/pos, 0x02 = up-edges.
    static const uint8_t DIRTY_MUT  = 0x01;
    static const uint8_t DIRTY_UP   = 0x02;
    std::vector<uint8_t> m_dirty;

    // Deferred work.
    std::vector<BubbleOp> m_pendingBubbles;
    std::unordered_set<NodeID> m_pendingSampleRemovals;
    std::unordered_set<NodeID> m_modifiedNodes;

    AuditCounters m_audit;
    RecombinerStats m_stats;
    bool m_instrument;
    bool m_debug;
    bool m_deferSampleUpdates;
    // Push-site pre-prune toggle; see setPrePruneEnabled() docs. Defaults to
    // true (optimization on). Initialized in the .cpp constructor.
    bool m_prePruneEnabled;

    // Offspring tracking (mirror of Python NEGATIVE_NODE_IDS +
    // _negative_node_index).
    std::vector<NodeID> m_negativeNodeIds;
    std::unordered_map<NodeID, size_t> m_negativeNodeIndex;
};

} // namespace grgl

#endif // GRGL_RECOMBINATION_H
