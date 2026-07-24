// NonDuplicationRecombiner implementation.
//
// Direct C++ port of benchmark/grg_recombination.py's NonDuplicationRecombination.
// Method semantics are preserved exactly so parity tests against the Python
// reference can drive correctness verification. See benchmark/native/DESIGN.md
// for the full porting decisions and benchmark/recombination_time_breakdown.md
// for the perf motivation.

#include "grgl/recombination.h"

#include "grgl/common.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <deque>
#include <limits>
#include <stdexcept>
#include <utility>

namespace grgl {

namespace {

// Sentinel used during span/anc-cov accumulation. Real BpPosition values are
// uint64_t, so we use the max as "no value yet" and accumulate via min/max.
constexpr BpPosition NO_POS_SENTINEL = std::numeric_limits<BpPosition>::max();

inline double elapsedSeconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

} // namespace

NonDuplicationRecombiner::NonDuplicationRecombiner(MutableGRGPtr grg, bool instrument)
    : m_grg(std::move(grg)),
      m_genomeLength(0),
      m_genVisited(0),
      m_genConnected(0),
      m_inputGraphNodeCount(0),
      m_audit(),
      m_stats(),
      m_instrument(instrument),
      m_debug(false),
      m_deferSampleUpdates(false),
      m_prePruneEnabled(true) {
    if (!m_grg) {
        throw std::invalid_argument("NonDuplicationRecombiner: null MutableGRGPtr");
    }
    m_genomeLength = m_grg->getBPRange().second;

    const size_t n = m_grg->numNodes();
    m_inputGraphNodeCount = n;
    m_spanCache.resize(n);   // default-constructed -> Uninit
    m_ancCovCache.resize(n); // default-constructed -> Uninit
    m_visitedGen.resize(n, 0);
    m_connectedGen.resize(n, 0);
    m_dirty.resize(n, 0);

    buildAncestralCaches();
}

// One-pass O(V + E) Kahn's-sort initialization of every per-node cache.
//
// Builds the input-graph CSR arrays:
//   m_inputUpEdges + m_inputUpEdgesOffsets   (parent lists, flat)
//   m_inputMutIds + m_inputPos + m_inputMutOffsets (mutation IDs + positions,
//                                                   parallel, sorted by
//                                                   position within each node)
// plus the dense per-node interval caches m_spanCache / m_ancCovCache.
//
// Replaces the previous hash-keyed caches (m_mutationCache, m_posCache,
// m_upEdgesCache); see DESIGN.md "Cache flattening to SoA" for motivation.
// Bubble / offspring nodes created mid-run live in m_overrideMutIds /
// m_overridePos / m_overrideUpEdges instead.
void NonDuplicationRecombiner::buildAncestralCaches() {
    std::chrono::steady_clock::time_point tStart;
    if (m_instrument) {
        tStart = std::chrono::steady_clock::now();
    }

    const size_t n = m_grg->numNodes();
    if (n == 0) {
        if (m_instrument) {
            m_stats.initCachesTime += elapsedSeconds(tStart);
        }
        return;
    }

    // Pass 1: fetch up-edges into a temporary, compute in-degrees, then build
    // the flat CSR. We can't avoid the per-node getUpEdges() copy (the API
    // returns by value), but flattening into m_inputUpEdges drops one
    // allocation per node compared to the old hash-map layout.
    std::vector<NodeIDList> upEdgesTmp(n);
    std::vector<uint32_t> inDegree(n, 0);
    m_inputUpEdgesOffsets.assign(n + 1, 0);
    for (NodeID node = 0; node < n; node++) {
        upEdgesTmp[node] = m_grg->getUpEdges(node);
        const size_t fanin = upEdgesTmp[node].size();
        inDegree[node] = static_cast<uint32_t>(fanin);
        m_inputUpEdgesOffsets[node + 1] = m_inputUpEdgesOffsets[node] + fanin;
    }
    m_inputUpEdges.resize(m_inputUpEdgesOffsets[n]);
    for (NodeID node = 0; node < n; node++) {
        const size_t off = m_inputUpEdgesOffsets[node];
        const NodeIDList& parents = upEdgesTmp[node];
        for (size_t i = 0; i < parents.size(); i++) {
            m_inputUpEdges[off + i] = parents[i];
        }
    }

    // Seed: roots (no parents) are immediately processable.
    std::deque<NodeID> queue;
    for (NodeID node = 0; node < n; node++) {
        if (inDegree[node] == 0) {
            queue.push_back(node);
        }
    }

    // Pass 2 (Kahn's topological sweep): for each node compute its sorted
    // mutation list (stored temporarily) and its span / anc_cov caches.
    // The parent's span is already computed by Kahn's invariant.
    std::vector<std::vector<MutationId>> mutIdsTmp(n);
    std::vector<std::vector<BpPosition>> posTmp(n);

    while (!queue.empty()) {
        const NodeID node = queue.front();
        queue.pop_front();

        // ---- Build sorted (mutationId, position) for `node` ----
        std::vector<MutationId> rawMutIds = m_grg->getMutationsForNode<MutationId>(node, /*allowSort=*/false);
        std::vector<std::pair<MutationId, BpPosition>> mutPairs;
        mutPairs.reserve(rawMutIds.size());
        for (MutationId mid : rawMutIds) {
            const Mutation& mut = m_grg->getMutationById(mid);
            mutPairs.emplace_back(mid, mut.getPosition());
        }
        if (mutPairs.size() > 1) {
            std::sort(mutPairs.begin(),
                      mutPairs.end(),
                      [](const std::pair<MutationId, BpPosition>& a, const std::pair<MutationId, BpPosition>& b) {
                          return a.second < b.second;
                      });
        }
        mutIdsTmp[node].reserve(mutPairs.size());
        posTmp[node].reserve(mutPairs.size());
        for (const std::pair<MutationId, BpPosition>& mp : mutPairs) {
            mutIdsTmp[node].push_back(mp.first);
            posTmp[node].push_back(mp.second);
        }

        // ---- Compute span_cache[node] = own muts (min,max) U each parent's span ----
        BpPosition minP = NO_POS_SENTINEL;
        BpPosition maxP = 0;
        bool anySet = false;
        if (!posTmp[node].empty()) {
            minP = posTmp[node].front();
            maxP = posTmp[node].back();
            anySet = true;
        }

        const NodeIDList& parents = upEdgesTmp[node];
        for (NodeID parent : parents) {
            const IntervalOpt& pSpan = m_spanCache[parent];
            if (pSpan.isSet()) {
                if (!anySet) {
                    minP = pSpan.lo;
                    maxP = pSpan.hi;
                    anySet = true;
                } else {
                    if (pSpan.lo < minP) {
                        minP = pSpan.lo;
                    }
                    if (pSpan.hi > maxP) {
                        maxP = pSpan.hi;
                    }
                }
            }
        }
        m_spanCache[node] = anySet ? IntervalOpt::makeSet(minP, maxP) : IntervalOpt::makeEmpty();

        // ---- Compute anc_cov_cache[node] = union of parents' span (with +1 on max) ----
        // Empty (Python None) if node has no parents (root).
        if (parents.empty()) {
            m_ancCovCache[node] = IntervalOpt::makeEmpty();
        } else {
            BpPosition ancMin = NO_POS_SENTINEL;
            BpPosition ancMax = 0;
            bool anyAnc = false;
            for (NodeID parent : parents) {
                const IntervalOpt& pSpan = m_spanCache[parent];
                if (pSpan.isSet()) {
                    if (!anyAnc) {
                        ancMin = pSpan.lo;
                        ancMax = pSpan.hi;
                        anyAnc = true;
                    } else {
                        if (pSpan.lo < ancMin) {
                            ancMin = pSpan.lo;
                        }
                        if (pSpan.hi > ancMax) {
                            ancMax = pSpan.hi;
                        }
                    }
                }
            }
            m_ancCovCache[node] = anyAnc ? IntervalOpt::makeSet(ancMin, ancMax + 1) : IntervalOpt::makeEmpty();
        }

        // ---- Release children whose final parent just finished ----
        NodeIDList children = m_grg->getDownEdges(node);
        for (NodeID child : children) {
            inDegree[child]--;
            if (inDegree[child] == 0) {
                queue.push_back(child);
            }
        }
    }

    // Pass 3: flatten per-node mutation lists into the CSR.
    m_inputMutOffsets.assign(n + 1, 0);
    for (NodeID node = 0; node < n; node++) {
        m_inputMutOffsets[node + 1] = m_inputMutOffsets[node] + posTmp[node].size();
    }
    m_inputMutIds.resize(m_inputMutOffsets[n]);
    m_inputPos.resize(m_inputMutOffsets[n]);
    for (NodeID node = 0; node < n; node++) {
        const size_t off = m_inputMutOffsets[node];
        const std::vector<MutationId>& mids = mutIdsTmp[node];
        const std::vector<BpPosition>& pos = posTmp[node];
        for (size_t i = 0; i < mids.size(); i++) {
            m_inputMutIds[off + i] = mids[i];
            m_inputPos[off + i] = pos[i];
        }
    }

    if (m_instrument) {
        m_stats.initCachesTime += elapsedSeconds(tStart);
    }
}

// ------------------------------------------------------------------
// Per-node array growth (mirrors Python _grow_node_arrays / _sync_to_grg)
// ------------------------------------------------------------------

void NonDuplicationRecombiner::growNodeArrays(NodeID nodeId) {
    const size_t target = static_cast<size_t>(nodeId) + 1;
    if (m_spanCache.size() < target) {
        m_spanCache.resize(target);   // default-constructed -> Uninit
        m_ancCovCache.resize(target); // default-constructed -> Uninit
    }
    if (m_visitedGen.size() < target) {
        m_visitedGen.resize(target, 0);
        m_connectedGen.resize(target, 0);
    }
}

void NonDuplicationRecombiner::syncToGrg() {
    const size_t n = m_grg->numNodes();
    if (n > 0) {
        growNodeArrays(static_cast<NodeID>(n - 1));
    }
}

// ------------------------------------------------------------------
// Cache access: CSR for input-graph nodes, override map for new/modified
// nodes. Spans are non-owning views; see Span<T> doc for invalidation rules.
// ------------------------------------------------------------------

void NonDuplicationRecombiner::populateMutationOverride(NodeID nodeId) {
    std::vector<MutationId> rawMutIds = m_grg->getMutationsForNode<MutationId>(nodeId, /*allowSort=*/false);
    std::vector<std::pair<MutationId, BpPosition>> mutPairs;
    mutPairs.reserve(rawMutIds.size());
    for (MutationId mid : rawMutIds) {
        const Mutation& mut = m_grg->getMutationById(mid);
        mutPairs.emplace_back(mid, mut.getPosition());
    }
    // p50 mutations/node is 0 across all benchmarked files; skip the sort on
    // the 0/1-item common case to match the Python optimization.
    if (mutPairs.size() > 1) {
        std::sort(mutPairs.begin(),
                  mutPairs.end(),
                  [](const std::pair<MutationId, BpPosition>& a, const std::pair<MutationId, BpPosition>& b) {
                      return a.second < b.second;
                  });
    }
    std::vector<MutationId> mids;
    std::vector<BpPosition> positions;
    mids.reserve(mutPairs.size());
    positions.reserve(mutPairs.size());
    for (const std::pair<MutationId, BpPosition>& mp : mutPairs) {
        mids.push_back(mp.first);
        positions.push_back(mp.second);
    }
    m_overrideMutIds[nodeId] = std::move(mids);
    m_overridePos[nodeId] = std::move(positions);
    if (static_cast<size_t>(nodeId) < m_inputGraphNodeCount) {
        m_dirty[nodeId] |= DIRTY_MUT;
    }
}

void NonDuplicationRecombiner::populateUpEdgesOverride(NodeID nodeId) {
    NodeIDList parents = m_grg->getUpEdges(nodeId);
    m_overrideUpEdges[nodeId] = std::vector<NodeID>(parents.begin(), parents.end());
    if (static_cast<size_t>(nodeId) < m_inputGraphNodeCount) {
        m_dirty[nodeId] |= DIRTY_UP;
    }
}

Span<BpPosition> NonDuplicationRecombiner::getPositionsView(NodeID nodeId) {
    if (static_cast<size_t>(nodeId) < m_inputGraphNodeCount) {
        if (!(m_dirty[nodeId] & DIRTY_MUT)) {
            const size_t lo = m_inputMutOffsets[nodeId];
            const size_t hi = m_inputMutOffsets[nodeId + 1];
            return Span<BpPosition>(m_inputPos.data() + lo, hi - lo);
        }
        const std::vector<BpPosition>& vec = m_overridePos.at(nodeId);
        return Span<BpPosition>(vec.data(), vec.size());
    }
    auto it = m_overridePos.find(nodeId);
    if (it != m_overridePos.end()) {
        return Span<BpPosition>(it->second.data(), it->second.size());
    }
    populateMutationOverride(nodeId);
    const std::vector<BpPosition>& vec = m_overridePos.at(nodeId);
    return Span<BpPosition>(vec.data(), vec.size());
}

Span<MutationId> NonDuplicationRecombiner::getMutationIdsView(NodeID nodeId) {
    if (static_cast<size_t>(nodeId) < m_inputGraphNodeCount) {
        if (!(m_dirty[nodeId] & DIRTY_MUT)) {
            const size_t lo = m_inputMutOffsets[nodeId];
            const size_t hi = m_inputMutOffsets[nodeId + 1];
            return Span<MutationId>(m_inputMutIds.data() + lo, hi - lo);
        }
        const std::vector<MutationId>& vec = m_overrideMutIds.at(nodeId);
        return Span<MutationId>(vec.data(), vec.size());
    }
    auto it = m_overrideMutIds.find(nodeId);
    if (it != m_overrideMutIds.end()) {
        return Span<MutationId>(it->second.data(), it->second.size());
    }
    populateMutationOverride(nodeId);
    const std::vector<MutationId>& vec = m_overrideMutIds.at(nodeId);
    return Span<MutationId>(vec.data(), vec.size());
}

Span<NodeID> NonDuplicationRecombiner::getUpEdgesView(NodeID nodeId) {
    if (static_cast<size_t>(nodeId) < m_inputGraphNodeCount) {
        if (!(m_dirty[nodeId] & DIRTY_UP)) {
            const size_t lo = m_inputUpEdgesOffsets[nodeId];
            const size_t hi = m_inputUpEdgesOffsets[nodeId + 1];
            return Span<NodeID>(m_inputUpEdges.data() + lo, hi - lo);
        }
        const std::vector<NodeID>& vec = m_overrideUpEdges.at(nodeId);
        return Span<NodeID>(vec.data(), vec.size());
    }
    auto it = m_overrideUpEdges.find(nodeId);
    if (it != m_overrideUpEdges.end()) {
        return Span<NodeID>(it->second.data(), it->second.size());
    }
    populateUpEdgesOverride(nodeId);
    const std::vector<NodeID>& vec = m_overrideUpEdges.at(nodeId);
    return Span<NodeID>(vec.data(), vec.size());
}

// ------------------------------------------------------------------
// Iterative span / ancestral coverage (lazy DFS)
// ------------------------------------------------------------------

IntervalOpt NonDuplicationRecombiner::getNodeAndAncestorSpan(NodeID nodeId) {
    if (!m_spanCache[nodeId].isUninit()) {
        return m_spanCache[nodeId];
    }

    // Iterative post-order DFS -- mirrors Python's two-phase stack. Each entry
    // is (nid, processed). processed=false means "push children, then re-push
    // self with processed=true"; processed=true means "all parents resolved,
    // compute and store own span_cache".
    //
    // Span lifetimes: positions / up-edges spans are acquired and consumed
    // within a single (nid, processed=true) iteration. Pushing parents for
    // recursion uses up-edges immediately, so no span outlives a potentially-
    // mutating call.
    std::vector<std::pair<NodeID, bool>> stack;
    std::unordered_set<NodeID> scheduled;
    stack.emplace_back(nodeId, false);
    scheduled.insert(nodeId);

    while (!stack.empty()) {
        const std::pair<NodeID, bool> entry = stack.back();
        stack.pop_back();
        const NodeID nid = entry.first;
        const bool processed = entry.second;

        if (processed) {
            BpPosition minP = NO_POS_SENTINEL;
            BpPosition maxP = 0;
            bool anySet = false;
            Span<BpPosition> positions = getPositionsView(nid);
            if (!positions.empty()) {
                minP = positions.front();
                maxP = positions.back();
                anySet = true;
            }
            Span<NodeID> parents = getUpEdgesView(nid);
            for (std::size_t i = 0; i < parents.size(); i++) {
                const NodeID parent = parents[i];
                const IntervalOpt& pSpan = m_spanCache[parent];
                if (pSpan.isSet()) {
                    if (!anySet) {
                        minP = pSpan.lo;
                        maxP = pSpan.hi;
                        anySet = true;
                    } else {
                        if (pSpan.lo < minP) {
                            minP = pSpan.lo;
                        }
                        if (pSpan.hi > maxP) {
                            maxP = pSpan.hi;
                        }
                    }
                }
            }
            m_spanCache[nid] = anySet ? IntervalOpt::makeSet(minP, maxP) : IntervalOpt::makeEmpty();
            scheduled.erase(nid);
        } else {
            stack.emplace_back(nid, true);
            Span<NodeID> parents = getUpEdgesView(nid);
            for (std::size_t i = 0; i < parents.size(); i++) {
                const NodeID parent = parents[i];
                if (m_spanCache[parent].isUninit() && scheduled.find(parent) == scheduled.end()) {
                    stack.emplace_back(parent, false);
                    scheduled.insert(parent);
                }
            }
        }
    }

    return m_spanCache[nodeId];
}

IntervalOpt NonDuplicationRecombiner::getAncestralCoverage(NodeID nodeId) {
    if (!m_ancCovCache[nodeId].isUninit()) {
        return m_ancCovCache[nodeId];
    }

    // Take the parent NodeID list by value before calling getNodeAndAncestorSpan,
    // which may lazy-fetch positions/up-edges for other nodes and rehash the
    // override maps, invalidating any span we hold.
    Span<NodeID> parentsView = getUpEdgesView(nodeId);
    if (parentsView.empty()) {
        m_ancCovCache[nodeId] = IntervalOpt::makeEmpty();
        return m_ancCovCache[nodeId];
    }
    std::vector<NodeID> parents(parentsView.begin(), parentsView.end());

    BpPosition minPos = NO_POS_SENTINEL;
    BpPosition maxPos = 0;
    bool anySet = false;
    for (NodeID parent : parents) {
        IntervalOpt pSpan = getNodeAndAncestorSpan(parent);
        if (pSpan.isSet()) {
            if (!anySet) {
                minPos = pSpan.lo;
                maxPos = pSpan.hi;
                anySet = true;
            } else {
                if (pSpan.lo < minPos) {
                    minPos = pSpan.lo;
                }
                if (pSpan.hi > maxPos) {
                    maxPos = pSpan.hi;
                }
            }
        }
    }
    m_ancCovCache[nodeId] = anySet ? IntervalOpt::makeSet(minPos, maxPos + 1) : IntervalOpt::makeEmpty();
    return m_ancCovCache[nodeId];
}

// ------------------------------------------------------------------
// Bubble extraction
// ------------------------------------------------------------------

NodeID
NonDuplicationRecombiner::extractBubble(NodeID nodeId, const std::vector<MutationId>& relMutIds, NodeID offspringId) {
    std::chrono::steady_clock::time_point t;
    if (m_instrument) {
        t = std::chrono::steady_clock::now();
    }
    const NodeID bubbleId = m_grg->makeNode();
    if (m_instrument) {
        m_stats.makeNodeTime += elapsedSeconds(t);
        m_stats.makeNodeCalls += 1;
    }
    m_audit.makeNodeCalls += 1;

    growNodeArrays(bubbleId);

    if (m_instrument) {
        t = std::chrono::steady_clock::now();
    }
    m_grg->connect(static_cast<SignedNodeID>(bubbleId), static_cast<SignedNodeID>(nodeId));
    m_grg->connect(static_cast<SignedNodeID>(bubbleId), -static_cast<SignedNodeID>(offspringId));
    if (m_instrument) {
        m_stats.connectTime += elapsedSeconds(t);
        m_stats.connectCalls += 2;
    }
    m_audit.connectCallsInExtract += 2;
    m_audit.extractBubbleCalls += 1;

    // node_id just gained a new up-edge (bubbleId is now a parent of node_id).
    // Eagerly refresh the override map so subsequent within-call traversals
    // (recurseAttach pushing parents after bubble extraction) see the new
    // bubble parent. For input nodes this shadows the CSR; for new nodes
    // this overwrites the prior override.
    populateUpEdgesOverride(nodeId);

    BubbleOp op;
    op.nodeId = nodeId;
    op.bubbleId = bubbleId;
    op.relMutIds = relMutIds;
    m_pendingBubbles.push_back(std::move(op));

    // Track only nodes whose own caches actually become stale:
    // node_id loses mutations (mutation/pos/anc_cov are stale) and the
    // new bubble_id needs a clean slot. node_id's parents are NOT
    // invalidated -- their span and anc_cov depend only on themselves
    // and their own ancestors, neither of which changes here.
    m_modifiedNodes.insert(nodeId);
    m_modifiedNodes.insert(bubbleId);

    return bubbleId;
}

// ------------------------------------------------------------------
// Apply deferred work + cache eviction
// ------------------------------------------------------------------

void NonDuplicationRecombiner::applyPendingBubbles() {
    std::chrono::steady_clock::time_point tApplyStart;
    double tGet = 0.0;
    double tAdd = 0.0;
    double tRem = 0.0;
    size_t nMutsTotal = 0;
    const size_t nBubbles = m_pendingBubbles.size();
    if (m_instrument) {
        tApplyStart = std::chrono::steady_clock::now();
    }

    for (const BubbleOp& op : m_pendingBubbles) {
        if (m_instrument) {
            nMutsTotal += op.relMutIds.size();
        }
        for (MutationId mid : op.relMutIds) {
            std::chrono::steady_clock::time_point t;
            if (m_instrument) {
                t = std::chrono::steady_clock::now();
            }
            const Mutation& mut = m_grg->getMutationById(mid);
            if (m_instrument) {
                tGet += elapsedSeconds(t);
                t = std::chrono::steady_clock::now();
            }
            m_grg->addMutation(mut, op.bubbleId);
            if (m_instrument) {
                tAdd += elapsedSeconds(t);
                t = std::chrono::steady_clock::now();
            }
            m_grg->removeMutation(mid, op.nodeId);
            if (m_instrument) {
                tRem += elapsedSeconds(t);
            }
        }
    }
    m_pendingBubbles.clear();

    if (!m_deferSampleUpdates) {
        flushSampleUpdates();
    }

    if (m_instrument) {
        const double elapsed = elapsedSeconds(tApplyStart);
        m_stats.applyBubblesTime += elapsed;
        m_stats.getMutationByIdTime += tGet;
        m_stats.addMutationTime += tAdd;
        m_stats.removeMutationTime += tRem;
        m_stats.getMutationByIdCalls += nMutsTotal;
        m_stats.addMutationCalls += nMutsTotal;
        m_stats.removeMutationCalls += nMutsTotal;
        m_stats.bubblesCreated += nBubbles;
        m_stats.mutationsMoved += nMutsTotal;
    }
}

void NonDuplicationRecombiner::flushSampleUpdates() {
    if (m_pendingSampleRemovals.empty()) {
        return;
    }
    std::chrono::steady_clock::time_point t;
    if (m_instrument) {
        t = std::chrono::steady_clock::now();
    }
    NodeIDList currentList = m_grg->getSampleNodes();
    std::unordered_set<NodeID> current(currentList.begin(), currentList.end());
    for (NodeID rm : m_pendingSampleRemovals) {
        current.erase(rm);
    }
    NodeIDList nextSamples(current.begin(), current.end());
    m_grg->setSamples(nextSamples);
    m_pendingSampleRemovals.clear();
    if (m_instrument) {
        m_stats.flushSamplesTime += elapsedSeconds(t);
    }
}

void NonDuplicationRecombiner::clearModifiedCaches() {
    std::chrono::steady_clock::time_point t;
    if (m_instrument) {
        t = std::chrono::steady_clock::now();
    }
    for (NodeID nodeId : m_modifiedNodes) {
        if (static_cast<size_t>(nodeId) < m_spanCache.size()) {
            m_spanCache[nodeId] = IntervalOpt();   // back to Uninit
            m_ancCovCache[nodeId] = IntervalOpt(); // back to Uninit
        }
        if (static_cast<size_t>(nodeId) < m_inputGraphNodeCount) {
            // Input-graph node: the CSR slice is now stale (applyPendingBubbles
            // moved some of node_id's mutations to the bubble). Eagerly refresh
            // the override map so it shadows the CSR with current GRG state.
            // m_overrideUpEdges was already refreshed eagerly by extractBubble
            // and stays valid until the next bubble extraction touches this
            // node.
            populateMutationOverride(nodeId);
        } else {
            // Bubble / offspring node: clear overrides so the next access
            // lazy-refetches from the GRG (which now reflects the post-
            // applyPendingBubbles state).
            m_overrideMutIds.erase(nodeId);
            m_overridePos.erase(nodeId);
            m_overrideUpEdges.erase(nodeId);
        }
    }
    m_modifiedNodes.clear();
    if (m_instrument) {
        m_stats.clearCachesTime += elapsedSeconds(t);
    }
}

SignedNodeID NonDuplicationRecombiner::registerOffspring(NodeID offspringId) {
    auto it = m_negativeNodeIndex.find(offspringId);
    size_t idx;
    if (it == m_negativeNodeIndex.end()) {
        idx = m_negativeNodeIds.size();
        m_negativeNodeIndex.emplace(offspringId, idx);
        m_negativeNodeIds.push_back(offspringId);
    } else {
        idx = it->second;
    }
    return -static_cast<SignedNodeID>(idx + 1);
}

// ------------------------------------------------------------------
// Public entry points (recurseAttach + recombine* in subsequent tasks)
// ------------------------------------------------------------------

SignedNodeID NonDuplicationRecombiner::recombine(NodeID hapA, NodeID hapB, BpPosition breakpoint) {
    m_audit.recombineCalls++;
    m_pendingBubbles.clear();

    std::chrono::steady_clock::time_point t;
    if (m_instrument) {
        t = std::chrono::steady_clock::now();
    }
    syncToGrg();
    if (m_instrument) {
        m_stats.syncToGrgTime += elapsedSeconds(t);
        t = std::chrono::steady_clock::now();
    }
    const NodeID offspringId = m_grg->makeNode(1, /*negative=*/true);
    if (m_instrument) {
        m_stats.makeNodeTime += elapsedSeconds(t);
        m_stats.makeNodeCalls += 1;
    }
    m_audit.makeNodeCalls++;
    growNodeArrays(static_cast<NodeID>(m_grg->numNodes() - 1));
    m_genConnected++;

    double recurseT = 0.0;
    std::chrono::steady_clock::time_point ts;
    if (m_instrument) {
        ts = std::chrono::steady_clock::now();
    }
    recurseAttach(hapA, offspringId, 0, breakpoint);
    if (m_instrument) {
        recurseT += elapsedSeconds(ts);
        m_stats.recurseAttachCalls += 1;
        m_stats.segmentsProcessed += 1;
        ts = std::chrono::steady_clock::now();
    }
    recurseAttach(hapB, offspringId, breakpoint, m_genomeLength);
    if (m_instrument) {
        recurseT += elapsedSeconds(ts);
        m_stats.recurseAttachCalls += 1;
        m_stats.segmentsProcessed += 1;
        m_stats.recurseAttachTime += recurseT;
    }

    applyPendingBubbles();
    clearModifiedCaches();

    if (m_instrument) {
        m_stats.offspringCount += 1;
    }

    return registerOffspring(offspringId);
}

SignedNodeID NonDuplicationRecombiner::recombineMulti(const std::vector<std::pair<NodeID, BpPosition>>& segments) {
    m_audit.recombineCalls++;
    m_pendingBubbles.clear();

    std::chrono::steady_clock::time_point t;
    if (m_instrument) {
        t = std::chrono::steady_clock::now();
    }
    syncToGrg();
    if (m_instrument) {
        m_stats.syncToGrgTime += elapsedSeconds(t);
        t = std::chrono::steady_clock::now();
    }
    const NodeID offspringId = m_grg->makeNode(1, /*negative=*/true);
    if (m_instrument) {
        m_stats.makeNodeTime += elapsedSeconds(t);
        m_stats.makeNodeCalls += 1;
    }
    m_audit.makeNodeCalls++;
    growNodeArrays(static_cast<NodeID>(m_grg->numNodes() - 1));
    m_genConnected++;

    double recurseT = 0.0;
    BpPosition start = 0;
    for (const auto& seg : segments) {
        const NodeID parentId = seg.first;
        const BpPosition end = seg.second;
        if (end > start) {
            if (m_debug) {
                std::fputs("BREAK\n", stderr);
            }
            std::chrono::steady_clock::time_point ts;
            if (m_instrument) {
                ts = std::chrono::steady_clock::now();
            }
            recurseAttach(parentId, offspringId, start, end);
            if (m_instrument) {
                recurseT += elapsedSeconds(ts);
                m_stats.recurseAttachCalls += 1;
                m_stats.segmentsProcessed += 1;
            }
        }
        start = end;
    }
    if (m_instrument) {
        m_stats.recurseAttachTime += recurseT;
    }

    applyPendingBubbles();
    clearModifiedCaches();

    if (m_instrument) {
        m_stats.offspringCount += 1;
    }

    return registerOffspring(offspringId);
}

// Rebuild m_inputPos / m_inputMutIds / m_inputMutOffsets from the GRG's
// current state. Called at endGeneration after sortMutations renumbers
// MutationIds; positions are unchanged by sortMutations but a node's mutation
// count may have shrunk (mutations relocated to bubbles by applyPendingBubbles).
// Up-edges CSR is unaffected -- topology doesn't change in sortMutations.
void NonDuplicationRecombiner::rebuildInputMutationCSR() {
    const size_t n = m_inputGraphNodeCount;
    if (n == 0) {
        return;
    }

    // Pass 1: build per-node sorted mutation pairs into temporaries; count sizes.
    std::vector<std::vector<MutationId>> mutIdsTmp(n);
    std::vector<std::vector<BpPosition>> posTmp(n);
    m_inputMutOffsets.assign(n + 1, 0);
    for (NodeID node = 0; node < n; node++) {
        std::vector<MutationId> rawMutIds = m_grg->getMutationsForNode<MutationId>(node, /*allowSort=*/false);
        std::vector<std::pair<MutationId, BpPosition>> mutPairs;
        mutPairs.reserve(rawMutIds.size());
        for (MutationId mid : rawMutIds) {
            const Mutation& mut = m_grg->getMutationById(mid);
            mutPairs.emplace_back(mid, mut.getPosition());
        }
        if (mutPairs.size() > 1) {
            std::sort(mutPairs.begin(),
                      mutPairs.end(),
                      [](const std::pair<MutationId, BpPosition>& a, const std::pair<MutationId, BpPosition>& b) {
                          return a.second < b.second;
                      });
        }
        mutIdsTmp[node].reserve(mutPairs.size());
        posTmp[node].reserve(mutPairs.size());
        for (const std::pair<MutationId, BpPosition>& mp : mutPairs) {
            mutIdsTmp[node].push_back(mp.first);
            posTmp[node].push_back(mp.second);
        }
        m_inputMutOffsets[node + 1] = m_inputMutOffsets[node] + mutPairs.size();
    }

    // Pass 2: flatten into CSR.
    m_inputMutIds.assign(m_inputMutOffsets[n], 0);
    m_inputPos.assign(m_inputMutOffsets[n], 0);
    for (NodeID node = 0; node < n; node++) {
        const size_t off = m_inputMutOffsets[node];
        const std::vector<MutationId>& mids = mutIdsTmp[node];
        const std::vector<BpPosition>& pos = posTmp[node];
        for (size_t i = 0; i < mids.size(); i++) {
            m_inputMutIds[off + i] = mids[i];
            m_inputPos[off + i] = pos[i];
        }
    }
}

void NonDuplicationRecombiner::endGeneration() {
    std::chrono::steady_clock::time_point t;
    if (m_instrument) {
        t = std::chrono::steady_clock::now();
    }
    m_grg->sortMutations();
    if (m_instrument) {
        m_stats.sortMutationsTime += elapsedSeconds(t);
    }
    // sortMutations renumbers MutationIds (by (position, allele)), so any
    // cached MutationIds are stale. Rebuild the input CSR fully (cheap;
    // O(V + E) once per generation). Clear all mutation overrides since
    // they also reference stale MutationIds.
    // m_overrideUpEdges survives -- up-edges are unaffected by sortMutations
    // and the bubble-parent records accumulated across recombines remain
    // valid.
    rebuildInputMutationCSR();
    for (const auto& entry : m_overrideMutIds) {
        if (static_cast<size_t>(entry.first) < m_inputGraphNodeCount) {
            m_dirty[entry.first] &= ~DIRTY_MUT;
        }
    }
    m_overrideMutIds.clear();
    m_overridePos.clear();
}

void NonDuplicationRecombiner::clearPendingSampleRemovals() { m_pendingSampleRemovals.clear(); }

// ------------------------------------------------------------------
// Iterative recurseAttach (hot path) -- direct port of Python _recurse_attach.
//
// Every visit past both skip guards lands in exactly one of 13 audit buckets
// (9 standard + 4 root variants). The audit_check() invariants (see
// AuditCounters in the header) verify the implementation matches the
// algorithm spec by reconciling the per-case counts against connect /
// extract_bubble call totals.
// ------------------------------------------------------------------

void NonDuplicationRecombiner::recurseAttach(NodeID rootId,
                                             NodeID offspringId,
                                             BpPosition leftBound,
                                             BpPosition rightBound) {
    m_genVisited++;
    const uint64_t genV = m_genVisited;
    const uint64_t genC = m_genConnected;

    m_audit.recurseAttachCalls++;

    const SignedNodeID negOffspring = -static_cast<SignedNodeID>(offspringId);

    // Stack entries: (nodeId, L, R). Stored as separate vectors would be
    // marginally faster but a single vector-of-struct keeps the port direct.
    struct Frame {
        NodeID nodeId;
        BpPosition L;
        BpPosition R;
    };
    std::vector<Frame> stack;
    stack.push_back({rootId, leftBound, rightBound});

    while (!stack.empty()) {
        const Frame frame = stack.back();
        stack.pop_back();
        const NodeID nodeId = frame.nodeId;
        const BpPosition L = frame.L;
        const BpPosition R = frame.R;

        if (L >= R) {
            m_audit.skipEmptyInterval++;
            continue;
        }
        if (m_visitedGen[nodeId] == genV) {
            m_audit.skipAlreadyVisited++;
            continue;
        }
        m_visitedGen[nodeId] = genV;
        m_audit.visits++;
        if (m_instrument) {
            m_stats.visitsTotal++;
        }

        // Positions span: CSR slice for input nodes, override-map view for new
        // / modified ones (lazy-populated on first miss). Span lifetime: only
        // used here for the two lower_bound calls; subsequent steps re-acquire
        // spans as needed (getAncestralCoverage may mutate override maps).
        Span<BpPosition> positions = getPositionsView(nodeId);
        size_t left = 0;
        size_t right = 0;
        if (!positions.empty()) {
            left = static_cast<size_t>(std::lower_bound(positions.begin(), positions.end(), L) - positions.begin());
            right = static_cast<size_t>(std::lower_bound(positions.begin(), positions.end(), R) - positions.begin());
        }
        const size_t numRel = right - left;
        const size_t numAll = positions.size();

        const bool hasAllRelevant = numAll > 0 && numRel == numAll;
        const bool hasNoRelevant = numRel == 0;
        const bool hasPartialRelevant = numRel > 0 && numRel < numAll;

        // Resolve ancestral coverage (anc_cov_cache). Uninit -> lazy DFS.
        // Empty -> root (Python None). Set -> (Iu0, Iu1).
        IntervalOpt Iu = m_ancCovCache[nodeId];
        if (Iu.isUninit()) {
            Iu = getAncestralCoverage(nodeId);
        }

        if (Iu.isEmpty()) {
            // Root: only own muts matter. Four mutually-exclusive sub-cases.
            if (hasAllRelevant) {
                if (m_connectedGen[nodeId] != genC) {
                    m_grg->connect(static_cast<SignedNodeID>(nodeId), negOffspring);
                    m_connectedGen[nodeId] = genC;
                    m_pendingSampleRemovals.insert(nodeId);
                    m_audit.directAttachRoot++;
                    m_audit.connectCallsInAttach++;
                } else {
                    m_audit.directAttachDup++;
                }
            } else if (hasPartialRelevant) {
                Span<MutationId> mutIds = getMutationIdsView(nodeId);
                std::vector<MutationId> relMutIds(mutIds.begin() + left, mutIds.begin() + right);
                const NodeID bubbleId = extractBubble(nodeId, relMutIds, offspringId);
                m_connectedGen[bubbleId] = genC;
                m_audit.bubbleStripPartialRt++;
            } else {
                m_audit.pruningRoot++;
            }
            continue;
        }

        // Non-root: classify by relevant-mutation status x interval overlap.
        const BpPosition Iu0 = Iu.lo;
        const BpPosition Iu1 = Iu.hi;
        const bool ancestralDisjoint = R <= Iu0 || L >= Iu1;
        const bool fullCoverage = Iu0 >= L && Iu1 <= R;

        if (hasAllRelevant && fullCoverage) {
            if (m_connectedGen[nodeId] != genC) {
                m_grg->connect(static_cast<SignedNodeID>(nodeId), negOffspring);
                m_connectedGen[nodeId] = genC;
                m_pendingSampleRemovals.insert(nodeId);
                m_audit.directAttach++;
                m_audit.connectCallsInAttach++;
            } else {
                m_audit.directAttachDup++;
            }
            continue;
        }

        if (hasNoRelevant && ancestralDisjoint) {
            m_audit.pruning++;
            continue;
        }

        if (hasNoRelevant) {
            Span<NodeID> parents = getUpEdgesView(nodeId);
            // Early-attach optimization: multi-parent empty intermediary
            // fully covering [L, R) attaches directly instead of walking up.
            // Gated on num_all==0 + full_coverage + fanout>1; see Python
            // comments above the corresponding block for the correctness
            // argument (multitree preservation, etc.).
            if (numAll == 0 && fullCoverage && parents.size() > 1) {
                if (m_connectedGen[nodeId] != genC) {
                    m_grg->connect(static_cast<SignedNodeID>(nodeId), negOffspring);
                    m_connectedGen[nodeId] = genC;
                    m_pendingSampleRemovals.insert(nodeId);
                    m_audit.pathCompressionAttach++;
                    m_audit.connectCallsInAttach++;
                } else {
                    m_audit.directAttachDup++;
                }
                continue;
            }
            const BpPosition newL = (L > Iu0) ? L : Iu0;
            const BpPosition newR = (R < Iu1) ? R : Iu1;
            if (fullCoverage) {
                m_audit.pathCompression++;
            } else {
                m_audit.decomposition++;
            }
            if (newL >= newR) {
                m_audit.skipEmptyTrim++;
                continue;
            }
            // Skip parents already attached to this offspring (typically
            // bubbles from earlier segments). Iterate in reverse so the
            // first parent is processed first off the stack.
            //
            // Pre-prune (gated on m_prePruneEnabled): if the parent is an
            // input-graph node with zero mutations (CSR slice empty AND
            // mut-override clean) AND its ancCov is disjoint from
            // [newL, newR), the visit would hit `pruning` and do nothing
            // useful. Skip the push entirely. The check is conservative —
            // any uncertainty (dirty bit, Uninit/Empty ancCov, bubble or
            // offspring node) falls through to normal visit.
            for (std::size_t i = parents.size(); i-- > 0;) {
                const NodeID parent = parents[i];
                if (m_connectedGen[parent] != genC) {
                    if (m_prePruneEnabled
                        && static_cast<size_t>(parent) < m_inputGraphNodeCount
                        && !(m_dirty[parent] & DIRTY_MUT)
                        && m_inputMutOffsets[parent] == m_inputMutOffsets[parent + 1]) {
                        const IntervalOpt& cov = m_ancCovCache[parent];
                        if (cov.isSet() && (newR <= cov.lo || newL >= cov.hi)) {
                            if (m_instrument) {
                                m_stats.prePrunedSkips++;
                            }
                            continue;
                        }
                    }
                    stack.push_back({parent, newL, newR});
                }
            }
            continue;
        }

        // Partial / all relevant, not full coverage -> bubble + maybe recurse.
        {
            Span<MutationId> mutIds = getMutationIdsView(nodeId);
            std::vector<MutationId> relMutIds(mutIds.begin() + left, mutIds.begin() + right);
            const NodeID bubbleId = extractBubble(nodeId, relMutIds, offspringId);
            m_connectedGen[bubbleId] = genC;
        }

        // Classify exactly one of the 5 non-root bubble cells.
        if (hasAllRelevant) {
            if (ancestralDisjoint) {
                m_audit.bubbleStrip++;
            } else {
                m_audit.bubbleSplit++;
            }
        } else {
            // hasPartialRelevant
            if (fullCoverage) {
                m_audit.bubbleFill++;
            } else if (ancestralDisjoint) {
                m_audit.bubbleStripPartial++;
            } else {
                m_audit.bubbleSplitPartial++;
            }
        }

        if (!ancestralDisjoint) {
            const BpPosition newL = (L > Iu0) ? L : Iu0;
            const BpPosition newR = (R < Iu1) ? R : Iu1;
            if (newL >= newR) {
                m_audit.skipEmptyTrim++;
                continue;
            }
            // extractBubble refreshed m_overrideUpEdges[nodeId]; re-fetch the
            // up-edges view to see the new bubble parent (skipped below via
            // the connectedGen guard since extractBubble's caller marked it).
            // Pre-prune mirrors the path-compression / decomposition branch;
            // see comment there for the correctness argument.
            Span<NodeID> parents = getUpEdgesView(nodeId);
            for (std::size_t i = parents.size(); i-- > 0;) {
                const NodeID parent = parents[i];
                if (m_connectedGen[parent] != genC) {
                    if (m_prePruneEnabled
                        && static_cast<size_t>(parent) < m_inputGraphNodeCount
                        && !(m_dirty[parent] & DIRTY_MUT)
                        && m_inputMutOffsets[parent] == m_inputMutOffsets[parent + 1]) {
                        const IntervalOpt& cov = m_ancCovCache[parent];
                        if (cov.isSet() && (newR <= cov.lo || newL >= cov.hi)) {
                            if (m_instrument) {
                                m_stats.prePrunedSkips++;
                            }
                            continue;
                        }
                    }
                    stack.push_back({parent, newL, newR});
                }
            }
        }
    }
}

} // namespace grgl
