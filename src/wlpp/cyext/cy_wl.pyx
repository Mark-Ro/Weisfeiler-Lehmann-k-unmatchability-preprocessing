# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: embedsignature=False
# cython: infer_types=True

cimport cython
import time
import xxhash
from collections import deque
from libc.stdint cimport uint64_t  # Fixed-width integers for hashing/buffers
from libc.stdlib cimport qsort, malloc, free  # C stdlib for sorting and memory management

ctypedef struct triple_t:  # Packed triple describing a neighbor
    uint64_t dir_tag   # 0 for incoming, 1 for outgoing
    uint64_t rel_id    # Compact integer relation ID
    uint64_t col       # Neighbor color (64-bit)

# qsort-compatible comparator (noexcept nogil)
cdef int _cmp_triple(const void* a, const void* b) noexcept nogil:
    cdef triple_t* ta = <triple_t*> a  # Casts to triple_t*
    cdef triple_t* tb = <triple_t*> b
    # When accessing a C pointer in Cython, dereference with [0] to read the struct fields
    # Order by dir_tag
    if ta[0].dir_tag < tb[0].dir_tag: return -1
    if ta[0].dir_tag > tb[0].dir_tag: return 1
    # Then by rel_id
    if ta[0].rel_id  < tb[0].rel_id:  return -1
    if ta[0].rel_id  > tb[0].rel_id:  return 1
    # Finally by color
    if ta[0].col     < tb[0].col:     return -1
    if ta[0].col     > tb[0].col:     return 1
    return 0  # Equal if all fields match


"""
color: sequence[object]. Current node colors; one per node (typically 64-bit hashes).

Build a canonical partition of node indices by color and return a tuple of sorted groups (suitable for fast equality checks).
"""
def partition_from_colors_cy(color):
    groups = {}  # Maps color -> list of node indices
    cdef int n = len(color)  # Number of nodes
    cdef int i
    cdef object c
    for i in range(n):  # Iterate node indices
        c = color[i]  # Read color at i
        if c in groups:
            groups[c].append(i)  # Append index to existing group
        else:
            groups[c] = [i]  # Create new group for this color
    return tuple(sorted(tuple(sorted(g)) for g in groups.values()))  # Canonical sort

"""
color: list[object]. Current node colors; one value per node (typically 64-bit hashes).

Build both (1) the frequency of each color and (2) the membership sets of node indices per color. Return (color_counts, color_members).
"""
def build_color_counts_and_members_cy(color):
    color_counts = {}  # Frequency map color -> count
    color_members = {}  # Membership map color -> set of indices
    cdef int i, n = len(color)
    cdef object c
    for i in range(n):  # Iterate nodes
        c = color[i]  # Read color
        color_counts[c] = color_counts.get(c, 0) + 1  # Increment frequency
        s = color_members.get(c)  # Get/create membership set
        if s is None:
            s = set()
            color_members[c] = s
        s.add(i)  # Add node index
    return color_counts, color_members  # Return both maps

"""
color: sequence[object]. Node colors per index.
color_counts: dict[object,int]. Color → class size.
subjects_idx: Iterable[int]. Indices of protected subject nodes.
k: int. Minimum required class size.

Return True iff every subject’s color class has size ≥ k.
"""
def check_k_wl_compliance_cy(color, color_counts, subjects_idx, int k):
    for i in subjects_idx:  # Iterate subject indices
        if color_counts[color[i]] < k:  # Class smaller than k?
            return False  # Fail on first violation
    return True  # All subjects meet threshold

"""
idx: int. Node index whose binary feature buffer should be (re)built.
X_V: list[dict]. Per-node features; for each i, X_V[i] contains:
    "t": int — node type code (0 = blank, 1 = constant),
    "c": list[int] — sorted concept IDs,
    "r": list[tuple[int,int,int]] — sorted per-relation (rel_rank, out, in),
    "f": bytes — binary buffer (this function overwrites it).

Rebuild X_V[idx]["f"] as a contiguous uint64 little-endian buffer:
  header [t, len(c), len(r)], then all concept IDs, then (rel_rank,out,in) triples.
Matches the Python backend layout for identical hashing.
"""
def update_feature_string_cy(int idx, X_V):
    cdef unsigned long long t_code = <unsigned long long> X_V[idx]["t"]  # Read numeric type
    cdef list c_ids = X_V[idx]["c"]                                     # Sorted concept IDs
    cdef list per_rel = X_V[idx]["r"]                                   # Sorted (rel_rank,out,in) triples

    cdef Py_ssize_t nc = len(c_ids)
    cdef Py_ssize_t nr = len(per_rel)

    # Allocate contiguous uint64 buffer: [t_code, nc, nr, c_ids..., (rel_rank,out,inn)*]
    cdef Py_ssize_t n64 = 3 + nc + 3 * nr
    cdef uint64_t* buf = <uint64_t*> malloc(n64 * sizeof(uint64_t))
    if buf == NULL:
        raise MemoryError()

    cdef Py_ssize_t i
    cdef tuple t
    try:
        buf[0] = t_code                             # Header t
        buf[1] = <unsigned long long> nc           # Header |c|
        buf[2] = <unsigned long long> nr           # Header |r|

        # Write concept IDs
        for i in range(nc):
            buf[3 + i] = <unsigned long long> c_ids[i]

        # Write per-relation triples (rel_rank, out, inn)
        for i in range(nr):
            t = <tuple> per_rel[i]
            buf[3 + nc + 3*i]     = <unsigned long long> t[0]
            buf[3 + nc + 3*i + 1] = <unsigned long long> t[1]
            buf[3 + nc + 3*i + 2] = <unsigned long long> t[2]

        # Export as Python bytes (little-endian memory view)
        X_V[idx]["f"] = (<char*> buf)[: n64 * sizeof(uint64_t)]
    finally:
        free(buf)

def fast_hash_cy(s):
    return xxhash.xxh3_64_intdigest(s)

"""
n: int. Number of nodes.
X_V: list[dict]. Per-node features with "f" already built as a little-endian binary buffer (bytes).
start_time: float or None. Start timestamp for budget checks.
max_seconds: float or None. Time budget in seconds.

Return the initial WL colors by hashing X_V[i]["f"] for all nodes.
"""
def wl_initial_coloring_cy(int n, X_V, start_time, max_seconds, verbose):
    # Early abort on time budget exceeded
    if time.time() - start_time > max_seconds:
        if verbose:
            print("\nTimeout during initial coloring.")
        return None
    cdef list out = [0] * n  # Preallocate list of colors
    cdef int i
    for i in range(n):  # Iterate nodes
        out[i] = fast_hash_cy(X_V[i]['f'])  # Hash each feature buffer (bytes)
    return out  # Return initial colors



"""
adj_v: sequence. The node’s neighbors as triples (dir_tag:int, rel_id:int, nb:int).
color: sequence[int]. Current node colors, indexable by node id.
self_color: unsigned long long. Node’s current color.

Compute a refined color by sorting the (dir_tag, rel_id, neighbor_color) triples, building a uint64 buffer [self_color, triples...], and hashing it (xxh3_64).
"""
@cython.boundscheck(False)
@cython.wraparound(False)
cdef unsigned long long _refine_node_color_cy(object adj_v, object color, unsigned long long self_color):
    cdef Py_ssize_t m = len(adj_v)  # Number of neighbor triples
    cdef triple_t* A
    cdef Py_ssize_t i
    cdef object t
    cdef Py_ssize_t n64
    cdef uint64_t* buf
    cdef bytes b
    cdef unsigned long long h
    cdef uint64_t sc

    if m == 0:
        sc = <uint64_t> self_color  # Copy self color into 64-bit scalar
        b = (<char*> &sc)[: sizeof(uint64_t)]  # View it as bytes
        return fast_hash_cy(b)  # Hash only self color

    A = <triple_t*> malloc(m * sizeof(triple_t))  # Allocate C array for triples
    if A == NULL:
        raise MemoryError()

    try:
        for i in range(m):  # Fill array with canonicalized triples
            t = adj_v[i]
            A[i].dir_tag = <uint64_t> t[0]  # Direction tag
            A[i].rel_id  = <uint64_t> t[1]  # Relation id
            A[i].col     = <uint64_t> color[t[2]]  # Neighbor color by index

        qsort(A, m, sizeof(triple_t), _cmp_triple)  # Sort by (dir_tag, rel_id, color)

        n64 = 1 + 3 * m  # Number of uint64s in buffer
        buf = <uint64_t*> malloc(n64 * sizeof(uint64_t))  # Allocate buffer
        if buf == NULL:
            raise MemoryError()

        buf[0] = <uint64_t> self_color  # First word is self color
        for i in range(m):  # Write sorted triples consecutively
            buf[1 + 3*i]     = A[i].dir_tag
            buf[1 + 3*i + 1] = A[i].rel_id
            buf[1 + 3*i + 2] = A[i].col

        b = (<char*> buf)[: n64 * sizeof(uint64_t)]  # View buffer as bytes
        h = fast_hash_cy(b)  # Hash the canonical byte buffer
        free(buf)  # Free temporary buffer
        return h  # Return refined color
    finally:
        free(A)  # Always free triples array

"""
n: int. Number of nodes.
adj: sequence. For each node v, adj[v] is a sequence of (dir_tag, rel_id, nb).
color: list[int]. Current node colors.
color_counts: dict[int,int]. Color → frequency map; updated in place.
start_time / max_seconds: floats | None. Time budget management.
verbose: bool. If True, print timeout messages.

Iteratively refine node colors (WL) until the partition stabilizes. Update color_counts and return the converged colors.
"""
def wl_coloring_cy(int n, adj, color, color_counts, start_time, max_seconds, verbose):

    if not isinstance(color_counts, dict):
        color_counts = dict(color_counts)  # Ensure a mutable dict

    color_counts.clear()  # Reset color-frequency dictionary in place
    for c in color:
        color_counts[c] = color_counts.get(c, 0) + 1  # Recompute frequencies

    prev_partition = None
    current_partition = partition_from_colors_cy(color)  # Initial partition

    cdef list new_color
    cdef int v
    cdef unsigned long long h

    while prev_partition != current_partition:  # Iterate until convergence
        # Early abort on time budget exceeded
        if time.time() - start_time > max_seconds:
            if verbose:
                print("\nTimeout during WL-coloring.")
            return color

        prev_partition = current_partition  # Remember current partition
        new_color = [0] * n  # Allocate refined colors
        new_counts = {}  # Fresh counts

        for v in range(n):  # Refine each node
            # Early abort on time budget exceeded
            if time.time() - start_time > max_seconds:
                if verbose:  # Print only if verbose is True (consistency with Python backend)
                    print("\nTimeout during WL-coloring.")
                return color
            h = _refine_node_color_cy(adj[v], color, color[v])  # Compute refined color
            new_color[v] = h  # Store refined color
            new_counts[h] = new_counts.get(h, 0) + 1  # Update frequency

        current_partition = partition_from_colors_cy(new_color)  # Rebuild partition
        color = new_color  # Swap refined colors
        color_counts.clear()  # Reset counts
        color_counts.update(new_counts)  # Replace with new counts

    return color  # Converged coloring


"""
n: int. Number of nodes.
adj: sequence. Adjacency as (dir_tag, rel_id, nb) triples for each node.
sources: Iterable[int]. Starting node indices for BFS.
start_time / max_seconds: floats | None. Time budget management.
verbose: bool. If True, print timeout messages.

Compute the shortest-path distance (in edges) from sources to all nodes using BFS; return dist[v], with float('inf') for unreachable nodes.
"""
def compute_distances_cy(int n, adj, sources, start_time, max_seconds, verbose):
    INF = float('inf')  # Sentinel for unreachable nodes
    dist = [INF] * n  # Initialize all distances as infinite
    q = deque()  # BFS queue

    # Seed sources at distance 0
    for s in sources:
        dist[s] = 0
        q.append(s)

    while q:
        # Early abort on time budget exceeded
        if time.time() - start_time > max_seconds:
            if verbose:
                print("\nTimeout during BFS (Cython).")
            return dist

        v = q.popleft()  # Pop next node
        for _, _, nb in adj[v]:  # Iterate neighbors
            # Relax edge if a shorter path is found
            if dist[nb] > dist[v] + 1:
                dist[nb] = dist[v] + 1
                q.append(nb)
    return dist  # Distance array

"""
n: int. Number of nodes.
adj: sequence. Adjacency as (dir_tag, rel_id, nb) triples per node.
X_V: list[dict]. Per-node numeric features with fields "t", "c", "r", "f".
changed_idx: int. Node index whose "t" changed; the change is reflected by rebuilding "f" and re-hashing it.
prev_color: list[int]. Coloring before the change.
color_counts: dict[int,int]. Color frequencies; updated in place.
start_time / max_seconds: floats | None. Time budget management.
verbose: bool. If True, print timeout messages.
distance_limit: int | None. If provided, only process nodes within this distance from changed_idx.

Propagate the local change from changed_idx through the graph, incrementally updating colors (optionally with a distance cap) and maintaining color frequencies. Return the updated colors.
"""
def wl_coloring_incremental_cy(int n, adj, X_V, int changed_idx, prev_color, color_counts, start_time, max_seconds, verbose, distance_limit=None):
    # Early abort on time budget exceeded
    if time.time() - start_time > max_seconds:
        if verbose:
            print("\nTimeout before incremental WL (Cython).")
        return prev_color

    if not isinstance(color_counts, dict):
        color_counts = dict(color_counts)  # Ensure a mutable dict

    color = prev_color.copy()  # Work on a copy of previous colors
    # Store (node, distance-from-changed) directly in the queue to avoid a separate BFS
    q = deque([(changed_idx, 0)])
    visited = set()  # Track visited nodes

    update_feature_string_cy(changed_idx, X_V)  # Rebuild binary "f" from numeric fields
    old_color = color[changed_idx]  # Old color
    new_color_val = fast_hash_cy(X_V[changed_idx]["f"])  # Compute new color from binary buffer
    if old_color != new_color_val:  # If node color changed
        color_counts[old_color] -= 1  # Decrement old color frequency
        color_counts[new_color_val] = color_counts.get(new_color_val, 0) + 1  # Increment new color frequency
        color[changed_idx] = new_color_val  # Write new color

    while q:
        # Early abort on time budget exceeded
        if time.time() - start_time > max_seconds:
            if verbose:
                print("\nTimeout during incremental WL (Cython).")
            return color

        v, d = q.popleft()  # Next node and its depth
        if v in visited:
            continue
        visited.add(v)  # Mark visited

        # Depth-based pruning without a precomputed distance array
        if distance_limit is not None and d > distance_limit:
            continue

        new_c = _refine_node_color_cy(adj[v], color, color[v])  # Recompute node color
        if new_c != color[v]:  # If color changed, propagate to neighbors
            old_c = color[v]
            color[v] = new_c  # Update node color
            color_counts[old_c] -= 1  # Decrement old
            color_counts[new_c] = color_counts.get(new_c, 0) + 1  # Increment new

            # Enqueue neighbors only if we haven't exceeded the distance limit
            if distance_limit is None or d < distance_limit:
                for _, _, nb in adj[v]:
                    if nb not in visited:
                        q.append((nb, d + 1))

    return color  # Updated colors


