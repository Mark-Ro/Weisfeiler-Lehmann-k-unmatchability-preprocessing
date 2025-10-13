import time
from collections import deque, Counter
from .utils import update_feature_string as update_feature_string_py, compute_distances as compute_distances_py
from .compliance import (partition_from_colors as partition_from_colors_py,
                        build_color_counts_and_members as build_color_counts_and_members_py,
                        check_k_wl_compliance as check_k_wl_compliance_py)
from .hash import fast_hash as fast_hash_py
import struct


USING_CYTHON = False # Backend state (also used by workers)

"""
adj_v: sequence of neighbor triples. Each item is a triple (dir_tag:int, rel_id:int, nb:int), where dir_tag is 0 for incoming and 1 for outgoing edges, rel_id is a compact integer ID of the relation, and nb is the neighbor node index. Example: [(1, 12, 5), (0, 7, 3), ...].
color: list of ints. The current node colors, one 64-bit hash per node. Example: [12345, 67890, ...].
self_color: int. The current color (hash) of the node whose neighbors are adj_v.

Generates a refined hashed color for a node by combining its own color with the (direction, relation, neighbor-color) multiset
in a canonical (sorted) order. The serialization is explicit little-endian uint64 to avoid ambiguity and to match the Cython backend.
"""
def _refine_node_color_py(adj_v, color, self_color):
    m = len(adj_v)  # Counts how many neighbor triples the node has

    # If there are no neighbors, hash only the node's own color
    if m == 0:
        buf = struct.pack('<Q', self_color)  # Pack self_color as little-endian uint64 (binary, unambiguous)
        return fast_hash(buf)  # Hash the canonical 8-byte buffer

    # Build canonicalized triples (dir_tag, rel_id, neighbor_color)
    triples = [(t[0], t[1], color[t[2]]) for t in adj_v]  # Builds the list of (dir_tag, rel_id, neighbor_color)
    triples.sort()  # Sorts triples by (dir_tag, rel_id, neighbor_color) to enforce a canonical representation

    # Pack self_color followed by all triples as little-endian uint64 words: [self, d, r, c]*
    parts = [struct.pack('<Q', self_color)]  # Start with self color (uint64 LE)
    for d, r, c in triples:
        parts.append(struct.pack('<QQQ', d, r, c))  # Append each sorted triple as (uint64, uint64, uint64) LE
    buf = b''.join(parts)  # Join into a single bytes object
    return fast_hash(buf)  # Hash canonical byte buffer to obtain the refined color



"""
n: int. Number of nodes in the graph.
X_V: list of dicts. Per-node features of length n. Each dict contains:
  - "t": int. Node tag code (0 for blank, 1 for constant).
  - "c": list of int. Sorted list of concept IDs.
  - "r": list of (int,int,int). Sorted list of per-relation degree triples (rel_rank, outdeg, indeg).
  - "f": bytes. Binary feature buffer (uint64 little-endian) built by update_feature_string.
start_time: float or None. Epoch seconds marking when processing started; used with max_seconds to enforce a time budget.
max_seconds: float or None. Time budget in seconds; processing aborts if time.time() - start_time > max_seconds.

Produces the initial coloring by hashing each node’s binary feature buffer.
"""
def wl_initial_coloring_py(n, X_V, start_time, max_seconds, verbose):
    if start_time and max_seconds and (time.time() - start_time > max_seconds):
        if verbose:
            print("\nTimeout during initial coloring.")
        return None
    return [fast_hash(X_V[i]['f']) for i in range(n)]  # Hashes each node's feature buffer to produce initial colors


"""
n: int. Number of nodes.
adj: tuple[tuple[...]]. Compact adjacency. For each node v, adj[v] is a sequence of (dir_tag:int, rel_id:int, nb:int).
color: list[int]. Current node colors (64-bit hashes).
color_counts: dict[int,int]. Color → frequency map; updated in place.
start_time: float | None. Start timestamp for time budgeting.
max_seconds: float | None. Time budget in seconds.
verbose: bool. If True, print timeout messages.

Iteratively refine node colors (Weisfeiler–Lehman) until the induced partition stabilizes. Update color_counts at each iteration and return the converged colors.
"""
def wl_coloring_py(n, adj, color, color_counts, start_time, max_seconds, verbose):

    color_counts.clear() # Resets the color-frequency dictionary in place
    color_counts.update(Counter(color)) # Recomputes frequencies for earch color

    prev_partition = None
    current_partition = partition_from_colors_py(color) # Computes the partition induced by the current colors

    while prev_partition != current_partition: # Iterates until the partition reaches a fixed point

        # Aborts early if the time budget has been exceeded
        if time.time() - start_time > max_seconds:
            if verbose:
                print("\nTimeout during WL-coloring.")
            return color

        prev_partition = current_partition # Remembers the current partition
        new_color = [0] * n # Allocates the array for refined colors
        new_counts = {} # Prepares fresh counts for the refined colors

        for v in range(n): # Processes each node to refine its color

            # Aborts early if the time budget has been exceeded
            if time.time() - start_time > max_seconds:
                if verbose:
                    print("\nTimeout during WL-coloring.")
                return color

            h = _refine_node_color_py(adj[v], color, color[v]) # Computes the refined color from neighbors and self
            new_color[v] = h # Stores the refined color
            new_counts[h] = new_counts.get(h, 0) + 1 # Updates the frequency for the refined color

        current_partition = partition_from_colors_py(new_color) # Recomputes the partition from the refined colors
        color = new_color # Replaces the working coloring with the refined one
        color_counts.clear() # Resets the color-frequency dictionary in place
        color_counts.update(new_counts) # Replaces it with the newly computed frequencies

    return color

"""
n: int. Number of nodes.
adj: tuple[tuple[...]]. Compact adjacency; see above.
X_V: list[dict]. Per-node numeric features with fields:
    "t": int, "c": list[int], "r": list[tuple[int,int,int]], "f": bytes (binary buffer).
changed_idx: int. Index of the node whose features ("t") changed and from which recoloring should propagate.
prev_color: list[int]. Coloring before the change.
color_counts: dict[int,int]. Color frequencies; updated in place as colors change.
start_time: float | None. Start timestamp for time budgeting.
max_seconds: float | None. Time budget in seconds.
verbose: bool. If True, print timeout messages.
distance_limit: int | None. If set, only nodes with graph distance ≤ distance_limit from changed_idx are reevaluated.

Propagate a local feature change (rebuild X_V[changed_idx]["f"] and re-hash it) through the graph, incrementally updating colors (optionally bounded by distance_limit) and maintaining color frequencies. Return the updated coloring.
"""
def wl_coloring_incremental_py(n, adj, X_V, changed_idx, prev_color, color_counts, start_time, max_seconds, verbose, distance_limit=None):
    # Aborts early if the time budget has been exceeded
    if time.time() - start_time > max_seconds:
        if verbose:
            print("\nTimeout before incremental WL.")
        return prev_color

    color = prev_color.copy()  # Works on a copy of the previous colors
    # Store (node, distance-from-changed) directly in the queue to avoid a separate BFS
    queue = deque([(changed_idx, 0)])
    visited = set()  # Tracks visited nodes to avoid reprocessing

    update_feature_string(changed_idx, X_V)  # Rebuilds the changed node's binary feature buffer "f" from numeric fields
    old_color = color[changed_idx]  # Reads the old color of the changed node
    new_color_val = fast_hash(X_V[changed_idx]["f"])  # Computes the new color by hashing the updated binary buffer
    if old_color != new_color_val:  # Checks whether the node's color actually changed
        color_counts[old_color] -= 1  # Decrements the frequency of the old color
        color_counts[new_color_val] = color_counts.get(new_color_val, 0) + 1  # Increments the frequency of the new color
        color[changed_idx] = new_color_val  # Writes back the new color

    while queue:  # Processes nodes in BFS order
        # Aborts early if the time budget has been exceeded
        if time.time() - start_time > max_seconds:
            if verbose:
                print("\nTimeout during incremental WL propagation.")
            return color

        v, d = queue.popleft()  # Pops the next node and its distance
        if v in visited:
            continue
        visited.add(v)  # Marks it as visited

        # Depth-based pruning without a precomputed distance array
        if distance_limit is not None and d > distance_limit:
            continue

        new_c = _refine_node_color_py(adj[v], color, color[v])  # Recomputes the node's color based on current neighbors

        # If the color changes, propagates to neighbors
        if new_c != color[v]:
            old_c = color[v]
            color[v] = new_c
            color_counts[old_c] -= 1  # Updates the node's color
            color_counts[new_c] = color_counts.get(new_c, 0) + 1  # Increments the frequency of the new color

            # Enqueue neighbors only if we haven't exceeded the distance limit
            if distance_limit is None or d < distance_limit:
                for _, _, nb in adj[v]:
                    if nb not in visited:
                        queue.append((nb, d + 1))

    return color


"""
use_cython: bool. If True, use the compiled Cython backend; otherwise use the pure-Python backend.
verbose: bool. If True, print which backend is selected.

Select the WL backend (Cython if available and requested, otherwise pure Python). Bind module-level function references:
  - wl_initial_coloring, wl_coloring, wl_coloring_incremental
  - partition_from_colors, build_color_counts_and_members, check_k_wl_compliance
  - compute_distances, update_feature_string, fast_hash
and set USING_CYTHON accordingly.
"""
def init_wl_backend(use_cython, verbose):
    global wl_initial_coloring, wl_coloring, wl_coloring_incremental
    global partition_from_colors, build_color_counts_and_members
    global check_k_wl_compliance, compute_distances, USING_CYTHON
    global update_feature_string, fast_hash

    USING_CYTHON = False  # Default: Python puro
    if use_cython:
        try:
            from .cyext import cy_wl
            wl_initial_coloring = cy_wl.wl_initial_coloring_cy
            wl_coloring = cy_wl.wl_coloring_cy
            wl_coloring_incremental = cy_wl.wl_coloring_incremental_cy
            partition_from_colors = cy_wl.partition_from_colors_cy
            build_color_counts_and_members = cy_wl.build_color_counts_and_members_cy
            check_k_wl_compliance = cy_wl.check_k_wl_compliance_cy
            compute_distances = cy_wl.compute_distances_cy
            update_feature_string = cy_wl.update_feature_string_cy
            fast_hash = cy_wl.fast_hash_cy
            USING_CYTHON = True
            if verbose:
                import inspect
                try:
                    src = inspect.getsourcefile(cy_wl) or str(cy_wl.__file__)
                except Exception:
                    src = str(getattr(cy_wl, "__file__", "unknown"))
                print(f"[WL Backend] Using Cython implementation ({src})")
            return
        except ImportError as e:
            raise RuntimeError ("Cython backend requested but 'wlpp.cyext.cy_wl' could not be imported.") from e

    # Fallback Python puro
    wl_initial_coloring = wl_initial_coloring_py
    wl_coloring = wl_coloring_py
    wl_coloring_incremental = wl_coloring_incremental_py
    partition_from_colors = partition_from_colors_py
    build_color_counts_and_members = build_color_counts_and_members_py
    check_k_wl_compliance = check_k_wl_compliance_py
    compute_distances = compute_distances_py
    update_feature_string = update_feature_string_py
    fast_hash = fast_hash_py
    USING_CYTHON = False
    if verbose:
        print("[WL Backend] Using pure Python implementation")


wl_initial_coloring = wl_initial_coloring_py
wl_coloring = wl_coloring_py
wl_coloring_incremental = wl_coloring_incremental_py
partition_from_colors = partition_from_colors_py
build_color_counts_and_members = build_color_counts_and_members_py
check_k_wl_compliance = check_k_wl_compliance_py
compute_distances = compute_distances_py
update_feature_string = update_feature_string_py
fast_hash = fast_hash_py

