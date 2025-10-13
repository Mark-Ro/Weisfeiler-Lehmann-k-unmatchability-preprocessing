import time
from collections import deque
import struct

"""
n: int. Number of nodes in the graph.
adj: tuple of tuples or list of lists. Compact adjacency; for each node v, adj[v] is a sequence of (dir_tag:int, rel_id:int, nb:int), where dir_tag is 0 for incoming and 1 for outgoing edges, rel_id is a compact integer relation ID, and nb is the neighbor node index.
sources: Iterable[int]. Set or sequence of node indices to use as BFS roots.
start_time: float | None. Start timestamp for time budgeting.
max_seconds: float | None. Time budget in seconds.
verbose: bool. If True, print timeout messages.

Compute the shortest-path distance (in edges) from the set of sources to every node using BFS; return a list where unreachable nodes are assigned float('inf').
"""
def compute_distances(n, adj, sources, start_time, max_seconds, verbose):
    dist = [float('inf')] * n # Initializes all distances to infinity (unreachable)
    queue = deque() # Initializes a queue for BFS traversal
    for s in sources:
        dist[s] = 0 # Sets distance of each source to 0
        queue.append(s) # Enqueues each source node
    while queue:
        if time.time() - start_time > max_seconds:
            if verbose:
                print("\nTimeout during BFS distance computation.")
            return dist
        v = queue.popleft() # Dequeues the next node to expand
        for _, _, neighbor in adj[v]: # Iterates over adjacency triples of v (dir_tag, rel_id, nb)
            if dist[neighbor] > dist[v] + 1: # Checks if a shorter path to neighbor has been found
                dist[neighbor] = dist[v] + 1 # Updates neighbor distance with the improved value
                queue.append(neighbor) # Enqueues neighbor to continue BFS
    return dist # Returns the list of distances for all nodes

"""
idx: int. Node index whose feature string should be rebuilt.
X_V: list[dict]. Per-node features of length n. For each i, X_V[i] contains:
    "t": int — node type code (0 = blank, 1 = constant),
    "c": list[int] — sorted concept IDs,
    "r": list[tuple[int,int,int]] — sorted per-relation triples (rel_rank, out, in),
    "f": bytes — binary buffer (this function overwrites it).

Rebuilds the cached binary buffer X_V[idx]["f"] in a portable little-endian format by packing:
  header [t, len(c), len(r)] as uint64 (LE),
  then all concept IDs (uint64 LE),
  then (rel_rank, out, in) triples (uint64 LE each).
This buffer is then hashed by the WL initial coloring.
"""
def update_feature_string(idx, X_V):
    t_code = X_V[idx]["t"]  # Read numeric node type code (0 or 1)
    c_ids = X_V[idx]["c"]   # Read sorted concept IDs (ints)
    per_rel = X_V[idx]["r"] # Read per-relation degree triples (rel_rank, out, in) as ints

    parts = [struct.pack('<QQQ', t_code, len(c_ids), len(per_rel))]  # LE header [t, |c|, |r|]

    if c_ids:
        parts.append(struct.pack('<' + 'Q' * len(c_ids), *c_ids))  # LE concept IDs as uint64 sequence

    for (rel_rank, out_deg, in_deg) in per_rel:
        parts.append(struct.pack('<QQQ', rel_rank, out_deg, in_deg))  # LE triples per relation

    X_V[idx]["f"] = b''.join(parts)  # Store as bytes, canonical LE buffer

