import os
import time
from collections import Counter

from joblib import Parallel, delayed
from .hash import fast_hash
from .parallel import make_batches, verify_blanks_batch
from . import coloring


"""
n: int. Number of nodes in the graph.
adj: tuple of tuples. Compact adjacency; for each node v, adj[v] is a sequence of (dir_tag:int, rel_id:int, nb:int).
X_V_dict: dict[str, dict]. Per-node raw features keyed by URI (from graph_io). For each URI u:
    "c": set[str]. Set of concept labels,
    "r": list[str]. Sorted list of per-relation degree descriptors "relation:outdeg,indeg".
index_to_node: dict[int,str]. Mapping from node index to node URI.
subject_idx: set[int]. Indices of nodes considered subjects to protect.
k: int. Minimum required color-class size for each subject (k-anonymity under WL coloring).
max_seconds: float. Time budget for the whole preprocessing (used with start_time).
incremental: bool. If True, use incremental WL when testing a candidate blank; otherwise recompute WL from the modified coloring.
early_stop: bool. If True (with incremental), bound propagation by the candidate’s distance from any subject.
parallel: bool. If True, distribute candidate verification across workers using joblib.
verbose: bool. If True, print progress and intermediate results.

Run the WL preprocessing pipeline preserving the original textual canonical order:
  1) Build stable, lexicographic global maps:
       - concept2id from all concept strings,
       - relname2rank from all relation *names*.
  2) For each node i, build numeric per-index features X_V[i] with fields:
       "t"=0 (blank), "c" = concept IDs in lexicographic order of names,
       "r" = (rel_rank, out, in) following the original textual order X_V_dict[u]["r"].
  3) Initial coloring: color[i] = hash(X_V[i]["f"]) where "f" is a little-endian uint64 buffer.
  4) WL refinement to a fixed point; compute color counts and memberships.
  5) Check k-compliance for subjects; seed necessary blanks and singletons.
  6) Rank remaining candidates by BFS distance from subjects.
  7) For each candidate, set "t"=1, rebuild "f", and run WL (incremental or full).
     If compliance fails, mark the candidate as necessary.
Return (set[str] of necessary blanks URIs, set[str] of singleton URIs).
"""
def wl_preprocessing(n, adj, X_V_dict, index_to_node, subject_idx, k, max_seconds, incremental, early_stop, parallel, verbose):

    # Returns early if there are no subjects
    if not subject_idx:
        if verbose:
            print("\nError: No subjects to anonymize in the graph.")
        return None, None

    # Reports the preprocessing mode
    if verbose:
        print("\n[Preprocessing] Starting WL{} preprocessing...\n".format(
            " incremental" if incremental else " full"))

    start_time = time.time()  # Records the preprocessing start time

    all_concepts = set()  # Collect all concept labels across nodes
    for i in range(n):  # Iterate nodes by index
        node_id = index_to_node[i]  # Resolve node URI
        all_concepts.update(X_V_dict[node_id]["c"])  # Add all labels for this node
    concept2id = {c: i + 1 for i, c in enumerate(all_concepts)}

    all_rel_names = set()  # Collect all relation names
    for i in range(n):  # Iterate nodes by index
        node_id = index_to_node[i]  # Resolve node URI
        for entry in X_V_dict[node_id]["r"]:  # Each entry is "relname:out,inn"
            name = entry.rsplit(":", 1)[0]  # Split from the right (tolerate ':' in IRIs)
            all_rel_names.add(name)
    relname2rank = {name: i + 1 for i, name in enumerate(all_rel_names)}

    # Initialize per-index feature dicts with numeric fields
    X_V = [{} for _ in range(n)]  # Allocate per-index feature dicts
    for i in range(n):
        node_id = index_to_node[i]  # Resolve the URI for index i

        X_V[i]["t"] = 0  # Initialize as blank (0) for B^top

        # Convert concepts preserving *lexicographic* order of names via concept2id
        X_V[i]["c"] = [concept2id[c] for c in sorted(X_V_dict[node_id]["c"])]  # Map sorted names -> stable IDs

        # Rebuild per-relation degrees using the already-ordered textual list X_V_dict[u]["r"]
        per_rel = []  # Will hold (rel_rank, out, inn) following original textual order
        for entry in X_V_dict[node_id]["r"]:  # Entries already sorted lexicographically by relation name
            name, degs = entry.rsplit(":", 1)  # 'relname' | 'out,inn'
            out_s, in_s = degs.split(",", 1)   # Split counts
            per_rel.append((relname2rank[name], int(out_s), int(in_s)))  # Map name -> rank and cast counts
        X_V[i]["r"] = per_rel  # Store numeric per-relation descriptors in textual order

        coloring.update_feature_string(i, X_V)  # Builds little-endian binary buffer from t_code, c_ids, per_rel

    color = coloring.wl_initial_coloring(n, X_V, start_time, max_seconds, verbose)  # Initial hash of binary buffers
    color_counts = Counter(color)  # Recomputes frequencies for the current coloring

    color = coloring.wl_coloring(n, adj, color, color_counts, start_time, max_seconds, verbose)  # Runs WL refinement until the partition stabilizes
    color_counts, color_members = coloring.build_color_counts_and_members(color)  # Builds both color frequencies and membership sets

    if not coloring.check_k_wl_compliance(color, color_counts, subject_idx, k):
        if verbose:
            print("[Preprocessing] No k-WL-compliant anonymization possible.")
        return None, None

    necessary_blanks = set(subject_idx)  # Starts with all subjects as necessary blanks
    singletons = {i for i, c in enumerate(color) if color_counts[c] == 1}  # Collects nodes that form singleton color classes

    for i in subject_idx:
        if color_counts[color[i]] == k:  # If a subject’s class size is exactly k
            necessary_blanks.update(color_members[color[i]])  # Marks all nodes in that class as necessary

    if verbose:
        print(f"Initially necessary: {[index_to_node[i] for i in necessary_blanks]}")
        print(f"Singletons: {[index_to_node[i] for i in singletons]}")

    distances = coloring.compute_distances(n, adj, subject_idx, start_time, max_seconds, verbose)  # Computes BFS distances from all subjects
    unmarked_blanks = set(range(n)) - necessary_blanks - singletons  # Defines candidate blanks excluding necessary and singletons
    ranked = sorted(unmarked_blanks, key=lambda b: distances[b])  # Ranks candidates by proximity to subjects

    if verbose:
        print(f"Blanks to verify: {[index_to_node[i] for i in ranked]}\n")

    if parallel:
        num_cores = os.cpu_count() or 1  # Detects available CPU cores
        max_workers = max(1, num_cores)  # Sets the number of workers
        batch_size = max(1, len(ranked) // max_workers)  # Splits candidates into batches
        if verbose:
            print(f"[Parallel] Using {max_workers} workers with batch size {batch_size}")

        color_data = list(color)  # Copies the current coloring for workers
        color_counts_data = dict(color_counts)  # Copies color frequencies for workers
        batches = list(make_batches(ranked, batch_size))  # Builds batches of candidate indices

        use_cython = getattr(coloring, "USING_CYTHON", False)  # Checks which backend is active

        # Launches parallel verification
        results = Parallel(n_jobs=max_workers)(
            delayed(verify_blanks_batch)(
                batch, X_V, color_data, color_counts_data, adj,
                subject_idx, k, distances, incremental, early_stop,
                start_time, max_seconds, index_to_node, verbose,
                use_cython
            )
            for batch in batches
        )
        # Aggregates results from workers
        for necessary_from_batch in results:
            necessary_blanks.update(necessary_from_batch)
    else:
        for b in ranked:
            # Aborts early if the time budget has been exceeded
            if time.time() - start_time > max_seconds:
                if verbose:
                    print(f"\nTimeout reached after {max_seconds} seconds. Stopping early.")
                break
            # Flip node type numerically (t_code) and rebuild binary buffer
            original_t = X_V[b]['t']  # Save original t_code
            X_V[b]['t'] = 1           # Mark candidate as 'c' (constant) = 1
            coloring.update_feature_string(b, X_V)  # Rebuild buffer for the changed node

            color_counts_snapshot = dict(color_counts)  # Clones color frequencies for this trial

            if incremental:
                if early_stop:
                    d = distances[b]  # Uses precomputed distance from subjects
                    if d == float('inf'):
                        color_to_check = coloring.wl_coloring_incremental(n, adj, X_V, b, color, color_counts_snapshot, start_time, max_seconds, verbose)
                    else:
                        color_to_check = coloring.wl_coloring_incremental(n, adj, X_V, b, color, color_counts_snapshot, start_time, max_seconds, verbose, distance_limit=int(d))
                else:
                    color_to_check = coloring.wl_coloring_incremental(n, adj, X_V, b, color, color_counts_snapshot, start_time, max_seconds, verbose)
            else:
                color_to_check = color.copy()  # Starts from current coloring
                color_b = fast_hash(X_V[b]["f"])  # Hash updated binary buffer
                old_color = color_to_check[b]  # Reads previous color
                color_to_check[b] = color_b  # Injects candidate's new color
                if color_b != old_color:
                    color_counts_snapshot[old_color] -= 1  # Decrements old color frequency
                    color_counts_snapshot[color_b] = color_counts_snapshot.get(color_b, 0) + 1  # Increments new color frequency
                color_to_check = coloring.wl_coloring(n, adj, color_to_check, color_counts_snapshot, start_time, max_seconds, verbose)  # Runs full WL refinement from the modified coloring

            compliant = coloring.check_k_wl_compliance(color_to_check, color_counts_snapshot, subject_idx, k)  # Verifies k-WL compliance under this trial

            # Marks candidate as necessary if compliance fails
            if not compliant:
                necessary_blanks.add(b)

            if verbose:
                status = "is necessary" if not compliant else "is not necessary"
                print(f"[Blank Verification] {index_to_node[b]} {status}")

            # Restore original t_code and rebuild buffer back
            X_V[b]['t'] = original_t  # Restore original numeric type
            coloring.update_feature_string(b, X_V)  # Restore buffer accordingly

    return {index_to_node[i] for i in necessary_blanks}, {index_to_node[i] for i in singletons}  # Returns URIs of necessary blanks and singletons
