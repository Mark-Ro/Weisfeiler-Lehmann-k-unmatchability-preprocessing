import time
from collections import defaultdict

from . import coloring
from .hash import *



"""
lst: list[T]. The list to split into batches.
batch_size: int. Maximum number of items per batch.

Yield contiguous batches of up to batch_size elements (as list slices), lazily.
"""
def make_batches(lst, batch_size):
    for i in range(0, len(lst), batch_size): # Iterates start indices with step = batch_size
        yield lst[i:i + batch_size] # Yields a slice representing one batch

"""
batch: list[int]. Node indices to verify in this worker.
X_V: list[dict]. Per-node numeric features at indices; for each, fields are "t", "c", "r", "f".
color_data: list[int]. WL colors at the time the batch starts.
color_counts_data: dict[int,int]. Seed color → frequency map for this worker.
adj: tuple of tuples. Compact adjacency where adj[v] is a sequence of (dir_tag:int, rel_id:int, nb:int).
subject_idx: Iterable[int]. Indices of subject nodes to protect.
k: int. Minimum WL color-class size for each subject (k-anonymity).
distances: list[float]. Precomputed BFS distances from any subject to each node (inf if unreachable).
incremental: bool. If True, use incremental WL; otherwise run full WL from the modified coloring.
early_stop: bool. If True and incremental, bound propagation by the subject distance of the candidate.
start_time: float. Global preprocessing start timestamp for timeout checks.
max_seconds: float. Global time budget in seconds.
index_to_node: dict[int,str]. Maps indices to URIs (used for verbose logs).
verbose: bool. If True, print per-candidate results in this worker.
use_cython: bool. If True, initialize the Cython backend; otherwise pure Python.

For each candidate in the batch, temporarily set X_V[b]["t"]=1, rebuild its "f", and test whether k-WL compliance still holds after WL propagation (incremental or full). Return the subset of candidates that are necessary blanks (those that break compliance).
"""
def verify_blanks_batch(batch, X_V, color_data, color_counts_data, adj, subject_idx, k, distances, incremental, early_stop, start_time, max_seconds, index_to_node, verbose, use_cython):
    coloring.init_wl_backend(use_cython, False) # Initializes the WL backend in the worker process

    necessary_in_batch = [] # Collects indices deemed necessary blanks
    color = color_data.copy() # Works on a private copy of the coloring
    color_counts = defaultdict(int, color_counts_data) # Seeds color frequencies as a mutable dict

    for b in batch: # Iterates over candidate indices in this batch
        # Aborts this batch on global timeout
        if time.time() - start_time > max_seconds:
            break

        # Flip numeric node type and rebuild binary buffer
        original_t = X_V[b]['t']  # Save original t_code
        X_V[b]['t'] = 1           # Mark as constant
        coloring.update_feature_string(b, X_V)  # Rebuild "f" for this trial

        color_copy = color.copy() # Clones the coloring for this single trial
        color_counts_copy = dict(color_counts) # Clones color frequencies for this single trial

        if incremental:
            if early_stop:
                d = distances[b] # Reads distance from subjects for this candidate
                if d == float('inf'):
                    color_to_check = coloring.wl_coloring_incremental(len(X_V), adj, X_V, b, color_copy, color_counts_copy, start_time, max_seconds, verbose) # Runs unbounded incremental WL if unreachable
                else:
                    color_to_check = coloring.wl_coloring_incremental(len(X_V), adj, X_V, b, color_copy, color_counts_copy, start_time, max_seconds, verbose, distance_limit=int(d)) # Runs incremental WL bounded by the distance cap
            else:
                color_to_check = coloring.wl_coloring_incremental(len(X_V), adj, X_V, b, color_copy, color_counts_copy, start_time, max_seconds, verbose) # Runs unbounded incremental WL
        else:
            # Compute candidate's new color from updated binary buffer
            new_color_b = fast_hash(X_V[b]["f"]) # Hash binary buffer
            old_color_b = color_copy[b] # Reads candidate's previous color
            color_copy[b] = new_color_b # Injects the new color into the trial coloring
            if new_color_b != old_color_b:
                color_counts_copy[old_color_b] -= 1 # Decrements old color frequency if changed
                color_counts_copy[new_color_b] = color_counts_copy.get(new_color_b, 0) + 1 # Increments new color frequency
            color_to_check = coloring.wl_coloring(len(X_V), adj, color_copy, color_counts_copy, start_time, max_seconds, verbose) # Runs full WL refinement from the modified coloring

        compliant = coloring.check_k_wl_compliance(color_to_check, color_counts_copy, subject_idx, k) # Checks k-WL compliance for all subjects

        # Records candidate as necessary when compliance fails
        if not compliant:
            necessary_in_batch.append(b)

        if verbose:
            status = "is necessary" if not compliant else "is not necessary"
            print(f"[Blank Verification] Blank {index_to_node[b]} {status}")

        # Restore original t_code and binary buffer
        X_V[b]['t'] = original_t  # Restore numeric type
        coloring.update_feature_string(b, X_V)  # Rebuild "f" back

    return necessary_in_batch # Returns indices deemed necessary in this batch

