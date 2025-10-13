from collections import defaultdict
from typing import Dict, List, Set, Tuple, Iterable

"""
color: list[object]. Current node colors; one value per node (typically 64-bit hashes as int).

Build both (1) the frequency of each color and (2) the membership sets of node indices per color. Return (color_counts, color_members).
"""
def build_color_counts_and_members(color: List[object]) -> Tuple[Dict[object, int], Dict[object, Set[int]]]:
    color_counts: Dict[object, int] = defaultdict(int) # Initializes a frequency map: color -> count (defaults to 0)
    color_members: Dict[object, Set[int]] = defaultdict(set) # Initializes a membership map: color -> set of node indices
    for idx, c in enumerate(color): # Iterates over nodes to read their colors
        color_counts[c] += 1 # Increments the frequency for this color
        color_members[c].add(idx) # Inserts the node index into the color's membership set
    return color_counts, color_members # Returns both maps as a tuple

"""
color: list[object]. Current node colors; one value per node (typically 64-bit hashes as int).
color_counts: dict[object,int]. Color → class size.
subjects_idx: Iterable[int]. Node indices representing the protected subjects.
k: int. Minimum required class size.

Return True iff every subject’s color class has size ≥ k.
"""
def check_k_wl_compliance(color: List[object], color_counts: Dict[object, int], subjects_idx: Iterable[int], k: int) -> bool:
    return all(color_counts[color[i]] >= k for i in subjects_idx) # Verifies that every subject's color class has size >= k

"""
color: list[object]. Current node colors; one value per node.

Partition node indices by color and return a canonical representation:
a tuple of sorted tuples (each inner tuple is a color class, sorted), with the outer tuple sorted as well (suitable for equality checks).
"""
def partition_from_colors(color: List[object]):
    groups: Dict[object, List[int]] = defaultdict(list) # Initializes a map: color -> list of node indices
    for idx in range(len(color)): # Iterates over all node indices
        groups[color[idx]].append(idx) # Appends the index to the list of its color class
    return tuple(sorted(tuple(sorted(g)) for g in groups.values())) # Returns a canonical tuple for deterministic comparison



