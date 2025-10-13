from collections import defaultdict
from rdflib import Graph, URIRef, RDF, OWL, RDFS

# Maps textual direction tags to integers (0 = incoming, 1 = outgoing)
_DIR_TAG = {'i': 0, 'o': 1}


"""
n: number of nodes (output).
adj: tuple of adjacency lists per node; each entry is (dir_tag:int, rel_id:int, nb:int).
X_V_dict: per-node raw features keyed by URI; for each node: 
    "c": set[str] of concept labels,
    "r": list[str] of per-relation degree descriptors "relation:outdeg,indeg" (sorted).
index_to_node: mapping node index → URI string.
subject_idx: set[int] of indices for subject nodes to be protected.

Parse an RDF file and build the compact graph representation used by the WL algorithm:
  - collect OWL.NamedIndividuals (excluding RDF/OWL/RDFS classes and properties),
  - map relation IRIs to compact integer IDs,
  - construct adjacency with direction tags (0 = incoming, 1 = outgoing),
  - compute per-relation degree descriptors for each node,
  - identify subject nodes either by concept match or by URI substring.
Return (n, adj, X_V_dict, index_to_node, subject_idx).
"""
def load_graph_from_rdf(file_path, subject_as_concept, subject_identifier):
    g = Graph() # Creates an RDFLib Graph object
    g.parse(file_path) # Parses the RDF file into triples

    V = [] # List of node URIs in insertion order
    E = [] # List of edges as (src_uri, rel_id, dst_uri)
    X_V_dict = {} # Per-node features: {"c": set(), "r": list[str]}
    subjects_set = set() # URIs of subject nodes

    uri_type_map = {} # Maps node URI -> RDF.type URI
    for subj, _, obj in g.triples((None, RDF.type, None)): # Reads all type declarations
        uri_type_map[str(subj)] = str(obj) # Stores type as string

    # Types that are not considered graph nodes
    non_node_types = {
        str(RDF.Property),
        str(OWL.ObjectProperty),
        str(OWL.DatatypeProperty),
        str(RDFS.Class),
        str(OWL.Class)
    }

    individuals = set() # URIs that are OWL.NamedIndividual and not excluded
    for subj in g.subjects(RDF.type, OWL.NamedIndividual):
        subj_str = str(subj)
        if uri_type_map.get(subj_str) not in non_node_types:
            individuals.add(subj_str) # Adds the individual as a graph node candidate

    # Maps relation name -> compact integer ID; also inverse map ID -> relation name
    rel2id = {}
    id_to_rel = {}
    next_rel_id = 1  # ID 0 reserved

    # Iterates over triples to collect nodes, concepts, and edges
    for subj, pred, obj in g:
        if isinstance(subj, URIRef): # Only consider subject if it is a URI node
            subj_str = str(subj)

            # Skips if subject is not a valid individual
            if subj_str not in individuals:
                continue

            if subj_str not in X_V_dict:
                V.append(subj_str) # Adds node to V
                X_V_dict[subj_str] = {"c": set(), "r": set()} # Initializes concept set and relation list

            if pred == RDF.type:
                concept = str(obj).split("/")[-1] # Extracts concept label from URI
                if subject_as_concept and concept == subject_identifier:
                    subjects_set.add(subj_str) # Marks as subject if concept matches
                else:
                    X_V_dict[subj_str]["c"].add(concept) # Adds concept to node's concept set
            else:
                relation = str(pred).split("}")[-1] if "}" in str(pred) else str(pred)
                if relation not in rel2id: # Assigns new ID to unseen relation
                    rel2id[relation] = next_rel_id
                    id_to_rel[next_rel_id] = relation
                    next_rel_id += 1
                E.append((subj_str, rel2id[relation], str(obj))) # Stores edge as (subject, rel_id, object)

    # Identifies subjects by URI substring if not using concept matching
    if not subject_as_concept:
        for uri in V:
            if subject_identifier.lower() in uri.lower():
                subjects_set.add(uri)

    # Builds index mappings for compact representation
    node_to_index = {v: i for i, v in enumerate(V)}
    index_to_node = {i: v for v, i in node_to_index.items()}
    n = len(V)

    # Counts degrees per relation per node: roles_per_rel[v][rel_name] -> {'o': outdeg, 'i': indeg}
    roles_per_rel = defaultdict(lambda: defaultdict(lambda: {'o': 0, 'i': 0}))
    for s, r_id, o in E:
        rel_name = id_to_rel[r_id]
        roles_per_rel[s][rel_name]['o'] += 1 # Increments outdegree for source in this relation
        roles_per_rel[o][rel_name]['i'] += 1 # Increments indegree for target in this relation

    # Converts degree counts into sorted list of strings "relation:outdeg,indeg"
    for v in V:
        per_rel_entries = []
        for rel_name in sorted(roles_per_rel[v].keys()):
            out_deg = roles_per_rel[v][rel_name]['o']
            in_deg = roles_per_rel[v][rel_name]['i']
            per_rel_entries.append(f"{rel_name}:{out_deg},{in_deg}")
        X_V_dict[v]['r'] = per_rel_entries # Stores sorted per-relation degree descriptors

    subject_idx = {node_to_index[s] for s in subjects_set if s in node_to_index} # Maps subject URIs to node indices

    # Builds compact adjacency: adj[v] = list of (dir_tag, rel_id, nb)
    adj_list = [[] for _ in range(n)]
    for s, r_id, o in E:
        if s in node_to_index and o in node_to_index:
            s_idx = node_to_index[s]
            o_idx = node_to_index[o]
            adj_list[s_idx].append((_DIR_TAG['o'], r_id, o_idx)) # Outgoing edge
            adj_list[o_idx].append((_DIR_TAG['i'], r_id, s_idx)) # Incoming edge

    adj = tuple(tuple(neigh) for neigh in adj_list) # Freezes adjacency as tuple of tuples
    return n, adj, X_V_dict, index_to_node, subject_idx

