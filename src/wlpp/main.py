if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    __package__ = "wlpp"

from .graph_io import load_graph_from_rdf
from .preprocessing import wl_preprocessing
from .cyext.build_backend import build_cython_backend
from .coloring import init_wl_backend
import os
import time


if __name__ == "__main__":
    ###########################################################################
    # >>> USER-CONFIGURABLE PARAMETERS <<<                                    #
    ###########################################################################

    file_path = Path("../../inputs/Esempio 1 subject_in_uri.rdf")

    k = 2
    incremental = False
    early_stop = False
    parallel = False
    USE_CYTHON = False
    profiling = False
    subject_as_concept = False
    subject_identifier = "subject"
    verbose = True
    max_seconds = 86400

    ###########################################################################
    # >>> END <<<                                                             #
    ###########################################################################

    _HERE = Path(__file__).resolve().parent
    file_path = (_HERE / file_path).resolve()


    if early_stop and not incremental:
        raise ValueError("Early stop can only be enabled if incremental=True.")

    if USE_CYTHON:
        build_cython_backend(profile=profiling)
    init_wl_backend(USE_CYTHON, verbose)

    start_loading_time = time.time()

    n, adj, X_V_dict, index_to_node, subject_idx = load_graph_from_rdf(str(file_path), subject_as_concept, subject_identifier)

    graph_loading_time = time.time() - start_loading_time
    start_preprocessing_time = time.time()

    necessary_blanks, singletons = wl_preprocessing(
        n, adj, X_V_dict, index_to_node, subject_idx,
        k, max_seconds, incremental, early_stop, parallel, verbose
    )

    preprocessing_time = time.time() - start_preprocessing_time

    # REPORTING / OUTPUT
    if necessary_blanks is None:
        print("\nError: Preprocessing stopped working.\n")
        print(f"Graph loading time: {graph_loading_time:.3f} seconds")
        print(f"Preprocessing time: {time.time() - start_preprocessing_time:.3f} seconds")
        exit(1)

    input_base_name = file_path.name
    param_suffix = (
        f"k={k}_"
        f"USE_CYTHON={USE_CYTHON}_"
        f"incremental={incremental}_"
        f"early_stop={early_stop}_"
        f"parallel={parallel}_"
        f"subject_as_concept={subject_as_concept}_"
        f"Profiling={profiling}"
    )

    RESULTS_DIR_LITERAL = Path("../../results")
    output_dir = (_HERE / RESULTS_DIR_LITERAL).resolve()
    os.makedirs(output_dir, exist_ok=True)
    output_file_name = output_dir / f"{input_base_name}_{param_suffix}.txt"

    with open(output_file_name, "w", encoding="utf-8") as f:
        f.write(f"Graph loading time: {graph_loading_time:.3f} seconds\n")
        f.write(f"Preprocessing time: {preprocessing_time:.3f} seconds\n")
        f.write(f"\nNumber of necessary blanks: {len(necessary_blanks)}\n")
        f.write(f"Number of singletons: {len(singletons)}\n")
        f.write("\nFinal necessary blanks:\n")
        f.write("\n".join(necessary_blanks) + "\n" if necessary_blanks else "(none)\n")
        f.write("\nSingletons:\n")
        f.write("\n".join(singletons) + "\n" if singletons else "(none)\n")

    if verbose:
        print("\n[Preprocessing] Final results:")
        print(f"Number of necessary blanks: {len(necessary_blanks)}")
        print(f"Number of singletons: {len(singletons)}")
        print("Final necessary blanks:", necessary_blanks if necessary_blanks else set())
        print("Singletons:", singletons if singletons else set())
        print(f"\nTotal preprocessing time: {preprocessing_time:.3f} seconds")
        print(f"Results written to: {output_file_name}")
