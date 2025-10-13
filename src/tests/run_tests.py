from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from pathlib import Path
from wlpp.graph_io import *
from wlpp.preprocessing import *
from wlpp import coloring
from wlpp.cyext.build_backend import build_cython_backend

VERBOSE_BACKEND = True
PROFILE_BACKEND = False

def run_test_case(file_path, expected_blanks, expected_singletons, k, incremental, early_stop, parallel, use_cython):
    print(f"\nRunning test on '{file_path.name}' with "
          f"k={k}, incremental={incremental}, early_stop={early_stop}, parallel={parallel}, "
          f"backend={'Cython' if use_cython else 'Python'}")

    coloring.init_wl_backend(use_cython, VERBOSE_BACKEND)

    try:
        n, adj, X_V_dict, index_to_node, subject_idx = load_graph_from_rdf(str(file_path), False, "subject")

        verbose = False
        necessary_blanks, singletons = wl_preprocessing(n, adj, X_V_dict, index_to_node, subject_idx, k, 86400, incremental, early_stop, parallel, verbose)

        got_blanks = set(necessary_blanks) if necessary_blanks is not None else set()
        got_singletons = set(singletons) if singletons is not None else set()

        passed_blanks = got_blanks == expected_blanks
        passed_singletons = got_singletons == expected_singletons

        if passed_blanks and passed_singletons:
            print("✅ Test PASSED")
            return True
        else:
            print("❌ Test FAILED")
            if not passed_blanks:
                print(f"  ✘ Blanks mismatch:\n    Expected: {expected_blanks}\n    Got:      {got_blanks}")
            if not passed_singletons:
                print(f"  ✘ Singletons mismatch:\n    Expected: {expected_singletons}\n    Got:      {got_singletons}")
            return False

    except Exception as e:
        print(f"❌ Test crashed with exception:\n  {e}")
        return False

def main():
    build_cython_backend(profile=PROFILE_BACKEND)

    k = 2

    project_dir = Path(__file__).parent
    inputs_dir = (project_dir / "inputs").resolve()

    tests_raw = [
        {
            "filename": "Example 1 subject_in_uri.rdf",
            "expected_blanks": {
                'http://example.org/subject/s1',
                'http://example.org/subject/s2',
                'http://example.org/c4',
                'http://example.org/c6'
            },
            "expected_singletons": set()
        },
        {
            "filename": "Example 2 subject_in_uri.rdf",
            "expected_blanks": {
                'http://example.org/subject/s1',
                'http://example.org/subject/s2',
                'http://example.org/c4'
            },
            "expected_singletons": {
                'http://example.org/c3',
                'http://example.org/c5'
            }
        },
        {
            "filename": "Example 3 subject_in_uri.rdf",
            "expected_blanks": {
                'http://example.org/subject/s1',
                'http://example.org/subject/s2',
                'http://example.org/c3',
                'http://example.org/c7'
            },
            "expected_singletons": {
                'http://example.org/c4'
            }
        },
        {
            "filename": "Example 4 subject_in_uri.rdf",
            "expected_blanks": {
                'http://example.org/subject/s1',
                'http://example.org/subject/s2',
                'http://example.org/c3',
                'http://example.org/c4',
                'http://example.org/c5',
                'http://example.org/c6'
            },
            "expected_singletons": set()
        },
        {
            "filename": "Example 5 subject_in_uri.rdf",
            "expected_blanks": {
                'http://example.org/subject/s1',
                'http://example.org/subject/s2',
                'http://example.org/c3',
                'http://example.org/c4'
            },
            "expected_singletons": set()
        },
        {
            "filename": "Example 6 subject_in_uri.rdf",
            "expected_blanks": {
                'http://example.org/subject/s1',
                'http://example.org/subject/s2',
                'http://example.org/c4',
                'http://example.org/c5'
            },
            "expected_singletons": {
                'http://example.org/c3'
            }
        }
    ]

    test_cases = []
    for t in tests_raw:
        p = inputs_dir / t["filename"]
        if not p.exists():
            print(f"⚠️  Skipping missing input: {p}")
            continue
        test_cases.append({
            "file_path": p,
            "expected_blanks": t["expected_blanks"],
            "expected_singletons": t["expected_singletons"],
        })

    if not test_cases:
        print("\nNo test cases found. Check your './inputs' folder and file names.\n")
        return

    parameter_sets = [
        {"incremental": False, "early_stop": False, "parallel": False},
        {"incremental": True,  "early_stop": False, "parallel": False},
        {"incremental": True,  "early_stop": True,  "parallel": False},
        {"incremental": False, "early_stop": False, "parallel": True},
        {"incremental": True,  "early_stop": False, "parallel": True},
        {"incremental": True,  "early_stop": True,  "parallel": True},
    ]

    backends_to_test = [False, True]

    print(f"Running tests for k={k} with {len(parameter_sets)} parameter combinations and {len(test_cases)} test cases...\n")

    grand_total = 0
    grand_passed = 0

    for backend_flag in backends_to_test:
        total = 0
        passed = 0

        print(f"\n=== Backend pass: {'Cython' if backend_flag else 'Python'} ===")

        for test in test_cases:
            for params in parameter_sets:
                total += 1
                grand_total += 1
                result = run_test_case(
                    file_path=test["file_path"],
                    expected_blanks=test["expected_blanks"],
                    expected_singletons=test["expected_singletons"],
                    k=k,
                    use_cython=backend_flag,
                    **params
                )
                if result:
                    passed += 1
                    grand_passed += 1

        print(f"\nSummary for {'Cython' if backend_flag else 'Python'}: {passed}/{total} tests passed.")

    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total tests passed: {grand_passed}/{grand_total}")

if __name__ == "__main__":
    main()
