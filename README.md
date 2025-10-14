# Weisfeiler-Lehmann Test for k‑unmatchability

This repository implements a pipeline for RDF graph anonymization using **Weisfeiler–Lehman** coloring.  
It identifies which RDF nodes must remain **blank** (unlabeled) to ensure **k‑anonymity** under structural equivalence induced by WL color refinement.

It includes both a **pure Python** and a **Cython-accelerated** backend for performance.

---

## Features

- **Weisfeiler–Lehman (WL) Coloring** for structural graph refinement  
- **Incremental refinement** when local changes occur  
- **k-WL compliance checking** for anonymization guarantees  
- **Parallel candidate verification** via Joblib  
- **Cython backend** for high-performance execution  
- **RDF graph parsing** using RDFlib

---

## Project Structure

```
Weisfeiler‑Lehmann-Test-for-k-unmatchability/
├── coloring.py
├── cy_wl.pyx
├── compliance.py
├── graph_io.py
├── utils.py
├── preprocessing.py
├── parallel.py
├── hash.py
├── main.py
├── run_tests.py
├── setup.py
└── inputs/            # Example RDF files
```

---

## Installation

You can install and run this project in two modes: **pure Python** (no compilation) or **Cython** (fast, but requires a C compiler).

### Option 1 — Quick setup (pure Python)

```bash
git clone https://github.com/Mark-Ro/Weisfeiler-Lehmann-Test-for-k-unmatchability.git
cd Weisfeiler-Lehmann-Test-for-k-unmatchability
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install rdflib xxhash joblib
```

Run the main script:
```bash
python main.py
```

This uses the **Python-only backend**, no compilation required.

---

### Option 2 — Full setup with Cython backend

```bash
git clone https://github.com/Mark-Ro/Weisfeiler-Lehmann-Test-for-k-unmatchability.git
cd Weisfeiler-Lehmann-Test-for-k-unmatchability
python -m venv venv
source venv/bin/activate
pip install rdflib xxhash joblib cython setuptools wheel
```

#### Step 1 — Ensure a C/C++ compiler is installed

Cython requires a working compiler to build the native extension.  
Here's how to set that up:

| Platform | Required toolchain | Installation hint |
|----------|---------------------|--------------------|
| **Windows** | Microsoft Visual C++ Build Tools | Install from Visual Studio Build Tools, selecting “C++ build tools”, ensure `cl.exe` is in PATH |
| **Linux** | GCC/G++ (build-essential) | e.g. `sudo apt install build-essential` |
| **macOS** | Apple Clang / Xcode CLI tools | Run `xcode-select --install` |

---

#### Step 2 — Build the Cython extension

```bash
python setup.py build_ext --inplace
```

This compiles `cy_wl.pyx` into a shared library:
- `.so` on Linux/macOS  
- `.pyd` on Windows  

Alternatively, you can skip manual build: set `USE_CYTHON = True` in `main.py`, and the script will build the extension automatically if missing.

---

### Notes

- **Required Python version:** 3.9 or newer  
- **C/C++ compiler** is necessary *only* for building the Cython backend  
- On **Windows**, install Visual C++ Build Tools for compilation  
- For best performance on larger graphs, the Cython backend is strongly recommended

---

### Verify the setup

Run the test suite:

```bash
python run_tests.py
```

It should pass for both Python and Cython backends (if compiled).

---

## Usage

Run `main.py`, editing only the **USER-CONFIGURABLE PARAMETERS** block:

```python
###########################################################################
    # >>> USER-CONFIGURABLE PARAMETERS <<<                                    #
    # Modify only this section to adapt preprocessing to your dataset or task #
    ###########################################################################

    file_path = "../inputs/Esempio 2 subject_in_uri.rdf"  # Path to the RDF input file

    k = 2  # k-WL: every subject must belong to a WL color class of size >= k

    # EXECUTION STRATEGY
    incremental = False  # If True, use incremental WL when testing each candidate blank
    early_stop = False   # If True (and incremental=True), bound propagation by candidate's distance to any subject
    parallel = False     # If True, verify candidates in parallel using joblib


    # BACKEND SELECTION / BUILD
    USE_CYTHON = False  # If True, use the Cython backend (much faster on large graphs)
    profiling = False   # If True and USE_CYTHON, build Cython in profiling mode (slower but traceable)

    # SUBJECT DETECTION (RDF)
    subject_as_concept = False       # If True, subjects are selected by RDF type concept == subject_identifier. If False, a node is considered a subject only if its URI contains subject_identifier
    subject_identifier = "subject"   # Concept label (if subject_as_concept) OR URI string to detect subjects

    # RUNTIME / LOGGING
    verbose = True     # Print progress and diagnostics to stdout
    max_seconds = 86400  # Global time budget (in seconds) for preprocessing

    ###########################################################################
    # >>> END OF USER-CONFIGURABLE SECTION <<<                                #
    # Do not edit below this line unless you know what you're doing.          #
    ###########################################################################
```

Results are stored automatically in a `results/` or `../results/` folder.

---

## Running the Test Suite

The repository provides `run_tests.py` to verify correctness across multiple RDF inputs and parameter combinations:

```bash
python run_tests.py
```

---

## Example Output

A typical output looks like:

```
[Preprocessing] Final results:
Number of necessary blanks: 3
Number of singletons: 2
Final necessary blanks: {'http://example.org/subject/s1', 'http://example.org/c4', ...}
Singletons: {'http://example.org/c5', 'http://example.org/c3'}
Results written to: ../results/Esempio 2 subject_in_uri.rdf_k=2_USE_CYTHON=False_...
```

---

## Requirements

- rdflib  
- xxhash  
- joblib  
- cython  
- setuptools  
- wheel  

You can produce a `requirements.txt` file:

```
rdflib
xxhash
joblib
cython
setuptools
wheel
```

---
