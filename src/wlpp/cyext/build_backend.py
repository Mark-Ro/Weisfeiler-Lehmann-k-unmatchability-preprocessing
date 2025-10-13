import os
import sys
import importlib
import subprocess
import shutil
from pathlib import Path

def build_cython_backend(profile: bool = False, verbose: bool = True):
    """
    Compila l'estensione Cython come modulo pacchettizzato: wlpp.cython.cy_wl
    e si assicura che il .pyd/.so finisca in src/wlpp/cython/.
    """
    try:
        import setuptools  # noqa
        import wheel       # noqa
        import Cython      # noqa
    except ImportError:
        pyexe = sys.executable
        subprocess.check_call([pyexe, "-m", "pip", "install", "setuptools", "wheel", "Cython"])

    from setuptools import Extension, setup
    from Cython.Build import cythonize

    here = Path(__file__).resolve().parent          # .../src/wlpp/cython
    src_root = here.parent.parent                   # .../src
    target_dir = here                               # dove deve vivere cy_wl.*

    IS_WINDOWS = sys.platform.startswith("win")
    # Flag ottimizzazione
    if IS_WINDOWS:
        extra_compile_args = ["/O2", "/GL", "/Gy", "/Gw", "/DNDEBUG"]
        extra_link_args = ["/LTCG"]
    else:
        extra_compile_args = [
            "-O3","-march=native","-mtune=native",
            "-fno-math-errno","-fno-trapping-math",
            "-pipe","-flto","-DNDEBUG",
        ]
        extra_link_args = ["-flto"]

    define_macros = []
    compiler_directives = {"language_level": 3}
    if profile:
        define_macros = [("CYTHON_TRACE", "1"), ("CYTHON_TRACE_NOGIL", "1")]
        compiler_directives.update({"profile": True, "linetrace": True, "binding": True})
        if IS_WINDOWS:
            extra_compile_args = ["/O2"]
            extra_link_args = []
        else:
            extra_compile_args = ["-O3"]
            extra_link_args = []

    ext = Extension(
        name="wlpp.cyext.cy_wl",
        sources=[str(here / "cy_wl.pyx")],
        language="c",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    )

    build_temp = here / "_build"
    build_temp.mkdir(exist_ok=True)

    # Cambia la CWD alla root 'src' prima di chiamare setup()
    old_cwd = Path.cwd()
    os.chdir(src_root)
    try:
        setup(
            name="wlpp-cy-wl-localbuild",
            ext_modules=cythonize([ext], language_level="3", force=True, compiler_directives=compiler_directives),
            script_args=["build_ext", "--inplace", "--force", "--build-temp", str(build_temp)],
            options={"build_ext": {"inplace": True}},
        )
    finally:
        # Ripristina la CWD anche in caso di eccezione
        os.chdir(old_cwd)

    # Invalida cache e prova import immediato
    importlib.invalidate_caches()
    try:
        import wlpp.cyext.cy_wl  # noqa: F401
        if verbose:
            print("Cython backend built successfully (wlpp.cython.cy_wl).")
        return
    except ImportError:
        # Fallback: copia manuale del .pyd/.so da build/lib.* a wlpp/cython/
        build_dir_parent = src_root / "build"
        if build_dir_parent.exists():
            # cerca la prima build/lib.* contenente il file dell'estensione
            candidates = sorted(build_dir_parent.glob("lib.*"))
            for libdir in candidates:
                for extname in ("*.pyd", "*.so", "*.dll"):
                    matches = list((libdir / "wlpp" / "cyext").glob(f"cy_wl.{extname.split('*.')[-1]}"))
                    if not matches:
                        matches = list((libdir / "wlpp" / "cyext").glob("cy_wl.*"))
                    for compiled in matches:
                        target_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(compiled, target_dir / compiled.name)
                        if verbose:
                            print(f"[Fallback] Copied {compiled} -> {target_dir / compiled.name}")

        # Riprova l'import
        importlib.invalidate_caches()
        import wlpp.cyext.cy_wl
        if verbose:
            print("Cython backend built successfully (wlpp.cyext.cy_wl).")
