"""Build the Windows executables with PyInstaller.

Two artefacts, because they have different jobs:

* ``LirikStudio.exe`` - windowed. The application, no console flashing behind it.
* ``lirik-cli.exe`` - console. ``--doctor``, ``--headless-demo`` and ``--offline``,
  for diagnosing a machine without opening a window.

ffmpeg is bundled when available so the packaged app can analyse audio without
asking the user to install anything. It is looked for next to the source tree, on
PATH, or via ``FFMPEG_DIR``; when it cannot be found the build still succeeds and
says so, because a build without audio analysis is still a usable app - silently
producing an app that cannot read a reference file is not.

Usage::

    python build_exe.py                # one-folder build (starts fast)
    python build_exe.py --onefile      # single file (slower start, easier to ship)
    python build_exe.py --no-bundle-ffmpeg
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"
#: Files that must not be excluded: the UI is loaded through these names, and the
#: settings dialog imports the audio engine lazily, so PyInstaller's static analysis
#: misses them without help.
HIDDEN_IMPORTS = [
    "suno_agents.ui.app",
    "suno_agents.ui.agentlog",
    "suno_agents.ui.chat",
    "suno_agents.ui.settings",
    "suno_agents.ui.studio",
    "suno_agents.ui.theme",
    "suno_agents.ui.widgets",
    "suno_agents.audio",
    "suno_agents.audio.analysis",
    "suno_agents.audio.profile",
    "suno_agents.audio.decode",
    "suno_agents.audio.probe",
    "suno_agents.testing.demo",
    "scipy.signal",
    "scipy.fft",
    "scipy.special",
]
#: Modules that must never be bundled. This is not tidiness: PyInstaller's scipy
#: hook collects *every available* array-API backend, so on a machine that happens to
#: have torch installed the build silently grew to 582 MB - 361 MB of it torch, a
#: package this application never imports. The same applies to the other heavy
#: optional backends and to the unrelated packages a shared virtualenv accumulates.
EXCLUDES = [
    # array-API backends scipy looks for but never needs here
    "torch",
    "torchvision",
    "torchaudio",
    "cupy",
    "jax",
    "jaxlib",
    "dask",
    "numba",
    "xarray",
    # unrelated scientific/ML stack
    "matplotlib",
    "pandas",
    "sklearn",
    "cv2",
    "easyocr",
    "ultralytics",
    "transformers",
    "onnx",
    "onnxruntime",
    "sympy",
    # interactive/notebook machinery
    "IPython",
    "ipykernel",
    "notebook",
    "jupyter",
    "jupyter_client",
    "jupyter_core",
    "nbformat",
    "nbconvert",
    "pyzmq",
    # GUI toolkits other than tkinter
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "wx",
    # test and packaging tools (third-party only: excluding stdlib modules breaks
    # libraries that import them - 'unittest' alone made scipy's array-API compat
    # layer raise ModuleNotFoundError at import time)
    "pytest",
    "_pytest",
    "setuptools",
    "pip",
    "wheel",
    "tkinter.test",
    # windows/office bindings pulled in by the environment
    "win32com",
    "pythoncom",
    "pywintypes",
    "Pythonwin",
    "lxml",
    "cryptography",
]

#: ffmpeg needs both executables; ffprobe is what actually reads the metadata.
FFMPEG_BINARIES = ("ffmpeg.exe", "ffprobe.exe")


def find_ffmpeg_dir() -> Path | None:
    """Locate a directory containing both ffmpeg.exe and ffprobe.exe."""
    candidates: list[Path] = []
    override = os.environ.get("FFMPEG_DIR", "").strip()
    if override:
        candidates.append(Path(override))
    for name in ("FFMPEG_BIN", "FFPROBE_BIN"):
        value = os.environ.get(name, "").strip()
        if value and os.path.isfile(value):
            candidates.append(Path(value).parent)
    for executable in ("ffmpeg", "ffprobe"):
        found = shutil.which(executable)
        if found:
            candidates.append(Path(found).parent)
    into_dist = DIST / "LirikStudio" / "ffmpeg"
    candidates.append(into_dist)

    for candidate in candidates:
        if all((candidate / name).is_file() for name in FFMPEG_BINARIES):
            return candidate
    return None


def ffmpeg_size(directory: Path) -> int:
    """Total bytes of the ffmpeg payload, including shared-library siblings.

    A shared ffmpeg build keeps its weight in ``avcodec``-style DLLs that sit beside
    the executables, so copying only the two exes would produce an app that starts
    ffmpeg and immediately fails to load.
    """
    total = 0
    for path in directory.iterdir():
        if path.is_file() and (
            path.name.lower() in FFMPEG_BINARIES
            or path.suffix.lower() in {".dll", ".so", ".dylib"}
        ):
            total += path.stat().st_size
    return total


def pyinstaller_available() -> bool:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        return False
    return True


def build(
    target: str,
    *,
    onefile: bool,
    windowed: bool,
    ffmpeg: Path | None,
    ffmpeg_dest: str = "ffmpeg",
) -> int:
    """Run PyInstaller for one target. Returns its exit code."""
    name = "LirikStudio" if windowed else "lirik-cli"
    args = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name",
        name,
        "--onefile" if onefile else "--onedir",
        "--windowed" if windowed else "--console",
        # The package lives beside this script, not on sys.path.
        f"--paths={ROOT}",
        f"--distpath={DIST}",
        f"--workpath={BUILD}",
        f"--specpath={BUILD}",
    ]
    for module in HIDDEN_IMPORTS:
        args += ["--hidden-import", module]
    for module in EXCLUDES:
        args += ["--exclude-module", module]
    if ffmpeg is not None:
        # Collected into a subfolder, which is where ffbin looks first.
        args += ["--add-data", f"{ffmpeg}{os.pathsep}{ffmpeg_dest}"]
    args.append(str(ROOT / "run_lirik_studio.py"))

    print(f"\n=== building {name} ({'onefile' if onefile else 'onedir'}, "
          f"{'windowed' if windowed else 'console'}) ===", flush=True)
    result = subprocess.run(args, cwd=str(ROOT))
    if result.returncode == 0:
        print(f"    OK -> {DIST / name}", flush=True)
    else:
        print(f"    FAILED (exit {result.returncode})", flush=True)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Lirik Studio executables.")
    parser.add_argument("--onefile", action="store_true",
                        help="Single-file build. Slower to start, easier to ship.")
    parser.add_argument("--no-bundle-ffmpeg", action="store_true",
                        help="Do not embed ffmpeg; the app will need it on PATH.")
    parser.add_argument("--cli-only", action="store_true", help="Build only the CLI.")
    parser.add_argument("--gui-only", action="store_true", help="Build only the GUI.")
    parser.add_argument("--ffmpeg-dir", default="", help="Directory holding ffmpeg.exe.")
    args = parser.parse_args()

    if not pyinstaller_available():
        print("PyInstaller is not installed. Run:  pip install pyinstaller")
        return 2

    ffmpeg: Path | None = None
    if not args.no_bundle_ffmpeg:
        ffmpeg = Path(args.ffmpeg_dir) if args.ffmpeg_dir else find_ffmpeg_dir()
        if ffmpeg is not None:
            megabytes = ffmpeg_size(ffmpeg) / (1024 * 1024)
            print(f"bundling ffmpeg from {ffmpeg} ({megabytes:.0f} MB)")
            if megabytes > 80:
                print(
                    "    note: this looks like a shared ffmpeg build; the executable "
                    "will be large. A static build (e.g. gyan.dev essentials) is "
                    "much smaller."
                )
        else:
            print(
                "WARNING: ffmpeg not found, so the packaged app will not be able to\n"
                "         analyse reference audio until ffmpeg is on PATH or dropped\n"
                "         into an 'ffmpeg' folder beside the executable.\n"
                "         Set FFMPEG_DIR to embed it."
            )

    codes: list[int] = []
    # Only the GUI build carries ffmpeg: the CLI is a developer/diagnostic tool and
    # finds whatever ffmpeg the machine already has. Bundling it twice would double a
    # 167 MB payload for no benefit.
    if not args.gui_only:
        codes.append(build("cli", onefile=args.onefile, windowed=False, ffmpeg=None))
    if not args.cli_only:
        codes.append(build("gui", onefile=args.onefile, windowed=True, ffmpeg=ffmpeg))

    # Smoke the CLI immediately: a build that cannot import its own audio path is a
    # broken deliverable, and discovering that after shipping is worse than a few
    # extra seconds here.
    if not args.cli_only and not args.onefile:
        smoke = DIST / "lirik-cli" / "lirik-cli.exe"
    elif not args.onefile:
        smoke = DIST / "lirik-cli" / "lirik-cli.exe"
    else:
        smoke = DIST / "lirik-cli.exe"
    if smoke.exists():
        print("\n=== smoke test: --doctor ===", flush=True)
        try:
            probe = subprocess.run(
                [str(smoke), "--doctor"], capture_output=True, text=True, timeout=180
            )
            ok = probe.returncode == 0 and "ffmpeg" in probe.stdout
            print(("    OK   " if ok else "    FAIL ") + f"exit={probe.returncode}")
            if not ok:
                print((probe.stdout + probe.stderr)[-1500:])
                codes.append(1)
            else:
                # And that the audio engine actually loads, which is the part that
                # silently broke the first time the exclusion list was too broad.
                if "MISSING" not in probe.stdout.split("numpy")[0]:
                    print("    audio engine import: ok")
        except subprocess.TimeoutExpired:
            print("    FAIL  --doctor did not finish in 180s")
            codes.append(1)

    print("\n=== results ===")
    print(f"  {len(EXCLUDES)} modules excluded (torch and friends are the big ones)")
    for path in sorted(DIST.glob("*")):
        kind = "dir" if path.is_dir() else "file"
        size = _size_of(path)
        print(f"  {path.name:24s} {kind:5s} {size:>9.1f} MB")
    print("\nJalankan:", DIST / ("LirikStudio" if not args.onefile else "LirikStudio.exe"))
    return 0 if all(code == 0 for code in codes) else 1


def _size_of(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


if __name__ == "__main__":
    sys.exit(main())
