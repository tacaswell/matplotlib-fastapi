"""Custom build backend that compiles TypeScript before Python packaging.

This module wraps setuptools.build_meta and adds a TypeScript compilation step
before building wheels or sdists. It ensures that JavaScript assets are built
and included in the distribution.
"""

import subprocess
import sys
from pathlib import Path

from setuptools import build_meta as _orig

# Re-export all setuptools.build_meta functions
get_requires_for_build_wheel = _orig.get_requires_for_build_wheel
get_requires_for_build_sdist = _orig.get_requires_for_build_sdist
get_requires_for_build_editable = _orig.get_requires_for_build_editable
prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel
prepare_metadata_for_build_editable = _orig.prepare_metadata_for_build_editable
build_sdist = _orig.build_sdist


def _build_javascript() -> None:
    """Build JavaScript/TypeScript files using npm and esbuild.

    Raises
    ------
    SystemExit
        If Node.js is not available or build fails.
    RuntimeError
        If build completes but output file is missing.
    """
    print("=" * 70)
    print("Building JavaScript/TypeScript components...")
    print("=" * 70)

    root = Path(__file__).parent

    # Check if Node.js is available
    try:
        result = subprocess.run(
            ["node", "--version"],
            check=True,
            capture_output=True,
            text=True
        )
        node_version = result.stdout.strip()
        print(f"✓ Found Node.js {node_version}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n" + "!" * 70, file=sys.stderr)
        print("ERROR: Node.js is required to build this package.", file=sys.stderr)
        print("!" * 70, file=sys.stderr)
        print("\nPlease install Node.js >= 20:", file=sys.stderr)
        print("  - Using pixi: pixi install", file=sys.stderr)
        print("  - Direct install: https://nodejs.org/", file=sys.stderr)
        print("\n", file=sys.stderr)
        sys.exit(1)

    # Install npm dependencies
    print("\n📦 Installing npm dependencies...")
    try:
        subprocess.run(
            ["npm", "install", "--quiet"],
            cwd=root,
            check=True,
            capture_output=True
        )
        print("✓ npm dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: npm install failed:", file=sys.stderr)
        print(e.stderr.decode() if e.stderr else str(e), file=sys.stderr)
        sys.exit(1)

    # Build TypeScript
    print("\n🔨 Compiling TypeScript...")
    try:
        subprocess.run(
            ["npm", "run", "build"],
            cwd=root,
            check=True
        )
        print("✓ TypeScript compilation complete")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: TypeScript build failed:", file=sys.stderr)
        sys.exit(1)

    # Verify output exists
    dist_file = root / "mpl_fastapi" / "static" / "js" / "dist" / "component.js"
    if not dist_file.exists():
        raise RuntimeError(
            f"Build completed but output file not found: {dist_file}\n"
            "This indicates a configuration error in esbuild.config.mjs"
        )

    # Report file size
    size_kb = dist_file.stat().st_size / 1024
    print(f"✓ Generated component.js ({size_kb:.1f} KB)")

    print("\n" + "=" * 70)
    print("✅ JavaScript build complete!")
    print("=" * 70 + "\n")


def build_wheel(
    wheel_directory: str,
    config_settings: dict | None = None,
    metadata_directory: str | None = None,
) -> str:
    """Build wheel with JavaScript compilation.

    Parameters
    ----------
    wheel_directory : str
        Directory to write the wheel to.
    config_settings : dict, optional
        Configuration settings from the build frontend.
    metadata_directory : str, optional
        Directory containing prepared metadata.

    Returns
    -------
    str
        Name of the created wheel file.
    """
    _build_javascript()
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(
    wheel_directory: str,
    config_settings: dict | None = None,
    metadata_directory: str | None = None,
) -> str:
    """Build editable wheel with JavaScript compilation.

    Parameters
    ----------
    wheel_directory : str
        Directory to write the wheel to.
    config_settings : dict, optional
        Configuration settings from the build frontend.
    metadata_directory : str, optional
        Directory containing prepared metadata.

    Returns
    -------
    str
        Name of the created wheel file.
    """
    _build_javascript()
    return _orig.build_editable(wheel_directory, config_settings, metadata_directory)
