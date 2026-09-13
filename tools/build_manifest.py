"""Build the deterministic SHA-256 manifest for the public release tree."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST.sha256"
EXCLUDED = {"MANIFEST.sha256"}
IGNORED_DIRS = {".git", ".pytest_cache", ".venv", "__pycache__"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    paths = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and path.relative_to(ROOT).as_posix() not in EXCLUDED
        and not (set(path.relative_to(ROOT).parts) & IGNORED_DIRS)
    )
    lines = [f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in paths]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.name}: {len(lines)} files")


if __name__ == "__main__":
    main()
