"""Assemble the exact file set a Hugging Face Space needs.

Run:  python -m scripts.build_hf_space <target-dir>

A Space is its own git repository, so deploying means copying a specific subset of this
project into it: the package, the UI, the requirements, and the Space-shaped Dockerfile
from deploy/huggingface. Doing that by hand is how a stale file or a forgotten one ends up
in production -- and how a `.env` ends up in a public repository.

What is deliberately NOT copied:
- `.env` and anything else holding secrets. The Space gets its keys from its own settings.
- the ingestion scripts, tests and eval data. The Space serves queries; it never builds
  the index (that happens once, from a developer machine, against Qdrant Cloud).
- `data/`, which holds the cached vectors -- 37 MB that the Space would never read.
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPACE_FILES = PROJECT_ROOT / "deploy" / "huggingface"

# (source, destination) relative to the project root and the target respectively.
COPY = [
    ("vibewatch", "vibewatch"),
    ("app.py", "app.py"),
    ("requirements.txt", "requirements.txt"),
    ("deploy/huggingface/Dockerfile", "Dockerfile"),
    ("deploy/huggingface/start.sh", "start.sh"),
    # Carries the YAML header that tells the Space it is a Docker app on port 7860.
    ("deploy/huggingface/README.md", "README.md"),
]

# Anything matching these must never reach a public Space.
FORBIDDEN = {".env", ".env.backup", "data", "qdrant_storage", "notes"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Hugging Face Space file set.")
    parser.add_argument("target", type=Path, help="the cloned Space repository")
    args = parser.parse_args()

    target: Path = args.target.resolve()
    if not target.exists():
        sys.exit(f"{target} does not exist -- clone your Space there first.")

    for source_name, dest_name in COPY:
        source = PROJECT_ROOT / source_name
        dest = target / dest_name
        if not source.exists():
            sys.exit(f"missing: {source}")
        if source.is_dir():
            shutil.rmtree(dest, ignore_errors=True)
            # __pycache__ would be dead weight and can shadow edited modules.
            shutil.copytree(source, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(source, dest)
        print(f"  {source_name}  ->  {dest_name}")

    # A guard, not a formality: this is a PUBLIC repository, and one stray .env is a
    # leaked API key that has to be rotated.
    leaked = sorted(p.name for p in target.iterdir() if p.name in FORBIDDEN)
    if leaked:
        sys.exit(f"\nREFUSING TO FINISH: {leaked} present in {target}. Remove before pushing.")

    print(f"\nReady. Now:\n  cd {target}\n  git add -A && git commit -m 'Deploy Vibewatch' && git push")
    print("\nRemember the three SECRETS in the Space settings:")
    print("  GEMINI_API_KEY · QDRANT_URL · QDRANT_API_KEY")


if __name__ == "__main__":
    main()
