"""On-disk cache for embeddings, so an interrupted run is resumable.

Why this exists: embeddings cost quota, money and time. An interrupted indexing run
once threw away hundreds of already-computed vectors that lived only in memory. This
cache persists each vector as soon as we have it, keyed by a hash of
(model, task_type, text). Re-running then skips everything already embedded and only
calls the API for what is genuinely new.

Kept storage-agnostic and tiny: a JSON file mapping key -> vector. Good enough for a
catalogue of ~1000 titles; the concept (checkpoint expensive work) is what matters.
"""

import hashlib
import json
from pathlib import Path


def text_hash(model: str, task_type: str, text: str) -> str:
    """A stable content key for one embedding.

    The model and task_type are part of the key on purpose: the SAME text embedded
    with a different model or task_type produces a DIFFERENT vector, so it must be a
    different cache entry. The \\x00 separators stop "a" + "bc" from colliding with
    "ab" + "c".
    """
    raw = f"{model}\x00{task_type}\x00{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class EmbeddingCache:
    """A dict of key -> vector, backed by a JSON file on disk."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._store: dict[str, list[float]] = {}
        if self.path.exists():
            self._store = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, key: str) -> list[float] | None:
        return self._store.get(key)

    def put(self, key: str, vector: list[float]) -> None:
        self._store[key] = vector

    def save(self) -> None:
        """Persist the cache atomically.

        We write to a temporary file and then rename it over the real one. rename is
        atomic on every OS, so a crash mid-write can never leave a half-written,
        corrupt cache -- either the old file or the complete new one survives.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._store), encoding="utf-8")
        tmp.replace(self.path)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return key in self._store
