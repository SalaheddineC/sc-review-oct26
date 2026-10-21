"""Check saved result files and their dependencies before displaying them.

The manifest identifies files in results/. Source hashes identify the code
recorded for each numerical run. Files outside results/ are checked for required
fields and matching dependencies, without requiring the baseline manifest.
"""
from hashlib import sha256
import json
from pathlib import Path

RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results"


def verified_result_bytes(path, expected_sha256=None):
    """Check the baseline manifest and any supplied cross-file hash."""
    path = Path(path)
    contents = path.read_bytes()
    actual = sha256(contents).hexdigest()
    if path.resolve().parent == RESULTS_DIRECTORY.resolve():
        manifest = json.loads((RESULTS_DIRECTORY / "manifest.json").read_text(encoding="utf-8"))
        expected = manifest.get("files_sha256", {}).get(path.name)
        if manifest.get("format_version") != 1 or expected != actual:
            raise ValueError(f"Published result integrity mismatch: {path.name}")
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"Saved dependency integrity mismatch: {path.name}")
    return contents


def load_result_json(path, exercise=None, required=()):
    """Check the saved record's identity and the fields needed by its consumer."""
    document = json.loads(verified_result_bytes(path))
    if not isinstance(document, dict) or any(name not in document for name in required):
        raise ValueError(f"Incomplete saved result structure: {Path(path).name}")
    if exercise is not None and document.get("exercise") != exercise:
        raise ValueError(f"Expected saved Exercise {exercise} results")
    for name in ("code_sha256", "core_source_sha256"):
        if name in document:
            hashes = document[name]
            if not isinstance(hashes, dict) or not hashes or any(
                not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hashes.values()):
                raise ValueError(f"Invalid original source identifiers: {Path(path).name}")
    return document
