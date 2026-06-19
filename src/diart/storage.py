import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Iterator, List

from .audio import FilePath

try:
    import fsspec

    IS_FSSPEC_AVAILABLE = True
except ImportError:
    fsspec = None
    IS_FSSPEC_AVAILABLE = False

# Protocols that point to the local filesystem and don't require fsspec.
LOCAL_PROTOCOLS = (None, "", "file", "local")

# File extensions recognized as audio when listing a remote directory.
AUDIO_EXTENSIONS = (
    ".wav",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
    ".aiff",
    ".aif",
)


def _split_protocol(path: FilePath):
    """Return the ``(protocol, path)`` pair for a path-like object."""
    text = str(path)
    if IS_FSSPEC_AVAILABLE:
        return fsspec.core.split_protocol(text)
    # Minimal fallback when fsspec is unavailable: detect a leading "scheme://".
    if "://" in text:
        protocol, rest = text.split("://", 1)
        return protocol, rest
    return None, text


def is_remote(path: FilePath) -> bool:
    """Whether `path` points to a remote filesystem (e.g. ``s3://bucket/key``)."""
    protocol, _ = _split_protocol(path)
    return protocol not in LOCAL_PROTOCOLS


def _require_fsspec(path: FilePath):
    """Return an fsspec filesystem for `path`, with a helpful error if a remote
    path is used but the required backend isn't installed."""
    if not IS_FSSPEC_AVAILABLE:
        raise ImportError(
            f"Reading from '{path}' requires extra dependencies. "
            "Install them with: pip install diart[s3]"
        )
    protocol, _ = _split_protocol(path)
    try:
        return fsspec.filesystem(protocol)
    except ImportError as e:
        raise ImportError(
            f"Reading from '{path}' requires the '{protocol}' fsspec backend. "
            "Install it with: pip install diart[s3]"
        ) from e


def get_stem(path: FilePath) -> str:
    """Return the file name without its extension for local or remote paths.

    Unlike `pathlib.Path`, this is safe for remote URLs whose ``//`` would
    otherwise be collapsed (e.g. ``s3://bucket/sub/file.wav`` -> ``file``).
    """
    name = str(path).rstrip("/").rsplit("/", 1)[-1]
    stem, _, _ = name.rpartition(".")
    return stem or name


def exists(path: FilePath) -> bool:
    """Whether a remote path exists (resolves to a directory or object)."""
    fs = _require_fsspec(path)
    _, stripped = _split_protocol(path)
    return fs.exists(stripped)


def list_audio_files(path: FilePath) -> List[str]:
    """List audio files under a remote directory as fully-qualified URLs.

    Parameters
    ----------
    path: FilePath
        Remote directory URL (e.g. ``s3://my-bucket/audio``).

    Returns
    -------
    urls: List[str]
        Fully-qualified URLs (protocol included) for each audio file found,
        sorted for deterministic ordering.
    """
    fs = _require_fsspec(path)
    _, stripped = _split_protocol(path)
    entries = fs.ls(stripped, detail=False)
    audio = [e for e in entries if e.lower().endswith(AUDIO_EXTENSIONS)]
    # `ls` strips the protocol; re-add it so downstream consumers stay protocol-aware.
    return sorted(fs.unstrip_protocol(e) for e in audio)


@contextmanager
def localize(path: FilePath) -> Iterator[FilePath]:
    """Yield a local path for `path`, downloading it first if it is remote.

    Local paths are yielded unchanged (no copy). Remote files are downloaded to
    a temporary directory, preserving the original base name so that the derived
    URI/stem stays correct, and removed when the context exits.
    """
    if not is_remote(path):
        yield path
        return

    fs = _require_fsspec(path)
    _, stripped = _split_protocol(path)
    tmp_dir = tempfile.mkdtemp(prefix="diart-")
    basename = str(path).rstrip("/").rsplit("/", 1)[-1]
    local_path = os.path.join(tmp_dir, basename)
    try:
        fs.get_file(stripped, local_path)
        yield local_path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
