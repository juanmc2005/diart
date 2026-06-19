import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Union

import fsspec

# Protocols that point to the local filesystem and don't require fsspec.
LOCAL_PROTOCOLS = (None, "", "file", "local")

# File extensions recognized as audio when listing a directory.
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


def _split_protocol(path: str):
    """Return the ``(protocol, path)`` pair for a path string."""
    return fsspec.core.split_protocol(path)


class FilePath:
    """A file or directory on a local or remote (fsspec) filesystem.

    Parameters
    ----------
    path: str, Path or FilePath
        The path to wrap. Remote URLs keep their protocol; local paths are
        expanded (``~`` resolution).
    """

    def __init__(self, path: "PathLike"):
        if isinstance(path, FilePath):
            self._protocol = path._protocol
            self._path = path._path
            return

        protocol, rest = _split_protocol(str(path))
        self._protocol = protocol
        if protocol in LOCAL_PROTOCOLS:
            # Normalize local path
            self._path = str(Path(rest).expanduser())
        else:
            # Keep remote paths as plain strings (pathlib collapses "//")
            self._path = rest.rstrip("/")

    @property
    def is_remote(self) -> bool:
        """Whether this path points to a remote filesystem."""
        return self._protocol not in LOCAL_PROTOCOLS

    @property
    def _fs(self):
        """The fsspec filesystem backing this (remote) path."""
        return fsspec.filesystem(self._protocol)

    @property
    def name(self) -> str:
        """The final path component, including any extension."""
        return self._path.rstrip("/").rsplit("/", 1)[-1]

    @property
    def stem(self) -> str:
        """The final path component without its extension."""
        name = self.name
        stem, _, _ = name.rpartition(".")
        return stem or name

    @property
    def suffix(self) -> str:
        """The file extension (including the leading dot), or ``""``."""
        name = self.name
        _, dot, ext = name.rpartition(".")
        return f"{dot}{ext}" if dot else ""

    def exists(self) -> bool:
        """Whether this path exists."""
        if self.is_remote:
            return self._fs.exists(self._path)
        return Path(self._path).exists()

    def is_dir(self) -> bool:
        """Whether this path is an existing directory."""
        if self.is_remote:
            return self._fs.isdir(self._path)
        return Path(self._path).is_dir()

    def iterdir(self) -> List["FilePath"]:
        """List the children of this directory as `FilePath` objects."""
        if self.is_remote:
            entries = self._fs.ls(self._path, detail=False)
            return [FilePath(self._fs.unstrip_protocol(e)) for e in entries]
        return [FilePath(child) for child in Path(self._path).iterdir()]

    def __truediv__(self, name: str) -> "FilePath":
        child = FilePath(self)
        child._path = f"{self._path}/{name}"
        return child

    @contextmanager
    def localize(self) -> Iterator["FilePath"]:
        """Yield a local `FilePath` for this path, downloading it if remote.

        Local paths are yielded unchanged (no copy). Remote files are downloaded
        to a temporary directory, preserving the original base name so the
        derived URI/stem stays correct, and removed when the context exits.
        """
        if not self.is_remote:
            yield self
            return

        tmp_dir = tempfile.mkdtemp(prefix="diart-")
        local_path = os.path.join(tmp_dir, self.name)
        try:
            self._fs.get_file(self._path, local_path)
            yield FilePath(local_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def __fspath__(self) -> str:
        if self.is_remote:
            raise TypeError(f"Remote path '{self}' cannot be used as a local path")
        return self._path

    def __str__(self) -> str:
        if self.is_remote:
            return f"{self._protocol}://{self._path}"
        return self._path

    def __repr__(self) -> str:
        return f"FilePath('{self}')"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FilePath):
            return (self._protocol, self._path) == (other._protocol, other._path)
        if isinstance(other, (str, Path)):
            return self == FilePath(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._protocol, self._path))


# Anything that can be coerced into a `FilePath`.
PathLike = Union[str, Path, FilePath]


class Dataset:
    """A benchmark dataset: a directory of audio files and optional reference RTTMs.

    Parameters
    ----------
    audio_path: PathLike
        Directory with audio files. May be local or a remote URL (e.g. ``s3://``).
    reference_path: PathLike or None
        Directory with reference RTTM files whose names match the audio files.
        Defaults to None.
    """

    def __init__(
        self,
        audio_path: PathLike,
        reference_path: Optional[PathLike] = None,
    ):
        self.speech = FilePath(audio_path)
        assert self.speech.is_dir(), "Speech path must be a directory"

        self.reference = None
        if reference_path is not None:
            self.reference = FilePath(reference_path)
            assert self.reference.is_dir(), "Reference path must be a directory"

    @property
    def has_reference(self) -> bool:
        """Whether reference RTTMs are available for evaluation."""
        return self.reference is not None

    def audio_files(self) -> List[FilePath]:
        """Return the audio files in the dataset, sorted for deterministic order."""
        files = [
            f for f in self.speech.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS
        ]
        return sorted(files, key=str)

    def reference_for(self, uri: str) -> FilePath:
        """Return the reference RTTM path matching a given audio URI."""
        assert self.reference is not None, "This dataset has no reference"
        return self.reference / f"{uri}.rttm"
