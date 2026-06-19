from pathlib import Path

import fsspec
import pandas as pd

from diart import SpeakerDiarization
from diart.inference import Benchmark
from diart.storage import Dataset, FilePath

DATA_DIR = Path(__file__).parent / "data"

AUDIO_FILE = DATA_DIR / "audio" / "sample.wav"
RTTM_FILE = DATA_DIR / "rttm" / "latency_0.5.rttm"


def test_is_remote():
    assert FilePath("s3://bucket/key").is_remote
    assert FilePath("memory://bench/audio").is_remote
    assert not FilePath("/tmp/audio").is_remote
    assert not FilePath("file:///tmp/audio").is_remote
    assert not FilePath(Path("/tmp/audio")).is_remote


def test_stem_name_suffix():
    remote = FilePath("s3://bucket/sub/file.wav")
    assert remote.stem == "file"
    assert remote.name == "file.wav"
    assert remote.suffix == ".wav"

    assert FilePath("memory://x/y.rttm").stem == "y"
    assert FilePath(Path("/tmp/a/file.flac")).stem == "file"
    # A trailing slash (e.g. a directory) yields the last component as the stem
    assert FilePath("s3://bucket/audio/").stem == "audio"


def test_str_roundtrips_protocol():
    assert str(FilePath("s3://bucket/sub/file.wav")) == "s3://bucket/sub/file.wav"
    # Remote paths keep their double slash (pathlib would collapse it)
    assert FilePath("s3://bucket/x.wav") == FilePath("s3://bucket/x.wav")


def test_truediv_joins_for_local_and_remote():
    assert FilePath("s3://bucket/rttm") / "conv.rttm" == FilePath(
        "s3://bucket/rttm/conv.rttm"
    )
    assert FilePath("/data/rttm") / "conv.rttm" == FilePath("/data/rttm/conv.rttm")


def test_localize_local_path_is_not_copied():
    local = FilePath(AUDIO_FILE)
    with local.localize() as localized:
        # Local paths are yielded unchanged (no copy)
        assert localized == local
        assert localized is local


def test_localize_remote_downloads_and_cleans_up():
    fs = fsspec.filesystem("memory")
    fs.pipe_file("loc/sample.wav", AUDIO_FILE.read_bytes())

    with FilePath("memory://loc/sample.wav").localize() as local:
        assert not local.is_remote
        # Base name is preserved so the derived URI/stem stays correct
        assert local.name == "sample.wav"
        path = Path(local)
        assert path.exists()
        assert path.read_bytes() == AUDIO_FILE.read_bytes()
        tmp_dir = path.parent

    # The temporary copy is removed on context exit
    assert not tmp_dir.exists()


def test_dataset_audio_files_filters_and_qualifies():
    fs = fsspec.filesystem("memory")
    fs.pipe_file("ds/audio/sample.wav", AUDIO_FILE.read_bytes())
    fs.pipe_file("ds/audio/other.flac", b"not really audio")
    fs.pipe_file("ds/audio/notes.txt", b"ignore me")
    fs.pipe_file("ds/rttm/sample.rttm", RTTM_FILE.read_bytes())

    dataset = Dataset("memory://ds/audio", "memory://ds/rttm")
    files = dataset.audio_files()

    # Only audio files are returned, each as a fully-qualified remote FilePath
    assert all(isinstance(f, FilePath) and f.is_remote for f in files)
    assert [f.stem for f in files] == ["other", "sample"]  # sorted


def test_dataset_reference_for():
    fs = fsspec.filesystem("memory")
    fs.pipe_file("refds/audio/sample.wav", AUDIO_FILE.read_bytes())
    fs.pipe_file("refds/rttm/sample.rttm", RTTM_FILE.read_bytes())

    dataset = Dataset("memory://refds/audio", "memory://refds/rttm")
    assert dataset.has_reference
    assert dataset.reference_for("sample") == FilePath(
        "memory://refds/rttm/sample.rttm"
    )


def test_dataset_without_reference():
    fs = fsspec.filesystem("memory")
    fs.pipe_file("noref/audio/sample.wav", AUDIO_FILE.read_bytes())

    dataset = Dataset("memory://noref/audio")
    assert not dataset.has_reference


def _populate_memory(prefix: str):
    fs = fsspec.filesystem("memory")
    fs.pipe_file(f"{prefix}/audio/sample.wav", AUDIO_FILE.read_bytes())
    fs.pipe_file(f"{prefix}/rttm/sample.rttm", RTTM_FILE.read_bytes())


def _run_benchmark(speech_path, reference_path, make_config):
    benchmark = Benchmark(
        Dataset(speech_path, reference_path),
        show_report=False,
    )
    return benchmark(SpeakerDiarization, make_config(0.5))


def test_benchmark_over_memory_matches_local(make_config, tmp_path):
    # Local reference layout: matching audio and rttm stems
    (tmp_path / "audio").mkdir()
    (tmp_path / "rttm").mkdir()
    (tmp_path / "audio" / "sample.wav").write_bytes(AUDIO_FILE.read_bytes())
    (tmp_path / "rttm" / "sample.rttm").write_bytes(RTTM_FILE.read_bytes())

    _populate_memory("bench")

    local_report = _run_benchmark(tmp_path / "audio", tmp_path / "rttm", make_config)
    remote_report = _run_benchmark(
        "memory://bench/audio", "memory://bench/rttm", make_config
    )

    assert isinstance(remote_report, pd.DataFrame)
    pd.testing.assert_frame_equal(remote_report, local_report)
