from pathlib import Path

import fsspec
import pandas as pd

from diart import SpeakerDiarization, storage
from diart.inference import Benchmark

DATA_DIR = Path(__file__).parent / "data"

AUDIO_FILE = DATA_DIR / "audio" / "sample.wav"
RTTM_FILE = DATA_DIR / "rttm" / "latency_0.5.rttm"


def test_is_remote():
    assert storage.is_remote("s3://bucket/key")
    assert storage.is_remote("memory://bench/audio")
    assert not storage.is_remote("/tmp/audio")
    assert not storage.is_remote("file:///tmp/audio")
    assert not storage.is_remote(Path("/tmp/audio"))


def test_get_stem():
    assert storage.get_stem("s3://bucket/sub/file.wav") == "file"
    assert storage.get_stem("memory://x/y.rttm") == "y"
    assert storage.get_stem(Path("/tmp/a/file.flac")) == "file"
    assert storage.get_stem("s3://bucket/audio/") == "audio"


def test_localize_local_path_is_not_copied():
    with storage.localize(AUDIO_FILE) as local:
        assert local == AUDIO_FILE


def test_localize_remote_downloads_and_cleans_up():
    fs = fsspec.filesystem("memory")
    fs.pipe_file("loc/sample.wav", AUDIO_FILE.read_bytes())

    with storage.localize("memory://loc/sample.wav") as local:
        local = Path(local)
        # Base name is preserved so the derived URI/stem stays correct
        assert local.name == "sample.wav"
        assert local.exists()
        assert local.read_bytes() == AUDIO_FILE.read_bytes()
        tmp_dir = local.parent

    # The temporary copy is removed on context exit
    assert not tmp_dir.exists()


def test_list_audio_files_filters_and_qualifies():
    fs = fsspec.filesystem("memory")
    fs.pipe_file("lst/sample.wav", AUDIO_FILE.read_bytes())
    fs.pipe_file("lst/other.flac", b"not really audio")
    fs.pipe_file("lst/notes.txt", b"ignore me")

    files = storage.list_audio_files("memory://lst")

    # Only audio files are returned, each as a fully-qualified (remote) URL
    assert len(files) == 2
    assert all(storage.is_remote(f) for f in files)
    assert sorted(storage.get_stem(f) for f in files) == ["other", "sample"]


def _populate_memory(prefix: str):
    fs = fsspec.filesystem("memory")
    fs.pipe_file(f"{prefix}/audio/sample.wav", AUDIO_FILE.read_bytes())
    fs.pipe_file(f"{prefix}/rttm/sample.rttm", RTTM_FILE.read_bytes())


def _run_benchmark(speech_path, reference_path, make_config):
    benchmark = Benchmark(
        speech_path,
        reference_path,
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
