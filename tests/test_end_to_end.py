import math
from pathlib import Path

import pytest
from pyannote.database.util import load_rttm

from diart import SpeakerDiarization
from diart.inference import StreamingInference
from diart.sources import FileAudioSource

DATA_DIR = Path(__file__).parent / "data"


@pytest.mark.parametrize("source_file", [DATA_DIR / "audio" / "sample.wav"])
@pytest.mark.parametrize("latency", [0.5, 1, 2, 3, 4, 5])
def test_benchmark(make_config, source_file, latency):
    config = make_config(latency)
    pipeline = SpeakerDiarization(config)

    padding = pipeline.config.get_file_padding(source_file)
    source = FileAudioSource(
        source_file,
        pipeline.config.sample_rate,
        padding,
        pipeline.config.step,
    )

    pipeline.set_timestamp_shift(-padding[0])
    inference = StreamingInference(
        pipeline, source, do_profile=False, do_plot=False, show_progress=False
    )

    pred = inference()

    expected_file = DATA_DIR / "rttm" / f"latency_{latency}.rttm"
    expected = load_rttm(expected_file).popitem()[1]

    assert len(pred) == len(expected)
    for track1, track2 in zip(
        pred.itertracks(yield_label=True), expected.itertracks(yield_label=True)
    ):
        pred_segment, _, pred_spk = track1
        expected_segment, _, expected_spk = track2
        # We can tolerate a difference of up to 50ms
        assert math.isclose(pred_segment.start, expected_segment.start, abs_tol=0.05)
        assert math.isclose(pred_segment.end, expected_segment.end, abs_tol=0.05)
        assert pred_spk == expected_spk
