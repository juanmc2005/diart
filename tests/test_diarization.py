from __future__ import annotations

import random

import pytest

from diart import SpeakerDiarizationConfig, SpeakerDiarization
from utils import build_waveform_swf


@pytest.fixture
def random_diarization_config(
    segmentation_model, embedding_model
) -> SpeakerDiarizationConfig:
    duration = round(random.uniform(1, 10), 1)
    step = round(random.uniform(0.1, duration), 1)
    latency = round(random.uniform(step, duration), 1)
    return SpeakerDiarizationConfig(
        segmentation=segmentation_model,
        embedding=embedding_model,
        duration=duration,
        step=step,
        latency=latency,
    )


@pytest.fixture(scope="session")
def min_latency_config(segmentation_model, embedding_model) -> SpeakerDiarizationConfig:
    return SpeakerDiarizationConfig(
        segmentation=segmentation_model,
        embedding=embedding_model,
        duration=5,
        step=0.5,
        latency="min",
    )


@pytest.fixture(scope="session")
def max_latency_config(segmentation_model, embedding_model) -> SpeakerDiarizationConfig:
    return SpeakerDiarizationConfig(
        segmentation=segmentation_model,
        embedding=embedding_model,
        duration=5,
        step=0.5,
        latency="max",
    )


def test_config(
    segmentation_model, embedding_model, min_latency_config, max_latency_config
):
    duration = round(random.uniform(1, 10), 1)
    step = round(random.uniform(0.1, duration), 1)
    latency = round(random.uniform(step, duration), 1)
    config = SpeakerDiarizationConfig(
        segmentation=segmentation_model,
        embedding=embedding_model,
        duration=duration,
        step=step,
        latency=latency,
    )

    assert config.duration == duration
    assert config.step == step
    assert config.latency == latency
    assert min_latency_config.latency == min_latency_config.step
    assert max_latency_config.latency == max_latency_config.duration


def test_bad_latency(segmentation_model, embedding_model):
    duration = round(random.uniform(1, 10), 1)
    step = round(random.uniform(0.5, duration - 0.2), 1)
    latency_too_low = round(random.uniform(0, step - 0.1), 1)
    latency_too_high = round(random.uniform(duration + 0.1, 100), 1)

    config1 = SpeakerDiarizationConfig(
        segmentation=segmentation_model,
        embedding=embedding_model,
        duration=duration,
        step=step,
        latency=latency_too_low,
    )
    config2 = SpeakerDiarizationConfig(
        segmentation=segmentation_model,
        embedding=embedding_model,
        duration=duration,
        step=step,
        latency=latency_too_high,
    )

    with pytest.raises(AssertionError):
        SpeakerDiarization(config1)

    with pytest.raises(AssertionError):
        SpeakerDiarization(config2)


def test_pipeline_build(random_diarization_config):
    pipeline = SpeakerDiarization(random_diarization_config)

    assert pipeline.get_config_class() == SpeakerDiarizationConfig

    hparams = pipeline.hyper_parameters()
    hp_names = [hp.name for hp in hparams]
    assert len(set(hp_names)) == 3

    for hparam in hparams:
        assert hparam.low == 0
        if hparam.name in ["tau_active", "rho_update"]:
            assert hparam.high == 1
        elif hparam.name == "delta_new":
            assert hparam.high == 2
        else:
            assert False

    assert pipeline.config == random_diarization_config


def test_timestamp_shift(random_diarization_config):
    pipeline = SpeakerDiarization(random_diarization_config)

    assert pipeline.timestamp_shift == 0

    new_shift = round(random.uniform(-10, 10), 1)
    pipeline.set_timestamp_shift(new_shift)
    assert pipeline.timestamp_shift == new_shift

    waveform = build_waveform_swf(
        random_diarization_config.duration,
        random_diarization_config.sample_rate,
    )
    prediction, _ = pipeline([waveform])[0]

    for segment, _, label in prediction.itertracks(yield_label=True):
        assert segment.start >= new_shift
        assert segment.end >= new_shift

    pipeline.reset()
    assert pipeline.timestamp_shift == 0


def test_call_min_latency(min_latency_config):
    pipeline = SpeakerDiarization(min_latency_config)
    waveform1 = build_waveform_swf(
        min_latency_config.duration,
        min_latency_config.sample_rate,
        start_time=0,
    )
    waveform2 = build_waveform_swf(
        min_latency_config.duration,
        min_latency_config.sample_rate,
        min_latency_config.step,
    )

    batch = [waveform1, waveform2]
    output = pipeline(batch)

    pred1, wave1 = output[0]
    pred2, wave2 = output[1]

    assert waveform1.data.shape[0] == wave1.data.shape[0]
    assert wave1.data.shape[0] > wave2.data.shape[0]

    pred1_timeline = pred1.get_timeline()
    pred2_timeline = pred2.get_timeline()
    pred1_duration = round(pred1_timeline[-1].end - pred1_timeline[0].start, 3)
    pred2_duration = round(pred2_timeline[-1].end - pred2_timeline[0].start, 3)

    expected_duration = round(min_latency_config.duration, 3)
    expected_step = round(min_latency_config.step, 3)
    assert not pred1_timeline or pred1_duration <= expected_duration
    assert not pred2_timeline or pred2_duration <= expected_step


def test_call_max_latency(max_latency_config):
    pipeline = SpeakerDiarization(max_latency_config)
    waveform1 = build_waveform_swf(
        max_latency_config.duration,
        max_latency_config.sample_rate,
        start_time=0,
    )
    waveform2 = build_waveform_swf(
        max_latency_config.duration,
        max_latency_config.sample_rate,
        max_latency_config.step,
    )

    batch = [waveform1, waveform2]
    output = pipeline(batch)

    pred1, wave1 = output[0]
    pred2, wave2 = output[1]

    assert waveform1.data.shape[0] > wave1.data.shape[0]
    assert wave1.data.shape[0] == wave2.data.shape[0]

    pred1_timeline = pred1.get_timeline()
    pred2_timeline = pred2.get_timeline()
    pred1_duration = pred1_timeline[-1].end - pred1_timeline[0].start
    pred2_duration = pred2_timeline[-1].end - pred2_timeline[0].start

    expected_step = round(max_latency_config.step, 3)
    assert not pred1_timeline or round(pred1_duration, 3) <= expected_step
    assert not pred2_timeline or round(pred2_duration, 3) <= expected_step
