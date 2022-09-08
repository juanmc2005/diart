import logging
from pathlib import Path
from typing import Union, Text, Optional, Callable, Tuple, List

import numpy as np
import pandas as pd
import rx
import rx.operators as ops
from pyannote.core import Annotation, SlidingWindowFeature
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
from rx.core import Observer

import diart.operators as dops
import diart.sources as src
from diart.blocks import OnlineSpeakerDiarization, Resample
from diart.sinks import DiarizationPredictionAccumulator, RTTMWriter, RealTimePlot


class RealTimeInference:
    """
    Simplifies inference in real time for users that do not want to play with the reactivex interface.
    Streams an audio source to an online speaker diarization pipeline.
    It allows users to attach a chain of operations in the form of hooks.

    Parameters
    ----------
    pipeline: OnlineSpeakerDiarization
        Configured speaker diarization pipeline.
    source: AudioSource
        Audio source to be read and streamed.
    do_plot: bool
        Whether to draw predictions in a moving plot. Defaults to True.
    TODO add remaining parameters
    """
    def __init__(
        self,
        pipeline: OnlineSpeakerDiarization,
        source: src.AudioSource,
        batch_size: int = 1,
        do_profile: bool = True,
        do_plot: bool = False,
        show_progress: bool = True,
        progress_desc: Optional[Text] = None,
    ):
        self.pipeline = pipeline
        self.source = source
        self.batch_size = batch_size
        self.do_plot = do_plot
        self.accumulator = DiarizationPredictionAccumulator(source.uri)

        chunk_duration = self.pipeline.config.duration
        step_duration = self.pipeline.config.step
        sample_rate = self.pipeline.config.sample_rate

        operators = []

        # Dynamic resampling if the audio source isn't compatible
        if sample_rate != source.sample_rate:
            msg = f"Audio source has sample rate {source.sample_rate}, " \
                  f"but pipeline's is {sample_rate}. Will resample."
            logging.warning(msg)
            operators.append(ops.map(Resample(source.sample_rate, sample_rate)))

        # Estimate the total number of chunks that the source will emit
        self.num_chunks = None
        if source.duration is not None:
            numerator = source.duration - chunk_duration + step_duration
            self.num_chunks = int(np.ceil(numerator / step_duration))

        # Add rx operators to manage the inputs and outputs of the pipeline
        operators += [
            dops.rearrange_audio_stream(chunk_duration, step_duration, sample_rate),
            ops.buffer_with_count(count=self.batch_size),
            ops.map(pipeline),
            ops.flat_map(lambda results: rx.from_iterable(results)),
            ops.do(self.accumulator),
        ]

        # Show progress if required
        if show_progress:
            desc = f"Streaming {source.uri}" if progress_desc is None else progress_desc
            operators.append(dops.progress(desc, total=self.num_chunks, leave=True))

        # Profile pipeline if required
        if do_profile:
            self.stream = dops.profile(self.source.stream, operators)
        else:
            self.stream = self.source.stream.pipe(*operators)

    def attach_hooks(self, *hooks: Callable[[Tuple[Annotation, SlidingWindowFeature]], None]):
        """Attach hooks to the pipeline.

        Parameters
        ----------
        *hooks: (Tuple[Annotation, SlidingWindowFeature]) -> None
            Hook functions to consume emitted annotations and audio.
        """
        self.stream = self.stream.pipe(*[ops.do_action(hook) for hook in hooks])

    def attach_observers(self, *observers: Observer):
        """Attach rx observers to the pipeline.

        Parameters
        ----------
        *observers: Observer
            Observers to consume emitted annotations and audio.
        """
        self.stream = self.stream.pipe(*[ops.do(sink) for sink in observers])

    def __call__(self) -> Annotation:
        """Stream audio chunks from `source` to `pipeline`.

        Returns
        -------
        predictions: Annotation
            Speaker diarization pipeline predictions
        """
        config = self.pipeline.config
        observable = self.stream
        if self.do_plot:
            # Buffering is needed for the real-time plot, so we do this at the very end
            observable = self.stream.pipe(
                dops.buffer_output(
                    duration=config.duration,
                    step=config.step,
                    latency=config.latency,
                    sample_rate=config.sample_rate,
                ),
                ops.do(RealTimePlot(config.duration, config.latency)),
            )
        observable.subscribe()
        self.source.read()
        return self.accumulator.get_prediction()


class Benchmark:
    """
    Run an online speaker diarization pipeline on a set of audio files in batches.
    It writes predictions to a given output directory.

    If the reference is given, it calculates the average diarization error rate.

    Parameters
    ----------
    speech_path: Text or Path
        Directory with audio files.
    reference_path: Text, Path or None
        Directory with reference RTTM files (same names as audio files).
        If None, performance will not be calculated.
        Defaults to None.
    output_path: Text, Path or None
        Output directory to store predictions in RTTM format.
        If None, predictions will not be written to disk.
        Defaults to None.
    show_progress: bool
        Whether to show progress bars.
        Defaults to True.
    show_report: bool
        Whether to print a performance report to stdout.
        Defaults to True.
    batch_size: int
        Inference batch size.
        If < 2, then it will run in real time.
        If >= 2, then it will pre-calculate segmentation and
        embeddings, running the rest in real time.
        The performance between this two modes does not differ.
        Defaults to 32.
    """
    def __init__(
        self,
        speech_path: Union[Text, Path],
        reference_path: Optional[Union[Text, Path]] = None,
        output_path: Optional[Union[Text, Path]] = None,
        show_progress: bool = True,
        show_report: bool = True,
        batch_size: int = 32,
    ):
        self.speech_path = Path(speech_path).expanduser()
        assert self.speech_path.is_dir(), "Speech path must be a directory"

        # If there's no reference and no output, then benchmark has no output
        msg = "Benchmark expected reference path, output path or both"
        assert reference_path is not None or output_path is not None, msg

        self.reference_path = reference_path
        if reference_path is not None:
            self.reference_path = Path(self.reference_path).expanduser()
            assert self.reference_path.is_dir(), "Reference path must be a directory"

        self.output_path = output_path
        if self.output_path is not None:
            self.output_path = Path(output_path).expanduser()
            self.output_path.mkdir(parents=True, exist_ok=True)

        self.show_progress = show_progress
        self.show_report = show_report
        self.batch_size = batch_size

    def __call__(self, pipeline: OnlineSpeakerDiarization) -> Union[pd.DataFrame, List[Annotation]]:
        """Run a given pipeline on a set of audio files.

        Parameters
        ----------
        pipeline: OnlineSpeakerDiarization
            Configured speaker diarization pipeline.

        Returns
        -------
        performance: pandas.DataFrame or List[Annotation]
            If reference annotations are given, a DataFrame with detailed
            performance on each file as well as average performance.

            If no reference annotations, a list of predictions.
        """
        audio_file_paths = list(self.speech_path.iterdir())
        num_audio_files = len(audio_file_paths)
        predictions = []
        for i, filepath in enumerate(audio_file_paths):
            stream_padding = pipeline.config.latency - pipeline.config.step
            block_size = int(np.rint(pipeline.config.step * pipeline.config.sample_rate))
            source = src.FileAudioSource(filepath, pipeline.config.sample_rate, stream_padding, block_size)
            inference = RealTimeInference(
                pipeline,
                source,
                self.batch_size,
                do_profile=False,
                do_plot=False,
                show_progress=self.show_progress,
                progress_desc=f"Streaming {source.uri} ({i + 1}/{num_audio_files})"
            )
            if self.output_path is not None:
                inference.attach_observers(
                    RTTMWriter(source.uri, self.output_path / f"{source.uri}.rttm")
                )
            predictions.append(inference())

        # Run evaluation
        if self.reference_path is not None:
            metric = DiarizationErrorRate(collar=0, skip_overlap=False)
            for hyp in predictions:
                ref = load_rttm(self.reference_path / f"{hyp.uri}.rttm").popitem()[1]
                metric(ref, hyp)

            return metric.report(display=self.show_report)

        return predictions
