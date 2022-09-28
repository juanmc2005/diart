import logging
from pathlib import Path
from traceback import print_exc
from typing import Union, Text, Optional, Callable, Tuple, List

import diart.operators as dops
import diart.sources as src
import numpy as np
import pandas as pd
import rx
import rx.operators as ops
from diart.blocks import OnlineSpeakerDiarization, Resample
from diart.sinks import DiarizationPredictionAccumulator, RTTMWriter, RealTimePlot, WindowClosedException
from diart.utils import Chronometer
from pyannote.core import Annotation, SlidingWindowFeature
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
from rx.core import Observer
from tqdm import tqdm


class RealTimeInference:
    """
    Performs inference in real time given a pipeline and an audio source.
    Streams an audio source to an online speaker diarization pipeline.
    It allows users to attach a chain of operations in the form of hooks.

    Parameters
    ----------
    pipeline: OnlineSpeakerDiarization
        Configured speaker diarization pipeline.
    source: AudioSource
        Audio source to be read and streamed.
    batch_size: int
        Number of inputs to send to the pipeline at once.
        Defaults to 1.
    do_profile: bool
        If True, compute and report the processing time of the pipeline.
        Defaults to True.
    do_plot: bool
        If True, draw predictions in a moving plot.
        Defaults to False.
    show_progress: bool
        If True, show a progress bar.
        Defaults to True.
    progress_desc: Optional[Text]
        Message to show in the progress bar.
        Defaults to 'Streaming <source uri>'.
    leave_progress_bar: bool
        If True, leaves the progress bar on the screen after the stream has finished.
        Defaults to False.
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
        leave_progress_bar: bool = False,
    ):
        self.pipeline = pipeline
        self.source = source
        self.batch_size = batch_size
        self.do_profile = do_profile
        self.do_plot = do_plot
        self.accumulator = DiarizationPredictionAccumulator(source.uri)
        self._chrono = Chronometer("chunk" if self.batch_size == 1 else "batch")
        self._observers = []

        chunk_duration = self.pipeline.config.duration
        step_duration = self.pipeline.config.step
        sample_rate = self.pipeline.config.sample_rate

        # Estimate the total number of chunks that the source will emit
        self.num_chunks = None
        if source.duration is not None:
            numerator = source.duration - chunk_duration + step_duration
            self.num_chunks = int(np.ceil(numerator / step_duration))

        self.stream = self.source.stream

        # Dynamic resampling if the audio source isn't compatible
        if sample_rate != source.sample_rate:
            msg = f"Audio source has sample rate {source.sample_rate}, " \
                  f"but pipeline's is {sample_rate}. Will resample."
            logging.warning(msg)
            self.stream = self.stream.pipe(
                ops.map(Resample(source.sample_rate, sample_rate))
            )

        # Add rx operators to manage the inputs and outputs of the pipeline
        self.stream = self.stream.pipe(
            dops.rearrange_audio_stream(chunk_duration, step_duration, sample_rate),
            ops.buffer_with_count(count=self.batch_size),
        )

        if self.do_profile:
            self.stream = self.stream.pipe(
                ops.do_action(on_next=lambda _: self._chrono.start()),
                ops.map(pipeline),
                ops.do_action(on_next=lambda _: self._chrono.stop()),
            )
        else:
            self.stream = self.stream.pipe(ops.map(pipeline))

        self.stream = self.stream.pipe(
            ops.flat_map(lambda results: rx.from_iterable(results)),
            ops.do(self.accumulator),
        )

        # Show progress if required
        self._pbar = None
        if show_progress:
            desc = f"Streaming {source.uri}" if progress_desc is None else progress_desc
            self._pbar = tqdm(desc=desc, total=self.num_chunks, unit="chunk", leave=leave_progress_bar)
            self.stream = self.stream.pipe(
                ops.do_action(on_next=lambda _: self._pbar.update())
            )

    def _close_pbar(self):
        if self._pbar is not None:
            self._pbar.close()

    def _close_chronometer(self):
        if self.do_profile:
            if self._chrono.is_running:
                self._chrono.stop(do_count=False)
            self._chrono.report()

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
        self._observers.extend(observers)

    def _handle_error(self, error: BaseException):
        # Compensate for Rx not always calling on_error
        for sink in self._observers:
            sink.on_error(error)
        # Always close the source in case of bad termination
        self.source.close()
        # Special treatment for a user interruption (counted as normal termination)
        window_closed = isinstance(error, WindowClosedException)
        interrupted = isinstance(error, KeyboardInterrupt)
        if not window_closed and not interrupted:
            print_exc()
        # Close progress and chronometer states
        self._close_pbar()
        self._close_chronometer()

    def _handle_completion(self):
        # Close progress and chronometer states
        self._close_pbar()
        self._close_chronometer()

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
        observable.subscribe(
            on_error=self._handle_error,
            on_completed=self._handle_completion,
        )
        # FIXME if read() isn't blocking, the prediction returned is empty
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
        Notice that the internal state of the pipeline is reset before benchmarking.

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
        # Reset pipeline to initial state in case it was modified before
        pipeline.reset()
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
                progress_desc=f"Streaming {source.uri} ({i + 1}/{num_audio_files})",
                leave_progress_bar=False,
            )
            pred = inference()
            pred.uri = source.uri
            predictions.append(pred)

            if self.output_path is not None:
                with open(self.output_path / f"{source.uri}.rttm", "w") as out_file:
                    pred.write_rttm(out_file)

            # Reset internal state for the next file
            pipeline.reset()

        # Run evaluation
        if self.reference_path is not None:
            metric = DiarizationErrorRate(collar=0, skip_overlap=False)
            for hyp in predictions:
                ref = load_rttm(self.reference_path / f"{hyp.uri}.rttm").popitem()[1]
                metric(ref, hyp)

            return metric.report(display=self.show_report)

        return predictions
