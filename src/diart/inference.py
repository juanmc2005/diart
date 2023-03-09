import logging
from multiprocessing import Pool, freeze_support, RLock, current_process
from pathlib import Path
from traceback import print_exc
from typing import Union, Text, Optional, Callable, Tuple, List, Dict, Hashable

import diart.operators as dops
import diart.sources as src
import numpy as np
import pandas as pd
import rx
import rx.operators as ops
import torch
from diart.blocks import BasePipeline, BasePipelineConfig, Resample
from diart.progress import ProgressBar, RichProgressBar, TQDMProgressBar
from diart.sinks import DiarizationPredictionAccumulator, RealTimePlot, WindowClosedException
from diart.utils import Chronometer
from pyannote.core import Annotation, SlidingWindowFeature
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
from rx.core import Observer
from tqdm import tqdm


class RealTimeInference:
    """Performs inference in real time given a pipeline and an audio source.
    Streams an audio source to an online speaker diarization pipeline.
    It allows users to attach a chain of operations in the form of hooks.

    Parameters
    ----------
    pipeline: BasePipeline
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
    progress_bar: Optional[diart.progress.ProgressBar]
        Progress bar with which to show progress.
        If description is not provided, it will be set to 'Streaming <source uri>'.
        Defaults to RichProgressBar().
    """
    def __init__(
        self,
        pipeline: BasePipeline,
        source: src.AudioSource,
        batch_size: int = 1,
        do_profile: bool = True,
        do_plot: bool = False,
        show_progress: bool = True,
        progress_bar: Optional[ProgressBar] = None,
    ):
        self.pipeline = pipeline
        self.source = source
        self.batch_size = batch_size
        self.do_profile = do_profile
        self.do_plot = do_plot
        self.show_progress = show_progress
        self.accumulator = DiarizationPredictionAccumulator(self.source.uri)
        self.unit = "chunk" if self.batch_size == 1 else "batch"
        self._observers = []

        chunk_duration = self.pipeline.config.duration
        step_duration = self.pipeline.config.step
        sample_rate = self.pipeline.config.sample_rate

        # Estimate the total number of chunks that the source will emit
        self.num_chunks = None
        if self.source.duration is not None:
            numerator = self.source.duration - chunk_duration + step_duration
            self.num_chunks = int(np.ceil(numerator / step_duration))

        # Show progress if required
        self._pbar = progress_bar
        if self.show_progress:
            if self._pbar is None:
                self._pbar = RichProgressBar()
            self._pbar.create(
                total=self.num_chunks,
                description=f"Streaming {self.source.uri}",
                unit=self.unit
            )

        # Initialize chronometer for profiling
        self._chrono = Chronometer(self.unit, self._pbar)

        self.stream = self.source.stream

        # Dynamic resampling if the audio source isn't compatible
        if sample_rate != self.source.sample_rate:
            msg = f"Audio source has sample rate {self.source.sample_rate}, " \
                  f"but pipeline's is {sample_rate}. Will resample."
            logging.warning(msg)
            self.stream = self.stream.pipe(
                ops.map(Resample(self.source.sample_rate, sample_rate))
            )

        # Add rx operators to manage the inputs and outputs of the pipeline
        self.stream = self.stream.pipe(
            dops.rearrange_audio_stream(chunk_duration, step_duration, sample_rate),
            ops.buffer_with_count(count=self.batch_size),
        )

        if self.do_profile:
            self.stream = self.stream.pipe(
                ops.do_action(on_next=lambda _: self._chrono.start()),
                ops.map(self.pipeline),
                ops.do_action(on_next=lambda _: self._chrono.stop()),
            )
        else:
            self.stream = self.stream.pipe(ops.map(self.pipeline))

        self.stream = self.stream.pipe(
            ops.flat_map(lambda results: rx.from_iterable(results)),
            ops.do(self.accumulator),
        )

        if show_progress:
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
        # Close internal states
        self._close_pbar()
        self._close_chronometer()

    def _handle_completion(self):
        # Close internal states
        self._close_pbar()
        self._close_chronometer()

    def __call__(self) -> Annotation:
        """Stream audio chunks from `source` to `pipeline`.

        Returns
        -------
        predictions: Annotation
            Speaker diarization pipeline predictions
        """
        if self.show_progress:
            self._pbar.start()
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
    num_workers: int
        Number of parallel workers.
        Defaults to 0 (no parallelism).
    """
    def __init__(
        self,
        speech_path: Union[Text, Path],
        reference_path: Optional[Union[Text, Path]] = None,
        output_path: Optional[Union[Text, Path]] = None,
        show_progress: bool = True,
        show_report: bool = True,
        batch_size: int = 32,
        num_workers: int = 0,
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
        self.num_workers = num_workers

    @staticmethod
    def _get_pbar_description(filepath: Path, idx_file: int, num_files: int) -> Text:
        return f"Streaming {filepath.stem} ({idx_file + 1}/{num_files})"

    def _run_inference(
        self,
        pipeline: BasePipeline,
        filepath: Path,
        progress_bar: ProgressBar
    ):
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
            progress_bar=progress_bar,
        )

        pred = inference()
        pred.uri = source.uri

        if self.output_path is not None:
            with open(self.output_path / f"{source.uri}.rttm", "w") as out_file:
                pred.write_rttm(out_file)

        return pred

    def _run_inference_job(
        self,
        pipeline_class: type,
        config_dict: Dict[Text, Hashable],
        filepath: Path,
        progress_description: Text,
    ):
        idx_process = int(current_process().name.split('-')[1]) - 1
        # TODO share models across processes
        config = pipeline_class.get_config_class().from_dict(config_dict)
        pipeline = pipeline_class(config)
        progress = TQDMProgressBar(
            progress_description,
            leave=False,
            position=idx_process,
            do_close=True,
        )
        return self._run_inference(pipeline, filepath, progress)

    def __call__(
        self,
        pipeline_class: type,
        config: Union[BasePipelineConfig, Dict[Text, Hashable]],
    ) -> Union[pd.DataFrame, List[Annotation]]:
        """Run a given pipeline on a set of audio files.
        Notice that the internal state of the pipeline is reset before benchmarking.

        Parameters
        ----------
        pipeline: BasePipeline
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

        if self.num_workers == 0:
            # Run sequentially
            predictions = []
            if isinstance(config, dict):
                config = pipeline_class.get_config_class().from_dict(config)
            pipeline = pipeline_class(config)

            for i, filepath in enumerate(audio_file_paths):
                pipeline.reset()
                progress = TQDMProgressBar(
                    self._get_pbar_description(filepath, i, num_audio_files),
                    leave=False,
                    do_close=True,
                )
                predictions.append(self._run_inference(pipeline, filepath, progress))
        else:
            # Run in parallel
            msg = "Benchmark cannot run in parallel " \
                  "if the pipeline config is already instantiated. " \
                  "Try passing a dictionary instead."
            assert isinstance(config, dict), msg

            # Workaround for multiprocessing with GPU
            torch.multiprocessing.set_start_method('spawn')
            # For Windows support
            freeze_support()

            pool = Pool(processes=self.num_workers, initargs=(RLock(),), initializer=tqdm.set_lock)
            arg_list = [
                (
                    pipeline_class,
                    config,
                    filepath,
                    self._get_pbar_description(filepath, i, num_audio_files)
                )
                for i, filepath in enumerate(audio_file_paths)
            ]

            jobs = [pool.apply_async(self._run_inference_job, args=args) for args in arg_list]
            pool.close()
            predictions = [job.get() for job in jobs]

        # Run evaluation
        if self.reference_path is not None:
            metric = DiarizationErrorRate(collar=0, skip_overlap=False)
            progress_bar = TQDMProgressBar("Computing DER", leave=False)
            progress_bar.create(total=len(predictions), unit="file")
            progress_bar.start()
            for hyp in predictions:
                ref = load_rttm(self.reference_path / f"{hyp.uri}.rttm").popitem()[1]
                metric(ref, hyp)
                progress_bar.update()
            progress_bar.close()

            return metric.report(display=self.show_report)

        return predictions
