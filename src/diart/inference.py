from pathlib import Path
from typing import Union, Text, Optional

import pandas as pd
import rx.operators as ops
from pyannote.core import Annotation
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

import diart.operators as dops
import diart.sources as src
from diart.pipelines import OnlineSpeakerDiarization
from diart.sinks import RTTMWriter, RealTimePlot


class RealTimeInference:
    """
    Streams an audio source to an online speaker diarization pipeline.
    It writes predictions to an output directory in RTTM format and plots them in real time.

    Parameters
    ----------
    output_path: Text or Path
        Output directory to store predictions in RTTM format.
    do_plot: bool
        Whether to draw predictions in a moving plot. Defaults to True.
    """
    def __init__(self, output_path: Union[Text, Path], do_plot: bool = True):
        self.output_path = Path(output_path).expanduser()
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.do_plot = do_plot

    def __call__(self, pipeline: OnlineSpeakerDiarization, source: src.AudioSource) -> Annotation:
        """
        Stream audio chunks from `source` to `pipeline` and write predictions to disk.

        Parameters
        ----------
        pipeline: OnlineSpeakerDiarization
            Configured speaker diarization pipeline.
        source: AudioSource
            Audio source to be read and streamed.

        Returns
        -------
        predictions: Annotation
            Speaker diarization pipeline predictions
        """
        rttm_path = self.output_path / f"{source.uri}.rttm"
        rttm_writer = RTTMWriter(path=rttm_path)
        observable = pipeline.from_source(source).pipe(
            dops.progress(f"Streaming {source.uri}", total=source.length, leave=True)
        )
        if not self.do_plot:
            # Write RTTM file only
            observable.subscribe(rttm_writer)
        else:
            # Write RTTM file + buffering and real-time plot
            observable.pipe(
                ops.do(rttm_writer),
                dops.buffer_output(
                    duration=pipeline.duration,
                    step=pipeline.config.step,
                    latency=pipeline.config.latency,
                    sample_rate=pipeline.sample_rate
                ),
            ).subscribe(RealTimePlot(pipeline.duration, pipeline.config.latency))
        # Stream audio through the pipeline
        source.read()

        return load_rttm(rttm_path)[source.uri]


class Benchmark:
    """
    Run an online speaker diarization pipeline on a set of audio files in batches.
    It writes predictions to a given output directory.

    If the reference is given, it calculates the average diarization error rate.

    Parameters
    ----------
    speech_path: Text or Path
        Directory with audio files.
    reference_path: Text or Path
        Directory with reference RTTM files (same names as audio files).
    output_path: Text or Path
        Output directory to store predictions in RTTM format.
    """
    def __init__(
        self,
        speech_path: Union[Text, Path],
        reference_path: Optional[Union[Text, Path]] = None,
        output_path: Optional[Union[Text, Path]] = None,
    ):
        self.speech_path = Path(speech_path).expanduser()
        assert self.speech_path.is_dir(), "Speech path must be a directory"

        self.reference_path = reference_path
        if reference_path is not None:
            self.reference_path = Path(self.reference_path).expanduser()
            assert self.reference_path.is_dir(), "Reference path must be a directory"

        if output_path is None:
            self.output_path = self.speech_path
        else:
            self.output_path = Path(output_path).expanduser()
            self.output_path.mkdir(parents=True, exist_ok=True)

    def __call__(self, pipeline: OnlineSpeakerDiarization, batch_size: int = 32) -> Optional[pd.DataFrame]:
        """
        Run a given pipeline on a set of audio files using
        pre-calculated segmentation and embeddings in batches.

        Parameters
        ----------
        pipeline: OnlineSpeakerDiarization
            Configured speaker diarization pipeline.
        batch_size: int
            Batch size. Defaults to 32.

        Returns
        -------
        performance: pandas.DataFrame, optional
            DataFrame with detailed performance on each file, as well as average performance.
            None if the reference is not provided.
        """
        chunk_loader = src.ChunkLoader(pipeline.sample_rate, pipeline.duration, pipeline.config.step)
        audio_file_paths = list(self.speech_path.iterdir())
        num_audio_files = len(audio_file_paths)
        for i, filepath in enumerate(audio_file_paths):
            num_chunks = chunk_loader.num_chunks(filepath)

            # Stream fully online if batch size is 1 or lower
            source = None
            if batch_size < 2:
                source = src.FileAudioSource(
                    filepath,
                    filepath.stem,
                    src.RegularAudioFileReader(pipeline.sample_rate, pipeline.duration, pipeline.config.step),
                    # Benchmark the processing time of a single chunk
                    profile=True,
                )
                observable = pipeline.from_source(source, output_waveform=False)
            else:
                observable = pipeline.from_file(
                    filepath,
                    batch_size=batch_size,
                    desc=f"Pre-calculating {filepath.stem} ({i + 1}/{num_audio_files})",
                )

            observable.pipe(
                dops.progress(
                    desc=f"Streaming {filepath.stem} ({i + 1}/{num_audio_files})",
                    total=num_chunks,
                    leave=source is None
                )
            ).subscribe(
                RTTMWriter(path=self.output_path / f"{filepath.stem}.rttm")
            )

            if source is not None:
                source.read()

        # Run evaluation
        if self.reference_path is not None:
            metric = DiarizationErrorRate(collar=0, skip_overlap=False)
            for ref_path in self.reference_path.iterdir():
                ref = load_rttm(ref_path).popitem()[1]
                hyp = load_rttm(self.output_path / ref_path.name).popitem()[1]
                metric(ref, hyp)

            return metric.report(display=True)
