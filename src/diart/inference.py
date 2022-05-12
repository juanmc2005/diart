from pathlib import Path
from typing import Union, Text, Optional

import pandas as pd
import rx.operators as ops
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

import diart.operators as dops
import diart.sources as src
from diart.pipelines import OnlineSpeakerDiarization
from diart.sinks import RTTMWriter, RealTimePlot


class RealTimeInference:
    def __init__(self, output_path: Union[Text, Path], do_plot: bool = True):
        self.output_path = Path(output_path).expanduser()
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.do_plot = do_plot

    def __call__(self, pipeline: OnlineSpeakerDiarization, source: src.AudioSource):
        rttm_writer = RTTMWriter(path=self.output_path / f"{source.uri}.rttm")
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


class Benchmark:
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
        # Run inference
        chunk_loader = src.ChunkLoader(pipeline.sample_rate, pipeline.duration, pipeline.config.step)
        for filepath in self.speech_path.iterdir():
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
                observable = pipeline.from_file(filepath, batch_size=batch_size, verbose=True)

            observable.pipe(
                dops.progress(f"Streaming {filepath.stem}", total=num_chunks, leave=source is None)
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
