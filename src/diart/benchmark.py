from pathlib import Path
from typing import Union, Text, Optional

import pandas as pd
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

import diart.operators as dops
import diart.sources as src
from diart.pipelines import OnlineSpeakerDiarization
from diart.sinks import RTTMWriter


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


if __name__ == "__main__":
    import argparse
    import torch
    import diart.argdoc as argdoc
    from diart.pipelines import PipelineConfig

    # Define script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, help="Directory with audio files CONVERSATION.(wav|flac|m4a|...)")
    parser.add_argument("--reference", type=str, help="Optional. Directory with RTTM files CONVERSATION.rttm. Names must match audio files")
    parser.add_argument("--step", default=0.5, type=float, help=f"{argdoc.STEP}. Defaults to 0.5")
    parser.add_argument("--latency", default=0.5, type=float, help=f"{argdoc.LATENCY}. Defaults to 0.5")
    parser.add_argument("--tau", default=0.5, type=float, help=f"{argdoc.TAU}. Defaults to 0.5")
    parser.add_argument("--rho", default=0.3, type=float, help=f"{argdoc.RHO}. Defaults to 0.3")
    parser.add_argument("--delta", default=1, type=float, help=f"{argdoc.DELTA}. Defaults to 1")
    parser.add_argument("--gamma", default=3, type=float, help=f"{argdoc.GAMMA}. Defaults to 3")
    parser.add_argument("--beta", default=10, type=float, help=f"{argdoc.BETA}. Defaults to 10")
    parser.add_argument("--max-speakers", default=20, type=int, help=f"{argdoc.MAX_SPEAKERS}. Defaults to 20")
    parser.add_argument("--batch-size", default=32, type=int, help="For segmentation and embedding pre-calculation. If BATCH_SIZE < 2, run fully online and estimate real-time latency. Defaults to 32")
    parser.add_argument("--gpu", dest="gpu", action="store_true", help=argdoc.GPU)
    parser.add_argument("--output", type=str, help=f"{argdoc.OUTPUT}. Defaults to `root`")
    args = parser.parse_args()

    # Set benchmark configuration
    benchmark = Benchmark(args.root, args.reference, args.output)

    # Define online speaker diarization pipeline
    pipeline = OnlineSpeakerDiarization(PipelineConfig(
        step=args.step,
        latency=args.latency,
        tau_active=args.tau,
        rho_update=args.rho,
        delta_new=args.delta,
        gamma=args.gamma,
        beta=args.beta,
        max_speakers=args.max_speakers,
        device=torch.device("cuda") if args.gpu else None,
    ))

    benchmark(pipeline, args.batch_size)
