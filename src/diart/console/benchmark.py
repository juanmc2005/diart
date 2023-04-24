import argparse
from pathlib import Path

import pandas as pd
from diart import argdoc
from diart import utils
from diart.inference import Benchmark, Parallelize


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Directory with audio files CONVERSATION.(wav|flac|m4a|...)")
    parser.add_argument("--pipeline", default="SpeakerDiarization", type=str,
                        help="Class of the pipeline to optimize. Defaults to 'SpeakerDiarization'")
    parser.add_argument("--whisper", default="small", type=str,
                        help=f"Whisper model for transcription pipeline. Defaults to 'small'")
    parser.add_argument("--language", default="en", type=str,
                        help=f"Transcribe in this language. Defaults to 'en' (English)")
    parser.add_argument("--segmentation", default="pyannote/segmentation", type=str,
                        help=f"{argdoc.SEGMENTATION}. Defaults to pyannote/segmentation")
    parser.add_argument("--embedding", default="pyannote/embedding", type=str,
                        help=f"{argdoc.EMBEDDING}. Defaults to pyannote/embedding")
    parser.add_argument("--reference", type=Path,
                        help="Optional. Directory with RTTM files CONVERSATION.rttm. Names must match audio files")
    parser.add_argument("--duration", default=5, type=float,
                        help=f"Duration of the sliding window (in seconds). Default value depends on the pipeline")
    parser.add_argument("--step", default=0.5, type=float, help=f"{argdoc.STEP}. Defaults to 0.5")
    parser.add_argument("--latency", default=0.5, type=float, help=f"{argdoc.LATENCY}. Defaults to 0.5")
    parser.add_argument("--tau", default=0.5, type=float, help=f"{argdoc.TAU}. Defaults to 0.5")
    parser.add_argument("--rho", default=0.3, type=float, help=f"{argdoc.RHO}. Defaults to 0.3")
    parser.add_argument("--delta", default=1, type=float, help=f"{argdoc.DELTA}. Defaults to 1")
    parser.add_argument("--gamma", default=3, type=float, help=f"{argdoc.GAMMA}. Defaults to 3")
    parser.add_argument("--beta", default=10, type=float, help=f"{argdoc.BETA}. Defaults to 10")
    parser.add_argument("--max-speakers", default=20, type=int, help=f"{argdoc.MAX_SPEAKERS}. Defaults to 20")
    parser.add_argument("--batch-size", default=32, type=int, help=f"{argdoc.BATCH_SIZE}. Defaults to 32")
    parser.add_argument("--num-workers", default=0, type=int,
                        help=f"{argdoc.NUM_WORKERS}. Defaults to 0 (no parallelism)")
    parser.add_argument("--cpu", dest="cpu", action="store_true",
                        help=f"{argdoc.CPU}. Defaults to GPU if available, CPU otherwise")
    parser.add_argument("--output", type=Path, help=f"{argdoc.OUTPUT}. Defaults to no writing")
    parser.add_argument("--hf-token", default="true", type=str,
                        help=f"{argdoc.HF_TOKEN}. Defaults to 'true' (required by pyannote)")
    args = parser.parse_args()

    pipeline_class = utils.get_pipeline_class(args.pipeline)

    benchmark = Benchmark(
        args.root,
        args.reference,
        args.output,
        show_progress=True,
        show_report=True,
        batch_size=args.batch_size,
    )

    config = pipeline_class.get_config_class().from_dict(vars(args))
    if args.num_workers > 0:
        benchmark = Parallelize(benchmark, args.num_workers)

    report = benchmark(pipeline_class, config)

    if args.output is not None and isinstance(report, pd.DataFrame):
        report.to_csv(args.output / "benchmark_report.csv")


if __name__ == "__main__":
    run()
