import argparse
from pathlib import Path

import pandas as pd
import torch

from diart import argdoc
from diart import models as m
from diart import utils
from diart.inference import Benchmark, Parallelize


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        help="Directory with audio files CONVERSATION.(wav|flac|m4a|...)",
    )
    parser.add_argument(
        "--pipeline",
        default="SpeakerDiarization",
        type=str,
        help="Class of the pipeline to optimize. Defaults to 'SpeakerDiarization'",
    )
    parser.add_argument(
        "--segmentation",
        default="pyannote/segmentation",
        type=str,
        help=f"{argdoc.SEGMENTATION}. Defaults to pyannote/segmentation",
    )
    parser.add_argument(
        "--embedding",
        default="pyannote/embedding",
        type=str,
        help=f"{argdoc.EMBEDDING}. Defaults to pyannote/embedding",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Optional. Directory with RTTM files CONVERSATION.rttm. Names must match audio files",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help=f"{argdoc.DURATION}. Defaults to training segmentation duration",
    )
    parser.add_argument(
        "--step", default=0.5, type=float, help=f"{argdoc.STEP}. Defaults to 0.5"
    )
    parser.add_argument(
        "--latency", default=0.5, type=float, help=f"{argdoc.LATENCY}. Defaults to 0.5"
    )
    parser.add_argument(
        "--tau-active", default=0.5, type=float, help=f"{argdoc.TAU}. Defaults to 0.5"
    )
    parser.add_argument(
        "--rho-update", default=0.3, type=float, help=f"{argdoc.RHO}. Defaults to 0.3"
    )
    parser.add_argument(
        "--delta-new", default=1, type=float, help=f"{argdoc.DELTA}. Defaults to 1"
    )
    parser.add_argument(
        "--gamma", default=3, type=float, help=f"{argdoc.GAMMA}. Defaults to 3"
    )
    parser.add_argument(
        "--beta", default=10, type=float, help=f"{argdoc.BETA}. Defaults to 10"
    )
    parser.add_argument(
        "--max-speakers",
        default=20,
        type=int,
        help=f"{argdoc.MAX_SPEAKERS}. Defaults to 20",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help=f"{argdoc.BATCH_SIZE}. Defaults to 32",
    )
    parser.add_argument(
        "--num-workers",
        default=0,
        type=int,
        help=f"{argdoc.NUM_WORKERS}. Defaults to 0 (no parallelism)",
    )
    parser.add_argument(
        "--cpu",
        dest="cpu",
        action="store_true",
        help=f"{argdoc.CPU}. Defaults to GPU if available, CPU otherwise",
    )
    parser.add_argument(
        "--output", type=Path, help=f"{argdoc.OUTPUT}. Defaults to no writing"
    )
    parser.add_argument(
        "--hf-token",
        default="true",
        type=str,
        help=f"{argdoc.HF_TOKEN}. Defaults to 'true' (required by pyannote)",
    )
    args = parser.parse_args()

    # Resolve device
    args.device = torch.device("cpu") if args.cpu else None

    # Resolve models
    hf_token = utils.parse_hf_token_arg(args.hf_token)
    args.segmentation = m.SegmentationModel.from_pyannote(args.segmentation, hf_token)
    args.embedding = m.EmbeddingModel.from_pyannote(args.embedding, hf_token)

    pipeline_class = utils.get_pipeline_class(args.pipeline)

    benchmark = Benchmark(
        args.root,
        args.reference,
        args.output,
        show_progress=True,
        show_report=True,
        batch_size=args.batch_size,
    )

    config = pipeline_class.get_config_class()(**vars(args))
    if args.num_workers > 0:
        benchmark = Parallelize(benchmark, args.num_workers)

    report = benchmark(pipeline_class, config)

    if args.output is not None and isinstance(report, pd.DataFrame):
        report.to_csv(args.output / "benchmark_report.csv")


if __name__ == "__main__":
    run()
