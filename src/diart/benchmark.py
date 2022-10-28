import argparse
from pathlib import Path

import diart.argdoc as argdoc
import pandas as pd
import torch
from diart import utils
from diart.blocks import OnlineSpeakerDiarization, PipelineConfig
from diart.inference import Benchmark
from diart.models import SegmentationModel, EmbeddingModel


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Directory with audio files CONVERSATION.(wav|flac|m4a|...)")
    parser.add_argument("--segmentation", default="pyannote/segmentation", type=str,
                        help=f"{argdoc.SEGMENTATION}. Defaults to pyannote/segmentation")
    parser.add_argument("--embedding", default="pyannote/embedding", type=str,
                        help=f"{argdoc.EMBEDDING}. Defaults to pyannote/embedding")
    parser.add_argument("--reference", type=Path,
                        help="Optional. Directory with RTTM files CONVERSATION.rttm. Names must match audio files")
    parser.add_argument("--step", default=0.5, type=float, help=f"{argdoc.STEP}. Defaults to 0.5")
    parser.add_argument("--latency", default=0.5, type=float, help=f"{argdoc.LATENCY}. Defaults to 0.5")
    parser.add_argument("--tau", default=0.5, type=float, help=f"{argdoc.TAU}. Defaults to 0.5")
    parser.add_argument("--rho", default=0.3, type=float, help=f"{argdoc.RHO}. Defaults to 0.3")
    parser.add_argument("--delta", default=1, type=float, help=f"{argdoc.DELTA}. Defaults to 1")
    parser.add_argument("--gamma", default=3, type=float, help=f"{argdoc.GAMMA}. Defaults to 3")
    parser.add_argument("--beta", default=10, type=float, help=f"{argdoc.BETA}. Defaults to 10")
    parser.add_argument("--max-speakers", default=20, type=int, help=f"{argdoc.MAX_SPEAKERS}. Defaults to 20")
    parser.add_argument("--batch-size", default=32, type=int, help=f"{argdoc.BATCH_SIZE}. Defaults to 32")
    parser.add_argument("--cpu", dest="cpu", action="store_true",
                        help=f"{argdoc.CPU}. Defaults to GPU if available, CPU otherwise")
    parser.add_argument("--output", type=Path, help=f"{argdoc.OUTPUT}. Defaults to no writing")
    parser.add_argument("--hf-token", default="true", type=str,
                        help=f"{argdoc.HF_TOKEN}. Defaults to 'true' (required by pyannote)")
    args = parser.parse_args()
    args.device = torch.device("cpu") if args.cpu else None
    args.hf_token = utils.parse_hf_token_arg(args.hf_token)

    # Download pyannote models (or get from cache)
    args.segmentation = SegmentationModel.from_pyannote(args.segmentation, args.hf_token)
    args.embedding = EmbeddingModel.from_pyannote(args.embedding, args.hf_token)

    benchmark = Benchmark(
        args.root,
        args.reference,
        args.output,
        show_progress=True,
        show_report=True,
        batch_size=args.batch_size,
    )

    pipeline = OnlineSpeakerDiarization(PipelineConfig.from_namespace(args))
    report = benchmark(pipeline)
    if args.output is not None and isinstance(report, pd.DataFrame):
        report.to_csv(args.output / "benchmark_report.csv")


if __name__ == "__main__":
    run()
