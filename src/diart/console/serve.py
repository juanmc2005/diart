import argparse
from pathlib import Path

import torch

from diart import argdoc
from diart import models as m
from diart import utils
from diart.websockets import WebSocketStreamingServer


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", type=str, help="Server host")
    parser.add_argument("--port", default=7007, type=int, help="Server port")
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
        "--duration",
        type=float,
        default=5,
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
        "--cpu",
        dest="cpu",
        action="store_true",
        help=f"{argdoc.CPU}. Defaults to GPU if available, CPU otherwise",
    )
    parser.add_argument(
        "--hf-token",
        default="true",
        type=str,
        help=f"{argdoc.HF_TOKEN}. Defaults to 'true' (required by pyannote)",
    )
    parser.add_argument(
        "--normalize-embedding-weights",
        action="store_true",
        help=f"{argdoc.NORMALIZE_EMBEDDING_WEIGHTS}. Defaults to False",
    )
    args = parser.parse_args()

    # Resolve device
    args.device = torch.device("cpu") if args.cpu else None

    # Resolve models
    hf_token = utils.parse_hf_token_arg(args.hf_token)
    args.segmentation = m.SegmentationModel.from_pretrained(args.segmentation, hf_token)
    args.embedding = m.EmbeddingModel.from_pretrained(args.embedding, hf_token)

    # Resolve pipeline configuration
    pipeline_class = utils.get_pipeline_class(args.pipeline)
    pipeline_config = pipeline_class.get_config_class()(**vars(args))

    # Initialize Websocket server
    server = WebSocketStreamingServer(
        pipeline_class=pipeline_class,
        pipeline_config=pipeline_config,
        host=args.host,
        port=args.port,
    )

    server.run()


if __name__ == "__main__":
    run()
