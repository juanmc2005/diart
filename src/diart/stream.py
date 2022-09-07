import argparse
from pathlib import Path

import torch

import diart.argdoc as argdoc
import diart.sources as src
from diart.inference import RealTimeInference
from diart.blocks import OnlineSpeakerDiarization, PipelineConfig
from diart.sinks import RTTMWriter
from diart.models import SegmentationModel, EmbeddingModel


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str, help="Path to an audio file | 'microphone'")
    parser.add_argument("--segmentation", default="pyannote/segmentation", type=str,
                        help=f"{argdoc.SEGMENTATION}. Defaults to pyannote/segmentation")
    parser.add_argument("--embedding", default="pyannote/embedding", type=str,
                        help=f"{argdoc.EMBEDDING}. Defaults to pyannote/embedding")
    parser.add_argument("--step", default=0.5, type=float, help=f"{argdoc.STEP}. Defaults to 0.5")
    parser.add_argument("--latency", default=0.5, type=float, help=f"{argdoc.LATENCY}. Defaults to 0.5")
    parser.add_argument("--tau", default=0.5, type=float, help=f"{argdoc.TAU}. Defaults to 0.5")
    parser.add_argument("--rho", default=0.3, type=float, help=f"{argdoc.RHO}. Defaults to 0.3")
    parser.add_argument("--delta", default=1, type=float, help=f"{argdoc.DELTA}. Defaults to 1")
    parser.add_argument("--gamma", default=3, type=float, help=f"{argdoc.GAMMA}. Defaults to 3")
    parser.add_argument("--beta", default=10, type=float, help=f"{argdoc.BETA}. Defaults to 10")
    parser.add_argument("--max-speakers", default=20, type=int, help=f"{argdoc.MAX_SPEAKERS}. Defaults to 20")
    parser.add_argument("--no-plot", dest="no_plot", action="store_true", help="Skip plotting for faster inference")
    parser.add_argument("--cpu", dest="cpu", action="store_true",
                        help=f"{argdoc.CPU}. Defaults to GPU if available, CPU otherwise")
    parser.add_argument("--output", type=str,
                        help=f"{argdoc.OUTPUT}. Defaults to home directory if SOURCE == 'microphone' or parent directory if SOURCE is a file")
    args = parser.parse_args()
    args.device = torch.device("cpu") if args.cpu else None
    args.segmentation = SegmentationModel.from_pyannote(args.segmentation)
    args.embedding = EmbeddingModel.from_pyannote(args.embedding)

    # Define online speaker diarization pipeline
    config = PipelineConfig.from_namespace(args)
    pipeline = OnlineSpeakerDiarization(config)

    # Manage audio source
    if args.source != "microphone":
        args.source = Path(args.source).expanduser()
        args.output = args.source.parent if args.output is None else Path(args.output)
        stream_padding = config.latency - config.step
        audio_source = src.FileAudioSource(args.source, config.sample_rate, padding_end=stream_padding)
    else:
        args.output = Path("~/").expanduser() if args.output is None else Path(args.output)
        audio_source = src.MicrophoneAudioSource(config.sample_rate)

    # Run online inference
    inference = RealTimeInference(pipeline, audio_source, do_profile=True, do_plot=not args.no_plot)
    inference.attach_observers(RTTMWriter(audio_source.uri, args.output / f"{audio_source.uri}.rttm"))
    inference()


if __name__ == "__main__":
    run()
