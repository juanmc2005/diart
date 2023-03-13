import argparse
from pathlib import Path

import diart.argdoc as argdoc
import diart.sources as src
from diart.blocks import OnlineSpeakerDiarization, PipelineConfig
from diart.inference import RealTimeInference
from diart.sinks import RTTMWriter


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", type=str, help="Server host")
    parser.add_argument("--port", default=7007, type=int, help="Server port")
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
    parser.add_argument("--cpu", dest="cpu", action="store_true",
                        help=f"{argdoc.CPU}. Defaults to GPU if available, CPU otherwise")
    parser.add_argument("--output", type=Path, help=f"{argdoc.OUTPUT}. Defaults to no writing")
    parser.add_argument("--hf-token", default="true", type=str,
                        help=f"{argdoc.HF_TOKEN}. Defaults to 'true' (required by pyannote)")
    args = parser.parse_args()

    # Define online speaker diarization pipeline
    config = PipelineConfig.from_dict(vars(args))
    pipeline = OnlineSpeakerDiarization(config)

    # Create websocket audio source
    audio_source = src.WebSocketAudioSource(config.sample_rate, args.host, args.port)

    # Run online inference
    inference = RealTimeInference(
        pipeline,
        audio_source,
        batch_size=1,
        do_profile=True,
        do_plot=False,
        show_progress=True,
    )

    # Write to disk if required
    if args.output is not None:
        inference.attach_observers(RTTMWriter(audio_source.uri, args.output / f"{audio_source.uri}.rttm"))

    # Send back responses as RTTM text lines
    inference.attach_hooks(lambda ann_wav: audio_source.send(ann_wav[0].to_rttm()))

    # Run server and pipeline
    inference()


if __name__ == "__main__":
    run()
