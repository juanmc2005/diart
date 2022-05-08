import argparse
from pathlib import Path

import rx.operators as ops
import torch

import diart.argdoc as argdoc
import diart.operators as dops
import diart.sources as src
from diart.pipelines import OnlineSpeakerDiarization, PipelineConfig
from diart.sinks import RealTimePlot, RTTMWriter

# Define script arguments
parser = argparse.ArgumentParser()
parser.add_argument("source", type=str, help="Path to an audio file | 'microphone'")
parser.add_argument("--step", default=0.5, type=float, help=f"{argdoc.STEP}. Defaults to 0.5")
parser.add_argument("--latency", default=0.5, type=float, help=f"{argdoc.LATENCY}. Defaults to 0.5")
parser.add_argument("--tau", default=0.5, type=float, help=f"{argdoc.TAU}. Defaults to 0.5")
parser.add_argument("--rho", default=0.3, type=float, help=f"{argdoc.RHO}. Defaults to 0.3")
parser.add_argument("--delta", default=1, type=float, help=f"{argdoc.DELTA}. Defaults to 1")
parser.add_argument("--gamma", default=3, type=float, help=f"{argdoc.GAMMA}. Defaults to 3")
parser.add_argument("--beta", default=10, type=float, help=f"{argdoc.BETA}. Defaults to 10")
parser.add_argument("--max-speakers", default=20, type=int, help=f"{argdoc.MAX_SPEAKERS}. Defaults to 20")
parser.add_argument("--no-plot", dest="no_plot", action="store_true", help="Skip plotting for faster inference")
parser.add_argument("--gpu", dest="gpu", action="store_true", help=argdoc.GPU)
parser.add_argument("--output", type=str, help=f"{argdoc.OUTPUT}. Defaults to home directory if SOURCE == 'microphone' or parent directory if SOURCE is a file")
args = parser.parse_args()

# Define online speaker diarization pipeline
config = PipelineConfig(
    step=args.step,
    latency=args.latency,
    tau_active=args.tau,
    rho_update=args.rho,
    delta_new=args.delta,
    gamma=args.gamma,
    beta=args.beta,
    max_speakers=args.max_speakers,
    device=torch.device("cuda") if args.gpu else None,
)
pipeline = OnlineSpeakerDiarization(config)

# Manage audio source
if args.source != "microphone":
    args.source = Path(args.source).expanduser()
    output_dir = args.source.parent if args.output is None else Path(args.output)
    audio_source = src.FileAudioSource(
        file=args.source,
        uri=args.source.stem,
        reader=src.RegularAudioFileReader(
            pipeline.sample_rate, pipeline.duration, config.step
        ),
    )
else:
    output_dir = Path("~/").expanduser() if args.output is None else Path(args.output)
    audio_source = src.MicrophoneAudioSource(pipeline.sample_rate)

# Build pipeline from audio source and stream predictions
rttm_writer = RTTMWriter(path=output_dir / f"{audio_source.uri}.rttm")
observable = pipeline.from_source(audio_source).pipe(
    dops.progress(f"Streaming {audio_source.uri}", total=audio_source.length, leave=True)
)
if args.no_plot:
    # Write RTTM file only
    observable.subscribe(rttm_writer)
else:
    # Write RTTM file + buffering and real-time plot
    observable.pipe(
        ops.do(rttm_writer),
        dops.buffer_output(
            duration=pipeline.duration,
            step=config.step,
            latency=config.latency,
            sample_rate=pipeline.sample_rate
        ),
    ).subscribe(RealTimePlot(pipeline.duration, config.latency))

# Read audio source as a stream
audio_source.read()
