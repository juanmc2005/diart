import argparse
from pathlib import Path

import diart.operators as dops
import diart.sources as src
import rx.operators as ops
from diart.pipelines import OnlineSpeakerDiarization
from diart.sinks import RealTimePlot, RTTMWriter

# Define script arguments
parser = argparse.ArgumentParser()
parser.add_argument("source", type=str, help="Path to an audio file | 'microphone'")
parser.add_argument("--step", default=0.5, type=float, help="Source sliding window step")
parser.add_argument("--latency", default=0.5, type=float, help="System latency")
parser.add_argument("--sample-rate", default=16000, type=int, help="Source sample rate")
parser.add_argument("--tau", default=0.5, type=float, help="Activity threshold tau active")
parser.add_argument("--rho", default=0.3, type=float, help="Speech duration threshold rho update")
parser.add_argument("--delta", default=1, type=float, help="Maximum distance threshold delta new")
parser.add_argument("--gamma", default=3, type=float, help="Parameter gamma for overlapped speech penalty")
parser.add_argument("--beta", default=10, type=float, help="Parameter beta for overlapped speech penalty")
parser.add_argument("--max-speakers", default=20, type=int, help="Maximum number of identifiable speakers")
parser.add_argument(
    "--output", type=str,
    help="Output directory to store the RTTM. Defaults to home directory "
         "if source is microphone or parent directory if source is a file"
)
args = parser.parse_args()

# Define online speaker diarization pipeline
pipeline = OnlineSpeakerDiarization(
    step=args.step,
    latency=args.latency,
    tau_active=args.tau,
    rho_update=args.rho,
    delta_new=args.delta,
    gamma=args.gamma,
    beta=args.beta,
    max_speakers=args.max_speakers,
)

# Manage audio source
if args.source != "microphone":
    args.source = Path(args.source).expanduser()
    uri = args.source.name.split(".")[0]
    output_dir = args.source.parent if args.output is None else Path(args.output)
    audio_source = src.FileAudioSource(
        file=args.source,
        uri=uri,
        reader=src.RegularAudioFileReader(
            args.sample_rate, pipeline.duration, pipeline.step
        ),
    )
else:
    output_dir = Path("~/").expanduser() if args.output is None else Path(args.output)
    audio_source = src.MicrophoneAudioSource(args.sample_rate)

# Build pipeline from audio source and stream predictions to a real-time plot
pipeline.from_source(audio_source).pipe(
    ops.do(RTTMWriter(path=output_dir / "output.rttm")),
    dops.buffer_output(
        duration=pipeline.duration,
        step=pipeline.step,
        latency=pipeline.latency,
        sample_rate=audio_source.sample_rate
    ),
).subscribe(RealTimePlot(pipeline.duration, pipeline.latency))

# Read audio source as a stream
if args.source == "microphone":
    print("Recording...")
audio_source.read()
