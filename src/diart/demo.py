import argparse
from pathlib import Path

import diart.sources as src
from diart.pipelines import OnlineSpeakerDiarization
from diart.sinks import OutputBuilder

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
uri = args.source
if args.source != "microphone":
    args.source = Path(args.source).expanduser()
    uri = args.source.name.split(".")[0]
    output_dir = args.source.parent if args.output is None else Path(args.output)
    # Simulate an unreliable recording protocol yielding new audio with a varying refresh rate
    audio_source = src.ReliableFileAudioSource(
        file=args.source,
        uri=uri,
        sample_rate=args.sample_rate,
        window_duration=pipeline.duration,
        step=args.step,
    )
else:
    output_dir = Path("~/").expanduser() if args.output is None else Path(args.output)
    audio_source = src.MicrophoneAudioSource(args.sample_rate)

# Configure output builder to write an RTTM file and to draw in real time
output_builder = OutputBuilder(
    duration=pipeline.duration,
    step=args.step,
    latency=args.latency,
    output_path=output_dir / "output.rttm",
    visualization="slide",
)
# Build pipeline from audio source and stream results to the output builder
pipeline.from_source(audio_source, output_waveform=True).subscribe(output_builder)
# Read audio source as a stream
if args.source == "microphone":
    print("Recording...")
audio_source.read()
