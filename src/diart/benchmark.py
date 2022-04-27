import argparse
from pathlib import Path

from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
from tqdm import tqdm

import diart.operators as dops
import diart.sources as src
from diart.pipelines import OnlineSpeakerDiarization, PipelineConfig
from diart.sinks import RTTMWriter

# Define script arguments
parser = argparse.ArgumentParser()
parser.add_argument("root", type=str, help="Directory with audio files <conversation>.(wav|flac|m4a|...)")
parser.add_argument("reference", type=str, help="Directory with RTTM files <conversation>.rttm")
parser.add_argument("--step", default=0.5, type=float, help="Source sliding window step")
parser.add_argument("--latency", default=0.5, type=float, help="System latency")
parser.add_argument("--tau", default=0.5, type=float, help="Activity threshold tau active")
parser.add_argument("--rho", default=0.3, type=float, help="Speech duration threshold rho update")
parser.add_argument("--delta", default=1, type=float, help="Maximum distance threshold delta new")
parser.add_argument("--gamma", default=3, type=float, help="Parameter gamma for overlapped speech penalty")
parser.add_argument("--beta", default=10, type=float, help="Parameter beta for overlapped speech penalty")
parser.add_argument("--max-speakers", default=20, type=int, help="Maximum number of identifiable speakers")
parser.add_argument("--batch-size", default=32, type=int, help="For segmentation and embedding pre-calculation")
parser.add_argument("--output", type=str, help="Output directory to store the RTTMs. Defaults to `root`")
args = parser.parse_args()

args.root = Path(args.root)
args.reference = Path(args.reference)
args.output = args.root if args.output is None else Path(args.output)
args.output.mkdir(parents=True, exist_ok=True)

assert args.root.is_dir(), "Root argument must be a directory"
assert args.reference.is_dir(), "Reference argument must be a directory"
assert args.output.is_dir(), "Output argument must be a directory"

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
)
pipeline = OnlineSpeakerDiarization(config)

# Run inference
audio_files = list(args.root.expanduser().iterdir())
pbar = tqdm(total=len(audio_files), unit="file")
chunk_loader = src.ChunkLoader(pipeline.sample_rate, pipeline.duration, config.step)
for filepath in audio_files:
    pbar.set_description(f"Processing {filepath.stem}")
    num_chunks = chunk_loader.num_chunks(filepath)
    # TODO run fully online if batch_size < 2
    pipeline.from_file(filepath, batch_size=args.batch_size, verbose=True).pipe(
        dops.progress(f"Streaming {filepath.stem}", total=num_chunks, leave=False)
    ).subscribe(
        RTTMWriter(path=args.output / f"{filepath.stem}.rttm")
    )
    pbar.update()
pbar.close()

# Run evaluation
metric = DiarizationErrorRate(collar=0, skip_overlap=False)
for ref_path in args.reference.iterdir():
    ref = load_rttm(ref_path).popitem()[1]
    hyp = load_rttm(args.output / ref_path.name).popitem()[1]
print(f"Diarization Error Rate: {100 * abs(metric):.1f}")
